from __future__ import annotations

import json
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn
from tqdm import tqdm

from curvature_bench.curvature import exact_newton_step
from curvature_bench.data import build_dataloaders
from curvature_bench.models import build_model, count_parameters
from curvature_bench.optimizers import DenseBFGS, build_optimizer


def set_seed(seed: int) -> None:
    """
    Make runs more reproducible.

    This will not guarantee perfect reproducibility across all hardware,
    but it is enough for the first version of the benchmark.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device(runtime_cfg: dict[str, Any]) -> torch.device:
    """
    Pick the runtime device.

    Config example:

    runtime:
      device: auto

    or:

    runtime:
      device: cuda

    or:

    runtime:
      device: cpu
    """
    requested = runtime_cfg.get("device", "auto")

    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    return torch.device(requested)


def sync_device(device: torch.device) -> None:
    """
    Synchronize before/after timing blocks.

    CUDA operations are asynchronous by default. Without synchronization,
    Python timers can under-report GPU time.

    For CPU, this does nothing.
    For MPS, PyTorch synchronization support is more limited, so we keep it simple.
    """
    if device.type == "cuda":
        torch.cuda.synchronize()


def current_peak_memory_mb(device: torch.device) -> float:
    """
    Return current peak allocated memory in MB.

    This is most meaningful on CUDA.
    For CPU/MPS, return 0 for now.
    """
    if device.type == "cuda":
        return torch.cuda.max_memory_allocated(device) / 1024**2

    return 0.0


def reset_peak_memory(device: torch.device) -> None:
    """
    Reset CUDA peak memory stats before a run.
    """
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)


def make_criterion(cfg: dict[str, Any]) -> nn.Module:
    """
    Build the loss function.

    For now, classification uses CrossEntropyLoss.
    Later, this can become a registry too.
    """
    loss_name = cfg.get("loss", "cross_entropy").lower()

    if loss_name == "cross_entropy":
        return nn.CrossEntropyLoss()

    raise ValueError(f"Unsupported loss function: {loss_name}")


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader,
    criterion: nn.Module,
    device: torch.device,
) -> dict[str, float]:
    """
    Evaluate the model on the validation set.

    This returns loss and accuracy for classification.
    Later, this can be generalized for language modeling, regression, etc.
    """
    model.eval()

    total_loss = 0.0
    total_correct = 0
    total_seen = 0

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)

        logits = model(x)
        loss = criterion(logits, y)

        batch_size = x.size(0)
        total_loss += float(loss.item()) * batch_size
        total_correct += int((logits.argmax(dim=1) == y).sum().item())
        total_seen += batch_size

    if total_seen == 0:
        return {
            "val_loss": float("nan"),
            "val_acc": float("nan"),
        }

    return {
        "val_loss": total_loss / total_seen,
        "val_acc": total_correct / total_seen,
    }


def get_current_lr(optimizer) -> float | None:
    """
    Read the learning rate from the first parameter group.

    Newton is special because it may not have a normal torch optimizer.
    """
    if optimizer is None:
        return None

    if not hasattr(optimizer, "param_groups"):
        return None

    if len(optimizer.param_groups) == 0:
        return None

    return float(optimizer.param_groups[0].get("lr", 0.0))


def is_closure_optimizer(optimizer) -> bool:
    """
    Some optimizers need a closure.

    PyTorch LBFGS needs a closure because it may reevaluate the loss
    multiple times during one optimizer step.

    DenseBFGS also uses a closure in this benchmark implementation.
    """
    return isinstance(optimizer, torch.optim.LBFGS) or isinstance(optimizer, DenseBFGS)


def run_standard_step(
    model: nn.Module,
    optimizer,
    criterion: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    device: torch.device,
) -> dict[str, Any]:
    """
    Run one normal first-order style optimizer step.

    This path is for:
      - SGD with momentum
      - Adam
      - AdamW
      - future optimizers that do not require closures

    It separately times:
      - forward pass
      - backward pass
      - optimizer step
    """
    model.train()

    step_info: dict[str, Any] = {
        "forward_ms": 0.0,
        "backward_ms": 0.0,
        "optimizer_ms": 0.0,
        "closure_calls": 0,
        "train_loss": None,
    }

    optimizer.zero_grad(set_to_none=True)

    sync_device(device)
    start = time.perf_counter()
    logits = model(x)
    loss = criterion(logits, y)
    sync_device(device)
    step_info["forward_ms"] = (time.perf_counter() - start) * 1000.0

    sync_device(device)
    start = time.perf_counter()
    loss.backward()
    sync_device(device)
    step_info["backward_ms"] = (time.perf_counter() - start) * 1000.0

    sync_device(device)
    start = time.perf_counter()
    optimizer.step()
    sync_device(device)
    step_info["optimizer_ms"] = (time.perf_counter() - start) * 1000.0

    step_info["train_loss"] = float(loss.detach().item())

    return step_info


def run_closure_step(
    model: nn.Module,
    optimizer,
    criterion: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    device: torch.device,
) -> dict[str, Any]:
    """
    Run one closure-based optimizer step.

    This path is for:
      - L-BFGS
      - dense BFGS

    Important detail:
    L-BFGS may call the closure multiple times inside one optimizer step.
    That means one "step" can contain several forward/backward passes.

    So we track:
      - total forward time across closure calls
      - total backward time across closure calls
      - number of closure calls
      - optimizer overhead outside forward/backward
    """
    model.train()

    closure_stats: dict[str, Any] = {
        "forward_ms": 0.0,
        "backward_ms": 0.0,
        "closure_calls": 0,
        "last_loss": None,
    }

    def closure():
        optimizer.zero_grad(set_to_none=True)

        sync_device(device)
        start = time.perf_counter()
        logits = model(x)
        loss = criterion(logits, y)
        sync_device(device)
        closure_stats["forward_ms"] += (time.perf_counter() - start) * 1000.0

        sync_device(device)
        start = time.perf_counter()
        loss.backward()
        sync_device(device)
        closure_stats["backward_ms"] += (time.perf_counter() - start) * 1000.0

        closure_stats["closure_calls"] += 1
        closure_stats["last_loss"] = float(loss.detach().item())

        return loss

    sync_device(device)
    opt_start = time.perf_counter()
    optimizer.step(closure)
    sync_device(device)
    opt_total_ms = (time.perf_counter() - opt_start) * 1000.0

    forward_ms = closure_stats["forward_ms"]
    backward_ms = closure_stats["backward_ms"]

    # Anything not explained by forward/backward is counted as optimizer overhead.
    optimizer_ms = max(opt_total_ms - forward_ms - backward_ms, 0.0)

    return {
        "forward_ms": forward_ms,
        "backward_ms": backward_ms,
        "optimizer_ms": optimizer_ms,
        "closure_calls": closure_stats["closure_calls"],
        "train_loss": closure_stats["last_loss"],
    }


def run_newton_step(
    model: nn.Module,
    criterion: nn.Module,
    optimizer_cfg: dict[str, Any],
    x: torch.Tensor,
    y: torch.Tensor,
    device: torch.device,
) -> dict[str, Any]:
    """
    Run one exact Newton step.

    Exact Newton is special because it is not a normal torch optimizer here.
    It needs:
      - model
      - criterion
      - current batch
      - damping
      - parameter vector
      - Hessian computation
      - linear solve

    This should only be used on tiny models.
    """
    model.train()

    sync_device(device)
    start = time.perf_counter()

    info = exact_newton_step(
        model=model,
        criterion=criterion,
        x=x,
        y=y,
        lr=float(optimizer_cfg.get("lr", 1.0)),
        damping=float(optimizer_cfg.get("damping", 1e-3)),
        max_params=int(optimizer_cfg.get("max_params", 10_000)),
    )

    sync_device(device)
    total_ms = (time.perf_counter() - start) * 1000.0

    return {
        "forward_ms": 0.0,
        "backward_ms": 0.0,
        "optimizer_ms": total_ms,
        "closure_calls": 0,
        "train_loss": info["loss"],
        "hessian_ms": info.get("hessian_ms", 0.0),
        "linear_solve_ms": info.get("linear_solve_ms", 0.0),
        "newton_total_ms": info.get("newton_total_ms", total_ms),
    }


def summarize_history(
    step_df: pd.DataFrame,
    epoch_df: pd.DataFrame,
    run_cfg: dict[str, Any],
    total_wall_time_s: float,
    num_params: int,
) -> dict[str, Any]:
    """
    Build a compact summary for one experiment.

    This is what you will compare across optimizers.
    """
    optimizer_name = run_cfg["optimizer"]["name"]

    target_acc = run_cfg.get("training", {}).get("target_acc", None)
    reached_target = None

    if target_acc is not None and len(epoch_df) > 0:
        hit = epoch_df[epoch_df["val_acc"] >= float(target_acc)]

        if len(hit) > 0:
            first_hit = hit.iloc[0]
            reached_target = {
                "target_acc": float(target_acc),
                "epoch": int(first_hit["epoch"]),
                "global_step": int(first_hit["global_step"]),
                "wall_time_s": float(first_hit["wall_time_s"]),
                "val_acc": float(first_hit["val_acc"]),
            }

    final_val_acc = None
    final_val_loss = None
    best_val_acc = None
    best_val_loss = None

    if len(epoch_df) > 0:
        final_val_acc = float(epoch_df.iloc[-1]["val_acc"])
        final_val_loss = float(epoch_df.iloc[-1]["val_loss"])
        best_val_acc = float(epoch_df["val_acc"].max())
        best_val_loss = float(epoch_df["val_loss"].min())

    summary = {
        "run_name": run_cfg.get("run_name", None),
        "optimizer": optimizer_name,
        "model": run_cfg["model"]["name"],
        "data": run_cfg["data"]["name"],
        "num_params": int(num_params),
        "epochs": int(run_cfg.get("training", {}).get("epochs", 0)),
        "global_steps": int(step_df["global_step"].max() + 1) if len(step_df) > 0 else 0,
        "total_wall_time_s": float(total_wall_time_s),
        "final_val_acc": final_val_acc,
        "final_val_loss": final_val_loss,
        "best_val_acc": best_val_acc,
        "best_val_loss": best_val_loss,
        "reached_target": reached_target,
        "avg_step_ms": float(step_df["step_ms"].mean()) if len(step_df) > 0 else None,
        "avg_forward_ms": float(step_df["forward_ms"].mean()) if len(step_df) > 0 else None,
        "avg_backward_ms": float(step_df["backward_ms"].mean()) if len(step_df) > 0 else None,
        "avg_optimizer_ms": float(step_df["optimizer_ms"].mean()) if len(step_df) > 0 else None,
        "avg_closure_calls": float(step_df["closure_calls"].mean()) if len(step_df) > 0 else None,
        "peak_mem_mb": float(step_df["peak_mem_mb"].max()) if len(step_df) > 0 else None,
    }

    if "hessian_ms" in step_df.columns:
        summary["avg_hessian_ms"] = float(step_df["hessian_ms"].fillna(0.0).mean())

    if "linear_solve_ms" in step_df.columns:
        summary["avg_linear_solve_ms"] = float(step_df["linear_solve_ms"].fillna(0.0).mean())

    return summary


def train_run(cfg: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """
    Run one complete experiment.

    This is the main entry point for:
      - scripts/run_experiment.py
      - testbed/mlp_testbed.py
      - notebooks

    The config controls everything:
      - dataset
      - model
      - optimizer
      - runtime
      - training length
      - output directory

    This function returns:
      - step-level history dataframe
      - epoch-level validation dataframe
      - summary dictionary
    """
    seed = int(cfg.get("seed", 123))
    set_seed(seed)

    runtime_cfg = cfg.get("runtime", {})
    device = get_device(runtime_cfg)

    output_dir = Path(cfg.get("output_dir", "results/debug"))
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / "resolved_config.json", "w") as f:
        json.dump(cfg, f, indent=2)

    train_loader, val_loader = build_dataloaders(cfg["data"])

    model = build_model(cfg["model"]).to(device)
    criterion = make_criterion(cfg.get("training", {})).to(device)

    optimizer_cfg = cfg["optimizer"]
    optimizer_name = optimizer_cfg["name"].lower()

    optimizer = build_optimizer(optimizer_cfg, model.parameters())

    num_params = count_parameters(model)

    training_cfg = cfg.get("training", {})
    epochs = int(training_cfg.get("epochs", 1))
    log_every = int(training_cfg.get("log_every", 10))
    limit_train_batches = training_cfg.get("limit_train_batches", None)
    eval_every_epoch = bool(training_cfg.get("eval_every_epoch", True))
    save_checkpoint = bool(training_cfg.get("save_checkpoint", False))

    reset_peak_memory(device)

    step_records: list[dict[str, Any]] = []
    epoch_records: list[dict[str, Any]] = []

    global_step = 0
    run_start = time.perf_counter()

    print(f"Device: {device}")
    print(f"Model: {cfg['model']['name']}")
    print(f"Optimizer: {optimizer_name}")
    print(f"Dataset: {cfg['data']['name']}")
    print(f"Parameters: {num_params:,}")

    for epoch in range(epochs):
        model.train()

        epoch_train_losses: list[float] = []

        iterator = enumerate(train_loader)

        if limit_train_batches is not None:
            # This is fine for small testbed runs.
            # Later, we can avoid materializing the list.
            iterator = list(iterator)[: int(limit_train_batches)]

        progress = tqdm(
            iterator,
            desc=f"{optimizer_name} | epoch {epoch + 1}/{epochs}",
            leave=False,
        )

        for batch_idx, (x, y) in progress:
            x = x.to(device)
            y = y.to(device)

            sync_device(device)
            step_start = time.perf_counter()

            if optimizer_name == "newton":
                step_info = run_newton_step(
                    model=model,
                    criterion=criterion,
                    optimizer_cfg=optimizer_cfg,
                    x=x,
                    y=y,
                    device=device,
                )

            elif is_closure_optimizer(optimizer):
                step_info = run_closure_step(
                    model=model,
                    optimizer=optimizer,
                    criterion=criterion,
                    x=x,
                    y=y,
                    device=device,
                )

            else:
                step_info = run_standard_step(
                    model=model,
                    optimizer=optimizer,
                    criterion=criterion,
                    x=x,
                    y=y,
                    device=device,
                )

            sync_device(device)
            step_ms = (time.perf_counter() - step_start) * 1000.0
            wall_time_s = time.perf_counter() - run_start

            train_loss = step_info.get("train_loss", None)

            if train_loss is not None:
                epoch_train_losses.append(float(train_loss))

            record = {
                "epoch": epoch,
                "batch_idx": batch_idx,
                "global_step": global_step,
                "optimizer": optimizer_name,
                "model": cfg["model"]["name"],
                "data": cfg["data"]["name"],
                "lr": get_current_lr(optimizer),
                "train_loss": train_loss,
                "step_ms": step_ms,
                "forward_ms": float(step_info.get("forward_ms", 0.0)),
                "backward_ms": float(step_info.get("backward_ms", 0.0)),
                "optimizer_ms": float(step_info.get("optimizer_ms", 0.0)),
                "closure_calls": int(step_info.get("closure_calls", 0)),
                "peak_mem_mb": current_peak_memory_mb(device),
                "wall_time_s": wall_time_s,
            }

            # Optional Newton-specific fields.
            if "hessian_ms" in step_info:
                record["hessian_ms"] = float(step_info["hessian_ms"])

            if "linear_solve_ms" in step_info:
                record["linear_solve_ms"] = float(step_info["linear_solve_ms"])

            if "newton_total_ms" in step_info:
                record["newton_total_ms"] = float(step_info["newton_total_ms"])

            step_records.append(record)

            if global_step % log_every == 0:
                progress.set_postfix(
                    {
                        "loss": train_loss,
                        "step_ms": f"{step_ms:.2f}",
                    }
                )

            global_step += 1

        if eval_every_epoch:
            val_metrics = evaluate(
                model=model,
                loader=val_loader,
                criterion=criterion,
                device=device,
            )

            epoch_wall_time_s = time.perf_counter() - run_start
            mean_train_loss = (
                float(np.mean(epoch_train_losses))
                if len(epoch_train_losses) > 0
                else float("nan")
            )

            epoch_record = {
                "epoch": epoch,
                "global_step": global_step,
                "wall_time_s": epoch_wall_time_s,
                "train_loss_mean": mean_train_loss,
                "val_loss": float(val_metrics["val_loss"]),
                "val_acc": float(val_metrics["val_acc"]),
            }

            epoch_records.append(epoch_record)

            print(
                f"Epoch {epoch + 1}/{epochs} | "
                f"train_loss={mean_train_loss:.4f} | "
                f"val_loss={val_metrics['val_loss']:.4f} | "
                f"val_acc={val_metrics['val_acc']:.4f} | "
                f"time={epoch_wall_time_s:.2f}s"
            )

    total_wall_time_s = time.perf_counter() - run_start

    step_df = pd.DataFrame(step_records)
    epoch_df = pd.DataFrame(epoch_records)

    step_df.to_csv(output_dir / "history_steps.csv", index=False)
    epoch_df.to_csv(output_dir / "history_epochs.csv", index=False)

    summary = summarize_history(
        step_df=step_df,
        epoch_df=epoch_df,
        run_cfg=cfg,
        total_wall_time_s=total_wall_time_s,
        num_params=num_params,
    )

    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    if save_checkpoint:
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "config": cfg,
                "summary": summary,
            },
            output_dir / "checkpoint.pt",
        )

    return step_df, epoch_df, summary