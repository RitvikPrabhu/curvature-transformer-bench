from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def _resolve_history_paths(path: str | Path) -> tuple[Path, Path | None]:
    path = Path(path)

    if path.is_dir():
        step_csv = path / "history_steps.csv"
        epoch_csv = path / "history_epochs.csv"
    else:
        step_csv = path
        epoch_csv = path.parent / "history_epochs.csv"

    if not step_csv.exists():
        raise FileNotFoundError(f"Could not find step history CSV: {step_csv}")

    if not epoch_csv.exists():
        epoch_csv = None

    return step_csv, epoch_csv


def plot_train_loss_vs_step(
    step_df: pd.DataFrame,
    out_dir: str | Path,
    title: str | None = None,
) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    out_path = out_dir / "train_loss_vs_step.png"

    plt.figure()
    plt.plot(step_df["global_step"], step_df["train_loss"])
    plt.xlabel("Training step")
    plt.ylabel("Training loss")
    plt.title(title or "Training loss vs step")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()

    return out_path


def plot_train_loss_vs_time(
    step_df: pd.DataFrame,
    out_dir: str | Path,
    title: str | None = None,
) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    out_path = out_dir / "train_loss_vs_time.png"

    plt.figure()
    plt.plot(step_df["wall_time_s"], step_df["train_loss"])
    plt.xlabel("Wall-clock time (s)")
    plt.ylabel("Training loss")
    plt.title(title or "Training loss vs wall-clock time")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()

    return out_path


def plot_val_acc_vs_time(
    epoch_df: pd.DataFrame,
    out_dir: str | Path,
    title: str | None = None,
) -> Path | None:
    if epoch_df is None or len(epoch_df) == 0:
        return None

    if "val_acc" not in epoch_df.columns:
        return None

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    out_path = out_dir / "val_acc_vs_time.png"

    plt.figure()
    plt.plot(epoch_df["wall_time_s"], epoch_df["val_acc"], marker="o")
    plt.xlabel("Wall-clock time (s)")
    plt.ylabel("Validation accuracy")
    plt.title(title or "Validation accuracy vs wall-clock time")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()

    return out_path


def plot_val_loss_vs_time(
    epoch_df: pd.DataFrame,
    out_dir: str | Path,
    title: str | None = None,
) -> Path | None:
    if epoch_df is None or len(epoch_df) == 0:
        return None

    if "val_loss" not in epoch_df.columns:
        return None

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    out_path = out_dir / "val_loss_vs_time.png"

    plt.figure()
    plt.plot(epoch_df["wall_time_s"], epoch_df["val_loss"], marker="o")
    plt.xlabel("Wall-clock time (s)")
    plt.ylabel("Validation loss")
    plt.title(title or "Validation loss vs wall-clock time")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()

    return out_path


def plot_time_breakdown(
    step_df: pd.DataFrame,
    out_dir: str | Path,
    title: str | None = None,
) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    columns = ["forward_ms", "backward_ms", "optimizer_ms"]
    available = [c for c in columns if c in step_df.columns]

    if len(available) == 0:
        raise ValueError("No timing columns found for time breakdown plot.")

    means = step_df[available].mean()

    out_path = out_dir / "time_breakdown.png"

    plt.figure()
    plt.bar([c.replace("_ms", "") for c in available], means.values)
    plt.ylabel("Average time per step (ms)")
    plt.title(title or "Average step-time breakdown")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()

    return out_path


def plot_optimizer_fraction(
    step_df: pd.DataFrame,
    out_dir: str | Path,
    title: str | None = None,
) -> Path | None:
    required = {"optimizer_ms", "step_ms"}

    if not required.issubset(step_df.columns):
        return None

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = step_df.copy()
    df = df[df["step_ms"] > 0]

    if len(df) == 0:
        return None

    df["optimizer_fraction"] = df["optimizer_ms"] / df["step_ms"]

    out_path = out_dir / "optimizer_fraction_vs_step.png"

    plt.figure()
    plt.plot(df["global_step"], df["optimizer_fraction"])
    plt.xlabel("Training step")
    plt.ylabel("Optimizer time / step time")
    plt.title(title or "Optimizer-time fraction vs step")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()

    return out_path


def plot_closure_calls(
    step_df: pd.DataFrame,
    out_dir: str | Path,
    title: str | None = None,
) -> Path | None:
    if "closure_calls" not in step_df.columns:
        return None

    if step_df["closure_calls"].max() <= 0:
        return None

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    out_path = out_dir / "closure_calls_vs_step.png"

    plt.figure()
    plt.plot(step_df["global_step"], step_df["closure_calls"])
    plt.xlabel("Training step")
    plt.ylabel("Closure calls")
    plt.title(title or "Closure calls per optimizer step")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()

    return out_path


def plot_newton_breakdown(
    step_df: pd.DataFrame,
    out_dir: str | Path,
    title: str | None = None,
) -> Path | None:
    required = {"hessian_ms", "linear_solve_ms"}

    if not required.issubset(step_df.columns):
        return None

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    means = step_df[["hessian_ms", "linear_solve_ms"]].fillna(0.0).mean()

    out_path = out_dir / "newton_breakdown.png"

    plt.figure()
    plt.bar(["hessian", "linear_solve"], means.values)
    plt.ylabel("Average time per step (ms)")
    plt.title(title or "Newton step breakdown")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()

    return out_path


def plot_history(
    path: str | Path,
    out_dir: str | Path | None = None,
) -> list[Path]:
    step_csv, epoch_csv = _resolve_history_paths(path)

    if out_dir is None:
        out_dir = step_csv.parent / "plots"

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    step_df = pd.read_csv(step_csv)

    if epoch_csv is not None:
        epoch_df = pd.read_csv(epoch_csv)
    else:
        epoch_df = pd.DataFrame()

    optimizer = (
        str(step_df["optimizer"].iloc[0])
        if "optimizer" in step_df.columns and len(step_df) > 0
        else "run"
    )

    title_prefix = optimizer.upper()

    outputs: list[Path] = []

    outputs.append(
        plot_train_loss_vs_step(
            step_df,
            out_dir,
            title=f"{title_prefix}: train loss vs step",
        )
    )

    outputs.append(
        plot_train_loss_vs_time(
            step_df,
            out_dir,
            title=f"{title_prefix}: train loss vs time",
        )
    )

    val_acc_path = plot_val_acc_vs_time(
        epoch_df,
        out_dir,
        title=f"{title_prefix}: validation accuracy vs time",
    )
    if val_acc_path is not None:
        outputs.append(val_acc_path)

    val_loss_path = plot_val_loss_vs_time(
        epoch_df,
        out_dir,
        title=f"{title_prefix}: validation loss vs time",
    )
    if val_loss_path is not None:
        outputs.append(val_loss_path)

    outputs.append(
        plot_time_breakdown(
            step_df,
            out_dir,
            title=f"{title_prefix}: time breakdown",
        )
    )

    optimizer_fraction_path = plot_optimizer_fraction(
        step_df,
        out_dir,
        title=f"{title_prefix}: optimizer-time fraction",
    )
    if optimizer_fraction_path is not None:
        outputs.append(optimizer_fraction_path)

    closure_path = plot_closure_calls(
        step_df,
        out_dir,
        title=f"{title_prefix}: closure calls",
    )
    if closure_path is not None:
        outputs.append(closure_path)

    newton_path = plot_newton_breakdown(
        step_df,
        out_dir,
        title=f"{title_prefix}: Newton breakdown",
    )
    if newton_path is not None:
        outputs.append(newton_path)

    return outputs


def load_summaries(result_dirs: list[str | Path]) -> pd.DataFrame:
    records = []

    for result_dir in result_dirs:
        result_dir = Path(result_dir)
        summary_path = result_dir / "summary.json"

        if not summary_path.exists():
            continue

        with open(summary_path, "r") as f:
            record = json.load(f)

        record["result_dir"] = str(result_dir)
        records.append(record)

    return pd.DataFrame(records)


def plot_summary_time_to_quality(
    summary_df: pd.DataFrame,
    out_dir: str | Path,
) -> Path | None:
    if len(summary_df) == 0:
        return None

    if "optimizer" not in summary_df.columns or "total_wall_time_s" not in summary_df.columns:
        return None

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    out_path = out_dir / "summary_total_wall_time.png"

    plt.figure()
    plt.bar(summary_df["optimizer"], summary_df["total_wall_time_s"])
    plt.xlabel("Optimizer")
    plt.ylabel("Total wall-clock time (s)")
    plt.title("Total wall-clock time by optimizer")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()

    return out_path


def plot_summary_accuracy(
    summary_df: pd.DataFrame,
    out_dir: str | Path,
) -> Path | None:
    if len(summary_df) == 0:
        return None

    if "optimizer" not in summary_df.columns or "best_val_acc" not in summary_df.columns:
        return None

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    out_path = out_dir / "summary_best_val_acc.png"

    plt.figure()
    plt.bar(summary_df["optimizer"], summary_df["best_val_acc"])
    plt.xlabel("Optimizer")
    plt.ylabel("Best validation accuracy")
    plt.title("Best validation accuracy by optimizer")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()

    return out_path
