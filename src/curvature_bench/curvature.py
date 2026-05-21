from __future__ import annotations

import time
from typing import Any

import torch
from torch import nn
from torch.nn.utils import parameters_to_vector, vector_to_parameters

try:
    from torch.func import functional_call
except ImportError:
    from torch.nn.utils.stateless import functional_call


def trainable_named_parameters(model: nn.Module) -> list[tuple[str, torch.nn.Parameter]]:
    return [(name, p) for name, p in model.named_parameters() if p.requires_grad]


def num_trainable_parameters(model: nn.Module) -> int:
    return sum(p.numel() for _, p in trainable_named_parameters(model))


def flatten_trainable_parameters(model: nn.Module) -> torch.Tensor:
    params = [p for _, p in trainable_named_parameters(model)]

    if len(params) == 0:
        raise RuntimeError("Model has no trainable parameters.")

    return parameters_to_vector(params)


def assign_trainable_parameters(model: nn.Module, theta: torch.Tensor) -> None:
    params = [p for _, p in trainable_named_parameters(model)]

    if len(params) == 0:
        raise RuntimeError("Model has no trainable parameters.")

    with torch.no_grad():
        vector_to_parameters(theta.detach(), params)


def _parameter_metadata(
    named_params: list[tuple[str, torch.nn.Parameter]],
) -> list[tuple[str, torch.Size, int]]:
    return [(name, p.shape, p.numel()) for name, p in named_params]


def _unflatten_to_param_dict(
    theta: torch.Tensor,
    metadata: list[tuple[str, torch.Size, int]],
) -> dict[str, torch.Tensor]:
    out: dict[str, torch.Tensor] = {}
    offset = 0

    for name, shape, n in metadata:
        out[name] = theta[offset : offset + n].view(shape)
        offset += n

    return out


def compute_full_hessian(
    model: nn.Module,
    criterion: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    max_params: int = 10_000,
    vectorize: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    named_params = trainable_named_parameters(model)
    params = [p for _, p in named_params]
    n_params = sum(p.numel() for p in params)

    if n_params == 0:
        raise RuntimeError("Cannot compute Hessian because model has no trainable parameters.")

    if n_params > max_params:
        raise RuntimeError(
            f"Full Hessian requested for {n_params:,} parameters, but max_params={max_params:,}. "
            "Use a smaller model or increase max_params intentionally."
        )

    theta0 = parameters_to_vector(params).detach().clone().requires_grad_(True)
    metadata = _parameter_metadata(named_params)
    buffers = dict(model.named_buffers())

    def loss_from_theta(theta: torch.Tensor) -> torch.Tensor:
        param_dict = _unflatten_to_param_dict(theta, metadata)

        state_dict: dict[str, torch.Tensor] = {}
        state_dict.update(buffers)
        state_dict.update(param_dict)

        logits = functional_call(model, state_dict, (x,))
        return criterion(logits, y)

    loss = loss_from_theta(theta0)
    grad = torch.autograd.grad(loss, theta0, create_graph=True)[0]

    hessian = torch.autograd.functional.hessian(
        loss_from_theta,
        theta0,
        vectorize=vectorize,
    )

    return loss, grad, hessian


def exact_newton_step(
    model: nn.Module,
    criterion: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    lr: float = 1.0,
    damping: float = 1e-3,
    max_params: int = 10_000,
    vectorize: bool = True,
) -> dict[str, Any]:
    n_params = num_trainable_parameters(model)

    if n_params > max_params:
        raise RuntimeError(
            f"Exact Newton requested for {n_params:,} parameters, but max_params={max_params:,}. "
            "Use a tiny MLP/subset for exact Newton."
        )

    total_start = time.perf_counter()

    hessian_start = time.perf_counter()
    loss, grad, hessian = compute_full_hessian(
        model=model,
        criterion=criterion,
        x=x,
        y=y,
        max_params=max_params,
        vectorize=vectorize,
    )
    hessian_ms = (time.perf_counter() - hessian_start) * 1000.0

    eye = torch.eye(
        hessian.size(0),
        device=hessian.device,
        dtype=hessian.dtype,
    )

    damped_hessian = hessian + damping * eye

    solve_start = time.perf_counter()

    try:
        step = torch.linalg.solve(damped_hessian, grad)
        solve_status = "solve"
    except RuntimeError:
        step = torch.linalg.lstsq(damped_hessian, grad).solution
        solve_status = "lstsq"

    linear_solve_ms = (time.perf_counter() - solve_start) * 1000.0

    theta = flatten_trainable_parameters(model).detach()
    theta_new = theta - lr * step.detach()
    assign_trainable_parameters(model, theta_new)

    total_ms = (time.perf_counter() - total_start) * 1000.0

    return {
        "loss": float(loss.detach().item()),
        "num_params": int(n_params),
        "grad_norm": float(torch.linalg.vector_norm(grad.detach()).item()),
        "step_norm": float(torch.linalg.vector_norm(step.detach()).item()),
        "hessian_ms": float(hessian_ms),
        "linear_solve_ms": float(linear_solve_ms),
        "newton_total_ms": float(total_ms),
        "solve_status": solve_status,
        "damping": float(damping),
        "lr": float(lr),
    }


def hessian_eigenvalue_summary(
    hessian: torch.Tensor,
) -> dict[str, float]:
    if hessian.ndim != 2 or hessian.size(0) != hessian.size(1):
        raise ValueError("hessian must be a square matrix.")

    eigvals = torch.linalg.eigvalsh(hessian.detach())

    min_eig = float(eigvals.min().item())
    max_eig = float(eigvals.max().item())

    abs_eig = eigvals.abs()
    nonzero = abs_eig[abs_eig > 1e-12]

    if len(nonzero) > 0:
        condition_estimate = float((nonzero.max() / nonzero.min()).item())
    else:
        condition_estimate = float("inf")

    num_negative = int((eigvals < 0).sum().item())
    num_positive = int((eigvals > 0).sum().item())
    num_near_zero = int((eigvals.abs() <= 1e-12).sum().item())

    return {
        "min_eig": min_eig,
        "max_eig": max_eig,
        "condition_estimate": condition_estimate,
        "num_negative": num_negative,
        "num_positive": num_positive,
        "num_near_zero": num_near_zero,
    }
