from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass
class AverageMeter:
    """
    Track a running average.

    Useful for loss, accuracy, timing, memory, etc.
    """

    name: str
    total: float = 0.0
    count: int = 0

    def update(self, value: float, n: int = 1) -> None:
        self.total += float(value) * n
        self.count += int(n)

    @property
    def avg(self) -> float:
        if self.count == 0:
            return 0.0
        return self.total / self.count

    def reset(self) -> None:
        self.total = 0.0
        self.count = 0


@torch.no_grad()
def classification_accuracy(
    logits: torch.Tensor,
    targets: torch.Tensor,
) -> float:
    """
    Compute top-1 classification accuracy.
    """
    preds = logits.argmax(dim=1)
    correct = (preds == targets).sum().item()
    total = targets.numel()

    if total == 0:
        return 0.0

    return correct / total


@torch.no_grad()
def topk_accuracy(
    logits: torch.Tensor,
    targets: torch.Tensor,
    topk: tuple[int, ...] = (1,),
) -> dict[str, float]:
    """
    Compute top-k accuracies.

    Example:
      topk_accuracy(logits, y, topk=(1, 5))
    """
    max_k = max(topk)
    batch_size = targets.size(0)

    _, pred = logits.topk(max_k, dim=1)
    pred = pred.t()

    correct = pred.eq(targets.view(1, -1).expand_as(pred))

    results = {}

    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum().item()
        results[f"top{k}_acc"] = correct_k / batch_size

    return results


@torch.no_grad()
def parameter_norm(model: nn.Module) -> float:
    """
    L2 norm of trainable parameters.
    """
    total = torch.tensor(0.0)

    for p in model.parameters():
        if p.requires_grad:
            total = total.to(p.device)
            total += torch.sum(p.detach() ** 2)

    return float(torch.sqrt(total).item())


@torch.no_grad()
def gradient_norm(model: nn.Module) -> float:
    """
    L2 norm of all available gradients.
    """
    total = torch.tensor(0.0)
    found_grad = False

    for p in model.parameters():
        if p.requires_grad and p.grad is not None:
            found_grad = True
            total = total.to(p.grad.device)
            total += torch.sum(p.grad.detach() ** 2)

    if not found_grad:
        return 0.0

    return float(torch.sqrt(total).item())


@torch.no_grad()
def update_norm_before_after(
    before: list[torch.Tensor],
    model: nn.Module,
) -> float:
    """
    Compute ||theta_after - theta_before||.

    Usage:
      before = clone_trainable_parameters(model)
      optimizer.step()
      update_norm = update_norm_before_after(before, model)
    """
    total = torch.tensor(0.0)
    idx = 0

    for p in model.parameters():
        if p.requires_grad:
            diff = p.detach() - before[idx].to(p.device)
            total = total.to(p.device)
            total += torch.sum(diff ** 2)
            idx += 1

    return float(torch.sqrt(total).item())


@torch.no_grad()
def clone_trainable_parameters(model: nn.Module) -> list[torch.Tensor]:
    """
    Clone trainable parameters before an optimizer step.

    This is useful if you want update_norm.
    """
    return [p.detach().clone().cpu() for p in model.parameters() if p.requires_grad]


def optimizer_state_numel(optimizer) -> int:
    """
    Count number of scalar values stored in optimizer state.

    This is useful for comparing SGD, Adam, AdamW, L-BFGS, etc.
    """
    if optimizer is None:
        return 0

    total = 0

    for state in optimizer.state.values():
        for value in state.values():
            if torch.is_tensor(value):
                total += value.numel()
            elif isinstance(value, list):
                for item in value:
                    if torch.is_tensor(item):
                        total += item.numel()

    return int(total)


def optimizer_state_memory_mb(optimizer) -> float:
    """
    Estimate optimizer state memory in MB.
    """
    if optimizer is None:
        return 0.0

    total_bytes = 0

    for state in optimizer.state.values():
        for value in state.values():
            if torch.is_tensor(value):
                total_bytes += value.numel() * value.element_size()
            elif isinstance(value, list):
                for item in value:
                    if torch.is_tensor(item):
                        total_bytes += item.numel() * item.element_size()

    return total_bytes / 1024**2


def model_parameter_memory_mb(model: nn.Module) -> float:
    """
    Estimate model parameter memory in MB.
    """
    total_bytes = 0

    for p in model.parameters():
        total_bytes += p.numel() * p.element_size()

    return total_bytes / 1024**2


def safe_float(value) -> float | None:
    """
    Convert to float when possible.

    Useful before writing JSON summaries.
    """
    if value is None:
        return None

    try:
        return float(value)
    except Exception:
        return None