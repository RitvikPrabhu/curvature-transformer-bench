from __future__ import annotations

import torch

from curvature_bench.registry import OPTIMIZER_REGISTRY


class DenseBFGS(torch.optim.Optimizer):
    def __init__(
        self,
        params,
        lr: float = 1.0,
        damping: float = 1e-6,
        max_params: int = 20_000,
    ) -> None:
        defaults = {
            "lr": lr,
            "damping": damping,
            "max_params": max_params,
        }
        super().__init__(params, defaults)

        self.H = None
        self.prev_params = None
        self.prev_grad = None

    def _params(self) -> list[torch.Tensor]:
        return [
            p
            for group in self.param_groups
            for p in group["params"]
            if p.requires_grad
        ]

    def _gather_flat_params(self) -> torch.Tensor:
        return torch.cat([p.detach().reshape(-1) for p in self._params()])

    def _gather_flat_grad(self) -> torch.Tensor:
        grads = []

        for p in self._params():
            if p.grad is None:
                grads.append(torch.zeros_like(p).reshape(-1))
            else:
                grads.append(p.grad.detach().reshape(-1))

        return torch.cat(grads)

    @torch.no_grad()
    def _set_flat_params(self, flat: torch.Tensor) -> None:
        offset = 0

        for p in self._params():
            n = p.numel()
            p.copy_(flat[offset : offset + n].view_as(p))
            offset += n

    def step(self, closure=None):
        if closure is None:
            raise RuntimeError("DenseBFGS requires a closure.")

        group = self.param_groups[0]
        lr = group["lr"]
        damping = group["damping"]
        max_params = group["max_params"]

        with torch.enable_grad():
            loss = closure()

        flat_params = self._gather_flat_params()
        flat_grad = self._gather_flat_grad()

        n = flat_params.numel()

        if n > max_params:
            raise RuntimeError(
                f"DenseBFGS requested with {n} parameters. "
                f"Limit is {max_params}. Use a tiny model."
            )

        if self.H is None:
            self.H = torch.eye(
                n,
                device=flat_params.device,
                dtype=flat_params.dtype,
            )

        if self.prev_params is not None and self.prev_grad is not None:
            s = flat_params - self.prev_params
            y = flat_grad - self.prev_grad

            ys = torch.dot(y, s)

            if ys > damping:
                rho = 1.0 / ys
                identity = torch.eye(
                    n,
                    device=flat_params.device,
                    dtype=flat_params.dtype,
                )

                sy_t = torch.outer(s, y)
                ys_t = torch.outer(y, s)
                ss_t = torch.outer(s, s)

                self.H = (
                    (identity - rho * sy_t)
                    @ self.H
                    @ (identity - rho * ys_t)
                    + rho * ss_t
                )

        direction = -self.H.mv(flat_grad)

        self.prev_params = flat_params.clone()
        self.prev_grad = flat_grad.clone()

        new_params = flat_params + lr * direction
        self._set_flat_params(new_params)

        return loss


@OPTIMIZER_REGISTRY.register("sgd_momentum")
def build_sgd_momentum(cfg: dict, params):
    return torch.optim.SGD(
        params,
        lr=cfg.get("lr", 1e-2),
        momentum=cfg.get("momentum", 0.9),
        weight_decay=cfg.get("weight_decay", 0.0),
    )


@OPTIMIZER_REGISTRY.register("adam")
def build_adam(cfg: dict, params):
    return torch.optim.Adam(
        params,
        lr=cfg.get("lr", 1e-3),
        betas=tuple(cfg.get("betas", [0.9, 0.999])),
        eps=cfg.get("eps", 1e-8),
        weight_decay=cfg.get("weight_decay", 0.0),
    )


@OPTIMIZER_REGISTRY.register("adamw")
def build_adamw(cfg: dict, params):
    return torch.optim.AdamW(
        params,
        lr=cfg.get("lr", 1e-3),
        betas=tuple(cfg.get("betas", [0.9, 0.999])),
        eps=cfg.get("eps", 1e-8),
        weight_decay=cfg.get("weight_decay", 1e-2),
    )


@OPTIMIZER_REGISTRY.register("lbfgs")
def build_lbfgs(cfg: dict, params):
    return torch.optim.LBFGS(
        params,
        lr=cfg.get("lr", 1.0),
        max_iter=cfg.get("max_iter", 5),
        max_eval=cfg.get("max_eval", None),
        tolerance_grad=cfg.get("tolerance_grad", 1e-7),
        tolerance_change=cfg.get("tolerance_change", 1e-9),
        history_size=cfg.get("history_size", 10),
        line_search_fn=cfg.get("line_search_fn", "strong_wolfe"),
    )


@OPTIMIZER_REGISTRY.register("bfgs")
def build_bfgs(cfg: dict, params):
    return DenseBFGS(
        params,
        lr=cfg.get("lr", 1.0),
        damping=cfg.get("damping", 1e-6),
        max_params=cfg.get("max_params", 20_000),
    )


@OPTIMIZER_REGISTRY.register("newton")
def build_newton(cfg: dict, params):
    return None


def build_optimizer(cfg: dict, params):
    return OPTIMIZER_REGISTRY.build(cfg, params)
