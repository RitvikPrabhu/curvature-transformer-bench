from __future__ import annotations

import torch
from torch import nn

from curvature_bench.registry import MODEL_REGISTRY


@MODEL_REGISTRY.register("mlp")
class MLP(nn.Module):
    def __init__(self, cfg: dict) -> None:
        super().__init__()

        input_dim = cfg.get("input_dim", 784)
        hidden_dims = cfg.get("hidden_dims", [512, 256])
        output_dim = cfg.get("output_dim", 10)
        activation = cfg.get("activation", "relu")
        dropout = cfg.get("dropout", 0.0)

        if activation == "relu":
            act_layer = nn.ReLU
        elif activation == "gelu":
            act_layer = nn.GELU
        elif activation == "tanh":
            act_layer = nn.Tanh
        else:
            raise ValueError(f"Unsupported activation: {activation}")

        layers: list[nn.Module] = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(act_layer())

            if dropout > 0:
                layers.append(nn.Dropout(dropout))

            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(x.size(0), -1)
        return self.net(x)


def build_model(cfg: dict) -> nn.Module:
    return MODEL_REGISTRY.build(cfg)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)