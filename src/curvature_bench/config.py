from __future__ import annotations

from pathlib import Path

import yaml


def load_yaml(path: str | Path) -> dict:
    path = Path(path)

    with open(path, "r") as f:
        return yaml.safe_load(f)


def deep_update(base: dict, update: dict) -> dict:
    out = dict(base)

    for key, value in update.items():
        if (
            key in out
            and isinstance(out[key], dict)
            and isinstance(value, dict)
        ):
            out[key] = deep_update(out[key], value)
        else:
            out[key] = value

    return out


def load_experiment_config(path: str | Path) -> dict:
    """
    Loads one complete experiment YAML.

    Keep this simple for now. Later, this can support config composition.
    """
    return load_yaml(path)