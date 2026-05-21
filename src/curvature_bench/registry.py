from __future__ import annotations

from typing import Any, Callable


class Registry:
    def __init__(self, name: str) -> None:
        self.name = name
        self._items: dict[str, Callable[..., Any]] = {}

    def register(self, name: str):
        name = name.lower()

        def decorator(fn_or_cls):
            if name in self._items:
                raise KeyError(f"{name} is already registered in {self.name}")
            self._items[name] = fn_or_cls
            return fn_or_cls

        return decorator

    def get(self, name: str):
        name = name.lower()
        if name not in self._items:
            available = ", ".join(sorted(self._items.keys()))
            raise KeyError(
                f"{name} is not registered in {self.name}. "
                f"Available: {available}"
            )
        return self._items[name]

    def build(self, cfg: dict, *args, **kwargs):
        name = cfg.get("name")
        if name is None:
            raise KeyError(f"Missing 'name' in config for registry {self.name}")

        item = self.get(name)
        return item(cfg, *args, **kwargs)

    def available(self) -> list[str]:
        return sorted(self._items.keys())


MODEL_REGISTRY = Registry("models")
DATA_REGISTRY = Registry("data")
OPTIMIZER_REGISTRY = Registry("optimizers")