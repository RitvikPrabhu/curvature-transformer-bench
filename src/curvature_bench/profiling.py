from __future__ import annotations

import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

import torch


def sync_device(device: torch.device) -> None:
    """
    Synchronize device before/after timing.

    CUDA is asynchronous, so timing without synchronization can be misleading.
    """
    if device.type == "cuda":
        torch.cuda.synchronize()


@dataclass
class TimerResult:
    name: str
    elapsed_ms: float


@dataclass
class StepTimer:
    """
    Simple reusable timer for one training step.

    Example:

      timer = StepTimer(device)

      with timer.time("forward"):
          logits = model(x)

      with timer.time("backward"):
          loss.backward()

      print(timer.results)
    """

    device: torch.device
    results: dict[str, float] = field(default_factory=dict)

    @contextmanager
    def time(self, name: str) -> Iterator[None]:
        sync_device(self.device)
        start = time.perf_counter()

        yield

        sync_device(self.device)
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        self.results[name] = self.results.get(name, 0.0) + elapsed_ms

    def get(self, name: str, default: float = 0.0) -> float:
        return float(self.results.get(name, default))

    def reset(self) -> None:
        self.results.clear()

    def as_dict(self) -> dict[str, float]:
        return dict(self.results)


@dataclass
class CudaMemoryTracker:
    """
    Track CUDA peak memory.

    On non-CUDA devices, returns zero.
    """

    device: torch.device

    def reset(self) -> None:
        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)

    def peak_mb(self) -> float:
        if self.device.type == "cuda":
            return torch.cuda.max_memory_allocated(self.device) / 1024**2

        return 0.0


class NullProfiler:
    """
    No-op profiler.

    Use this when profiling is disabled but you still want the same interface.
    """

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        return False

    def step(self) -> None:
        pass


class TorchProfilerWrapper:
    """
    Thin wrapper around torch.profiler.

    This is optional and heavier than the manual timers.

    For initial experiments, manual timing is enough. Use this later when you
    want Chrome traces or operator-level breakdowns.
    """

    def __init__(
        self,
        output_dir: str | Path,
        enabled: bool = False,
        wait: int = 1,
        warmup: int = 1,
        active: int = 3,
        repeat: int = 1,
        record_shapes: bool = True,
        profile_memory: bool = True,
        with_stack: bool = False,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.enabled = enabled
        self.wait = wait
        self.warmup = warmup
        self.active = active
        self.repeat = repeat
        self.record_shapes = record_shapes
        self.profile_memory = profile_memory
        self.with_stack = with_stack
        self.profiler = None

    def __enter__(self):
        if not self.enabled:
            return NullProfiler()

        self.output_dir.mkdir(parents=True, exist_ok=True)

        activities = [torch.profiler.ProfilerActivity.CPU]

        if torch.cuda.is_available():
            activities.append(torch.profiler.ProfilerActivity.CUDA)

        self.profiler = torch.profiler.profile(
            activities=activities,
            schedule=torch.profiler.schedule(
                wait=self.wait,
                warmup=self.warmup,
                active=self.active,
                repeat=self.repeat,
            ),
            on_trace_ready=torch.profiler.tensorboard_trace_handler(
                str(self.output_dir)
            ),
            record_shapes=self.record_shapes,
            profile_memory=self.profile_memory,
            with_stack=self.with_stack,
        )

        self.profiler.__enter__()
        return self.profiler

    def __exit__(self, exc_type, exc_value, traceback):
        if self.profiler is not None:
            return self.profiler.__exit__(exc_type, exc_value, traceback)

        return False


def build_profiler(cfg: dict, output_dir: str | Path):
    """
    Build profiler from runtime config.

    Example config:

      runtime:
        profiler:
          enabled: true
          wait: 1
          warmup: 1
          active: 3
          repeat: 1
    """
    profiler_cfg = cfg.get("profiler", {})
    enabled = bool(profiler_cfg.get("enabled", False))

    if not enabled:
        return NullProfiler()

    return TorchProfilerWrapper(
        output_dir=Path(output_dir) / "torch_profiler",
        enabled=True,
        wait=int(profiler_cfg.get("wait", 1)),
        warmup=int(profiler_cfg.get("warmup", 1)),
        active=int(profiler_cfg.get("active", 3)),
        repeat=int(profiler_cfg.get("repeat", 1)),
        record_shapes=bool(profiler_cfg.get("record_shapes", True)),
        profile_memory=bool(profiler_cfg.get("profile_memory", True)),
        with_stack=bool(profiler_cfg.get("with_stack", False)),
    )