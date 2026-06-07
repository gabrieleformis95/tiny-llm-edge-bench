"""Peak memory measurement during inference (RSS + Apple Silicon unified memory)."""

from __future__ import annotations

import os
import threading
import time
from typing import Callable, TypeVar

import psutil

from src.config import MemoryResult

T = TypeVar("T")

_SAMPLE_INTERVAL_S = 0.05  # 50ms


def measure_peak_memory(fn: Callable[[], T]) -> tuple[T, MemoryResult]:
    """Run fn() while sampling RSS every 50ms; return (result, MemoryResult)."""
    proc = psutil.Process(os.getpid())
    peak_rss_bytes = 0
    peak_unified_bytes = 0
    stop_event = threading.Event()

    def _sample() -> None:
        nonlocal peak_rss_bytes, peak_unified_bytes
        while not stop_event.is_set():
            try:
                rss = proc.memory_info().rss
                peak_rss_bytes = max(peak_rss_bytes, rss)
                unified = _mlx_active_memory()
                if unified is not None:
                    peak_unified_bytes = max(peak_unified_bytes, unified)
            except psutil.NoSuchProcess:
                break
            time.sleep(_SAMPLE_INTERVAL_S)

    sampler = threading.Thread(target=_sample, daemon=True)
    sampler.start()
    try:
        result = fn()
    finally:
        stop_event.set()
        sampler.join(timeout=1.0)

    return result, MemoryResult(
        peak_rss_mb=peak_rss_bytes / (1024**2),
        peak_unified_mb=(peak_unified_bytes / (1024**2)) if peak_unified_bytes else None,
    )


def _mlx_active_memory() -> int | None:
    """Return MLX active unified memory in bytes, or None if not available."""
    try:
        import mlx.core as mx  # type: ignore[import]
        return mx.metal.get_active_memory()
    except Exception:
        return None
