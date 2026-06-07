"""macOS powermetrics-based power measurement.

Requires sudo. Gracefully returns None on non-Apple-Silicon or when sudo unavailable.
"""

from __future__ import annotations

import platform
import re
import subprocess
import threading
import time
from collections.abc import Callable
from typing import TypeVar

from src.config import PowerResult, settings

T = TypeVar("T")

_SAMPLE_MS = 100  # powermetrics sample interval in ms


def measure_power(fn: Callable[[], T]) -> tuple[T, PowerResult | None]:
    """Run fn() under powermetrics sampling; return (result, PowerResult | None)."""
    if settings.skip_power_measurement:
        return fn(), None
    if platform.system() != "Darwin":
        return fn(), None
    if not _has_sudo():
        import warnings

        warnings.warn(
            "sudo unavailable - skipping power measurement. "
            "Run 'sudo -v' once before benchmarking to enable it.",
            stacklevel=2,
        )
        return fn(), None

    watts_samples: list[float] = []
    stop_event = threading.Event()

    def _run_powermetrics() -> None:
        proc = subprocess.Popen(
            [
                "sudo",
                "powermetrics",
                "--samplers",
                "cpu_power,gpu_power",
                "-i",
                str(_SAMPLE_MS),
                "-f",
                "plist",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            if stop_event.is_set():
                proc.terminate()
                break
            # CPU+GPU combined power in mW
            m = re.search(r"<key>combined_power</key>\s*<real>([\d.]+)</real>", line)
            if m:
                watts_samples.append(float(m.group(1)) / 1000.0)  # mW -> W

    pm_thread = threading.Thread(target=_run_powermetrics, daemon=True)
    pm_thread.start()
    t0 = time.perf_counter()
    try:
        result = fn()
    finally:
        duration_s = time.perf_counter() - t0
        stop_event.set()
        pm_thread.join(timeout=2.0)

    if not watts_samples:
        return result, None

    avg_watts = sum(watts_samples) / len(watts_samples)
    return result, PowerResult(
        avg_watts=avg_watts,
        joules_per_query=avg_watts * duration_s,
        duration_s=duration_s,
    )


def _has_sudo() -> bool:
    """Check whether we can run sudo without a password prompt."""
    result = subprocess.run(["sudo", "-n", "true"], capture_output=True)
    return result.returncode == 0
