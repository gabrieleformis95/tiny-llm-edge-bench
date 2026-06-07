"""macOS powermetrics power measurement with idle baseline subtraction.

Protocol:
  1. Measure idle power for BASELINE_S seconds -> avg_baseline_W
  2. Run the inference callable under powermetrics for its full duration
  3. attributed_W = avg_inference_W - avg_baseline_W  (clamped >= 0)
  4. energy_per_query = attributed_W * seconds_per_query

Requires: sudo powermetrics (macOS only). Degrades gracefully if unavailable.

All results labeled MEASURED in output JSON.
"""

from __future__ import annotations

import platform
import re
import subprocess
import threading
import time
from collections.abc import Callable
from typing import TypeVar

from src.config import EnergyResult, settings

T = TypeVar("T")

_SAMPLE_MS = 100  # powermetrics interval
BASELINE_S = 30  # idle baseline window


def measure_energy_mac(
    fn: Callable[[], T],
    tokens_generated: int,
) -> tuple[T, EnergyResult | None]:
    """Run fn() under powermetrics with baseline subtraction.

    Returns (fn_result, EnergyResult | None).
    EnergyResult is None when powermetrics is unavailable.
    tokens_generated is used to compute tokens_per_joule KPI.
    """
    if settings.skip_power_measurement:
        return fn(), None
    if platform.system() != "Darwin":
        return fn(), None
    if not _has_sudo():
        import warnings

        warnings.warn(
            "sudo unavailable - skipping power measurement. "
            "Run 'sudo -v' once before benchmarking to enable.",
            stacklevel=2,
        )
        return fn(), None

    avg_baseline_w = _measure_baseline()

    # Inference pass under powermetrics
    watts_samples: list[float] = []
    stop_event = threading.Event()

    def _pm_thread() -> None:
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
            m = re.search(r"<key>combined_power</key>\s*<real>([\d.]+)</real>", line)
            if m:
                watts_samples.append(float(m.group(1)) / 1000.0)  # mW -> W

    t = threading.Thread(target=_pm_thread, daemon=True)
    t.start()
    t0 = time.perf_counter()
    try:
        result = fn()
    finally:
        duration_s = time.perf_counter() - t0
        stop_event.set()
        t.join(timeout=2.0)

    if not watts_samples:
        return result, None

    avg_inference_w = sum(watts_samples) / len(watts_samples)
    attributed_w = max(0.0, avg_inference_w - avg_baseline_w)
    joules_per_query = attributed_w * duration_s
    tpj = tokens_generated / joules_per_query if joules_per_query > 0 else None

    return result, EnergyResult(
        measured_joules_per_query=joules_per_query,
        measured_tokens_per_joule=tpj,
        estimation_method="powermetrics_mac",
        notes=(
            f"MEASURED. avg_inference={avg_inference_w:.2f}W, "
            f"avg_baseline={avg_baseline_w:.2f}W, "
            f"attributed={attributed_w:.2f}W, "
            f"duration={duration_s:.1f}s. "
            f"Baseline: {BASELINE_S}s idle. "
            "Caveat: unified memory on Apple Silicon means RSS over-counts; "
            "powermetrics combined_power includes ANE but not DRAM."
        ),
    )


def _measure_baseline() -> float:
    """Measure idle combined power for BASELINE_S seconds. Returns avg watts."""
    samples: list[float] = []
    stop = threading.Event()

    def _pm() -> None:
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
            if stop.is_set():
                proc.terminate()
                break
            m = re.search(r"<key>combined_power</key>\s*<real>([\d.]+)</real>", line)
            if m:
                samples.append(float(m.group(1)) / 1000.0)

    t = threading.Thread(target=_pm, daemon=True)
    t.start()
    time.sleep(BASELINE_S)
    stop.set()
    t.join(timeout=2.0)
    return sum(samples) / len(samples) if samples else 0.0


def _has_sudo() -> bool:
    return (
        subprocess.run(
            [
                "sudo",
                "-n",
                "/usr/bin/powermetrics",
                "--samplers",
                "cpu_power",
                "-i",
                "100",
                "-n",
                "1",
            ],
            capture_output=True,
        ).returncode
        == 0
    )
