"""Capture a full HardwareFingerprint at run start."""

from __future__ import annotations

import os
import platform
import subprocess

from src.config import HardwareFingerprint
from src.registry.hardware import detect_hardware


def capture_fingerprint(ambient_temp_c: float | None = None) -> HardwareFingerprint:
    """Snapshot hardware + software state. Call once per benchmark run."""
    hw = detect_hardware()
    return HardwareFingerprint(
        host_id=hw.id,
        family=hw.family,
        cores=hw.cores,
        ram_gb=hw.ram_gb,
        cpu_freq_mhz=_cpu_freq_mhz(),
        cpu_governor=_cpu_governor(),
        omp_threads=int(os.environ.get("OMP_NUM_THREADS", hw.cores)),
        ac_powered=_ac_powered(),
        ambient_temp_c=ambient_temp_c,
        llama_cpp_sha=_llama_cpp_sha(),
        python_version=platform.python_version(),
        os_release=platform.platform(),
    )


def _cpu_freq_mhz() -> float | None:
    if platform.system() == "Darwin":
        try:
            out = subprocess.check_output(
                ["sysctl", "-n", "hw.cpufrequency"], stderr=subprocess.DEVNULL
            )
            return float(out.strip()) / 1e6
        except Exception:
            # Apple Silicon does not expose hw.cpufrequency; return None
            return None
    try:
        import psutil

        freq = psutil.cpu_freq()
        return freq.current if freq else None
    except Exception:
        return None


def _cpu_governor() -> str | None:
    try:
        with open("/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor") as f:
            return f.read().strip()
    except FileNotFoundError:
        return None


def _ac_powered() -> bool | None:
    if platform.system() != "Darwin":
        return None
    try:
        out = subprocess.check_output(["pmset", "-g", "ps"], stderr=subprocess.DEVNULL).decode()
        return "AC Power" in out
    except Exception:
        return None


def _llama_cpp_sha() -> str | None:
    try:
        import llama_cpp

        # llama_cpp exposes __version__ but not git SHA; use version as proxy
        return getattr(llama_cpp, "__version__", None)
    except ImportError:
        return None
