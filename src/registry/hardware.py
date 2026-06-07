"""Detect host hardware and match to a HardwareProfile."""

from __future__ import annotations

import platform
import subprocess
from functools import lru_cache
from pathlib import Path

import psutil
import yaml

from src.config import HardwareProfile

_HW_YAML = Path(__file__).parents[2] / "configs" / "hardware.yaml"


@lru_cache(maxsize=1)
def detect_hardware() -> HardwareProfile:
    """Auto-detect host hardware and return the best matching HardwareProfile."""
    ram_gb = psutil.virtual_memory().total / (1024**3)
    cores = psutil.cpu_count(logical=False) or 1

    if _is_apple_silicon():
        family = "apple_silicon"
        can_power = True
    elif _is_rpi():
        family = "rpi"
        can_power = False
    else:
        family = "x86_linux"
        can_power = False

    # Try to match a known profile by family + approximate RAM
    for profile in load_profiles():
        if profile.family == family and abs(profile.ram_gb - ram_gb) <= 2.0:
            return profile

    # Fallback: synthesize a profile from detected values
    return HardwareProfile(
        id=f"{family}_{cores}core_{int(ram_gb)}gb",
        family=family,
        cores=cores,
        ram_gb=round(ram_gb, 1),
        can_run_powermetrics=can_power,
    )


@lru_cache(maxsize=1)
def load_profiles() -> list[HardwareProfile]:
    """Return all HardwareProfile entries from hardware.yaml."""
    with open(_HW_YAML) as f:
        data = yaml.safe_load(f)
    return [HardwareProfile(**p) for p in data["profiles"]]


def _is_apple_silicon() -> bool:
    return platform.system() == "Darwin" and platform.machine() == "arm64"


def _is_rpi() -> bool:
    try:
        with open("/proc/device-tree/model") as f:
            return "Raspberry Pi" in f.read()
    except FileNotFoundError:
        return False
