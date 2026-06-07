"""Shared pytest fixtures."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.config import HardwareProfile, ModelSpec, QuantSpec, TaskSpec


@pytest.fixture
def sample_model_spec() -> ModelSpec:
    return ModelSpec(
        id="qwen2.5-0.5b-instruct",
        hf_gguf_repo="Qwen/Qwen2.5-0.5B-Instruct-GGUF",
        params_b=0.5,
        context_len=32768,
    )


@pytest.fixture
def sample_quant_spec() -> QuantSpec:
    return QuantSpec(name="Q4_K_M", bits=4.5, file_suffix="Q4_K_M.gguf")


@pytest.fixture
def sample_hardware() -> HardwareProfile:
    return HardwareProfile(
        id="mac_m2_pro_16gb",
        family="apple_silicon",
        cores=12,
        ram_gb=16.0,
        can_run_powermetrics=True,
    )


@pytest.fixture
def sample_task_spec(tmp_path: Path) -> TaskSpec:
    dataset = tmp_path / "dummy.json"
    dataset.write_text("[]")
    return TaskSpec(
        id="test_task",
        kind="qa",
        dataset_path=dataset,
        metric="exact_match",
        n_samples=5,
    )
