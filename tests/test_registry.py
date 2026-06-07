"""Tests for model/quant/task registry loading and validation."""

from __future__ import annotations

import pytest

from src.registry.models import load_models, load_quants, get_model, get_quant
from src.registry.tasks import load_tasks, get_task
from src.registry.hardware import detect_hardware, load_profiles


def test_load_models_returns_list():
    models = load_models()
    assert len(models) == 4
    assert all(m.id for m in models)


def test_model_ids_unique():
    models = load_models()
    ids = [m.id for m in models]
    assert len(ids) == len(set(ids))


def test_load_quants_returns_8():
    quants = load_quants()
    assert len(quants) == 8  # 6 GGUF + 1 AWQ + 1 GPTQ


def test_load_tasks_returns_list():
    tasks = load_tasks()
    assert len(tasks) >= 1


def test_task_ids_unique():
    tasks = load_tasks()
    ids = [t.id for t in tasks]
    assert len(ids) == len(set(ids))


def test_model_spec_fields(sample_model_spec):
    assert sample_model_spec.id == "qwen2.5-0.5b-instruct"
    assert sample_model_spec.params_b == 0.5
    assert sample_model_spec.context_len == 32768


def test_quant_spec_fields(sample_quant_spec):
    assert sample_quant_spec.name == "Q4_K_M"
    assert sample_quant_spec.bits == 4.5


def test_get_model_found():
    m = get_model("qwen2.5-0.5b-instruct")
    assert m.params_b == 0.5


def test_get_model_not_found():
    with pytest.raises(KeyError):
        get_model("does-not-exist")


def test_get_quant_found():
    q = get_quant("Q4_K_M")
    assert q.bits == 4.5


def test_get_task_found():
    t = get_task("ragas_industrial")
    assert t.kind == "qa"


def test_detect_hardware_returns_profile():
    hw = detect_hardware()
    assert hw.id
    assert hw.ram_gb > 0
    assert hw.cores > 0


def test_load_profiles_not_empty():
    profiles = load_profiles()
    assert len(profiles) >= 1
