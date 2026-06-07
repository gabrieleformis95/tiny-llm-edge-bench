"""Integration tests: quality pipeline with mocked backend across all 4 tasks."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from src.benchmarks.quality import run_quality_benchmark
from src.config import QualityResult, TaskSpec
from src.inference.base import GenerationResult


def _mock_backend(reply: str = "A") -> MagicMock:
    backend = MagicMock()
    backend.generate.return_value = GenerationResult(
        text=reply,
        ttft_ms=None,
        tpot_ms=5.0,
        total_tokens=10,
        prompt_tokens=5,
    )
    return backend


def _task_spec(task_id: str, kind: str, path: Path, metric: str, n: int) -> TaskSpec:
    return TaskSpec(id=task_id, kind=kind, dataset_path=path, metric=metric, n_samples=n)


@pytest.fixture(autouse=True)
def _disable_groq_judge(monkeypatch):
    """Keep the suite hermetic: force the ROUGE-L path so the ragas task never
    makes a live Groq call when GROQ_API_KEY happens to be set (e.g. via .env)."""
    from src.config import settings

    monkeypatch.setattr(settings, "groq_api_key", "", raising=False)


# --- ragas_industrial ---


def test_ragas_industrial_pipeline_schema():
    from src.tasks.ragas_industrial import RagasIndustrialTask

    path = Path("data/golden/ragas_golden.json")
    task = RagasIndustrialTask(path)
    spec = _task_spec("ragas_industrial", "qa", path, "faithfulness", 8)
    backend = _mock_backend("Turbine blade degradation causes high EGT.")

    result = run_quality_benchmark(backend, task, spec)

    assert isinstance(result, QualityResult)
    assert result.task_id == "ragas_industrial"
    assert result.n_samples == 8
    assert 0.0 <= result.primary_metric_value <= 1.0
    assert len(result.per_sample_results) == 8
    for row in result.per_sample_results:
        assert "sample_id" in row
        assert "score" in row
        assert "prediction" in row


# --- mmlu_subset ---


def test_mmlu_pipeline_schema(tmp_path):
    from src.tasks.mmlu_subset import MMLUSubsetTask

    data = [
        {"id": f"q{i}", "question": f"Q{i}", "choices": ["A", "B", "C", "D"], "answer": "A"}
        for i in range(5)
    ]
    p = tmp_path / "mmlu.json"
    p.write_text(json.dumps(data))

    task = MMLUSubsetTask(p, n_samples=5)
    spec = _task_spec("mmlu_subset", "classification", p, "exact_match", 5)
    backend = _mock_backend("A")

    result = run_quality_benchmark(backend, task, spec)

    assert result.task_id == "mmlu_subset"
    assert result.n_samples == 5
    assert result.primary_metric_value == 1.0  # mock always returns "A" == answer "A"
    assert len(result.per_sample_results) == 5


# --- json_following ---


def test_json_following_pipeline_schema():
    from src.tasks.json_following import JSONFollowingTask

    path = Path("data/prompts/json_following.json")
    task = JSONFollowingTask(path)
    spec = _task_spec("json_following", "json", path, "json_validity", 20)
    backend = _mock_backend('{"name": "test", "value": 1, "unit": "kg", "status": "ok"}')

    result = run_quality_benchmark(backend, task, spec)

    assert result.task_id == "json_following"
    assert result.n_samples == 20
    assert 0.0 <= result.primary_metric_value <= 1.0
    assert len(result.per_sample_results) == 20


# --- instruction_follow ---


def test_instruction_follow_pipeline_schema():
    from src.tasks.instruction_follow import InstructionFollowTask

    path = Path("data/prompts/instruction_follow.json")
    task = InstructionFollowTask(path)
    spec = _task_spec("instruction_follow", "classification", path, "compliance_rate", 30)
    backend = _mock_backend("1. Item one\n2. Item two\n3. Item three")

    result = run_quality_benchmark(backend, task, spec)

    assert result.task_id == "instruction_follow"
    assert result.n_samples == 30
    assert 0.0 <= result.primary_metric_value <= 1.0
    assert len(result.per_sample_results) == 30


# --- schema field completeness ---


def test_quality_result_serializable():
    from src.tasks.ragas_industrial import RagasIndustrialTask

    path = Path("data/golden/ragas_golden.json")
    task = RagasIndustrialTask(path)
    spec = _task_spec("ragas_industrial", "qa", path, "faithfulness", 8)
    result = run_quality_benchmark(_mock_backend("answer"), task, spec)

    dumped = result.model_dump_json()
    reloaded = QualityResult.model_validate_json(dumped)
    assert reloaded.task_id == result.task_id
    assert reloaded.n_samples == result.n_samples
