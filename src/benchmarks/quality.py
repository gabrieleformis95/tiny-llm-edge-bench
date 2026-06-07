"""Quality eval orchestrator: load task -> run inference -> score."""

from __future__ import annotations

from src.config import QualityResult, TaskSpec
from src.inference.base import InferenceBackend
from src.tasks.base import Task


def run_quality_benchmark(
    backend: InferenceBackend,
    task: Task,
    task_spec: TaskSpec,
    temperature: float = 0.0,
    seed: int = 42,
) -> QualityResult:
    """Iterate task samples, run inference, score, and return aggregated QualityResult."""
    per_sample: list[dict] = []
    scores: list[float] = []

    for sample in task.iter_samples():
        prompt = task.build_prompt(sample)
        result = backend.generate(prompt, max_tokens=512, temperature=temperature, seed=seed)
        score = task.score(result.text, sample.get("reference", ""))
        scores.append(score)
        entry: dict = {
            "sample_id": sample.get("id", len(per_sample)),
            "score": score,
            "prediction": result.text[:500],
        }
        if hasattr(task, "last_diagnostics") and task.last_diagnostics:
            entry.update(task.last_diagnostics)
        per_sample.append(entry)

    avg = sum(scores) / len(scores) if scores else 0.0
    return QualityResult(
        task_id=task_spec.id,
        n_samples=len(scores),
        primary_metric_value=avg,
        per_sample_results=per_sample,
    )
