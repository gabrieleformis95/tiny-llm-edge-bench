"""Phase 4 DoD: run all 4 quality tasks with a single GGUF backend.

Usage:
  python scripts/run_quality_dod.py --gguf-path /path/to/model.gguf [--max-samples N]

Produces one results/quality_{task_id}.json per task.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1]))

from src.benchmarks.quality import run_quality_benchmark
from src.config import QualityResult, TaskSpec, settings
from src.inference.llama_cpp_backend import LlamaCppBackend
from src.registry.tasks import load_tasks
from src.tasks.instruction_follow import InstructionFollowTask
from src.tasks.json_following import JSONFollowingTask
from src.tasks.mmlu_subset import MMLUSubsetTask
from src.tasks.ragas_industrial import RagasIndustrialTask


def _build_task(spec: TaskSpec, max_samples: int):
    if spec.id == "ragas_industrial":
        return RagasIndustrialTask(spec.dataset_path)
    if spec.id == "mmlu_subset":
        return MMLUSubsetTask(spec.dataset_path, n_samples=min(spec.n_samples, max_samples))
    if spec.id == "json_following":
        return JSONFollowingTask(spec.dataset_path)
    if spec.id == "instruction_follow":
        return InstructionFollowTask(spec.dataset_path)
    raise ValueError(f"Unknown task: {spec.id}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gguf-path", required=True, type=Path)
    parser.add_argument("--max-samples", type=int, default=10,
                        help="Cap for MMLU (other tasks use their full dataset)")
    parser.add_argument("--n-gpu-layers", type=int, default=0,
                        help="GPU layers for llama.cpp (0=CPU)")
    args = parser.parse_args()

    if not args.gguf_path.exists():
        sys.exit(f"ERROR: {args.gguf_path} not found")

    settings.results_dir.mkdir(parents=True, exist_ok=True)
    specs = {s.id: s for s in load_tasks()}

    backend = LlamaCppBackend()
    backend.load(args.gguf_path, n_ctx=2048, n_gpu_layers=args.n_gpu_layers)

    results: dict[str, QualityResult] = {}
    for task_id in ["ragas_industrial", "json_following", "instruction_follow", "mmlu_subset"]:
        spec = specs[task_id]
        task = _build_task(spec, args.max_samples)
        n = min(spec.n_samples, args.max_samples) if task_id == "mmlu_subset" else spec.n_samples
        print(f"\n[{task_id}] running {n} samples...", flush=True)
        result = run_quality_benchmark(backend, task, spec)
        results[task_id] = result
        out = settings.results_dir / f"quality_{task_id}.json"
        out.write_text(result.model_dump_json(indent=2))
        print(f"  score={result.primary_metric_value:.3f}  n={result.n_samples}  -> {out}")

    backend.unload()

    print("\n" + "=" * 60)
    print("Phase 4 DoD — Quality Results")
    print("=" * 60)
    for tid, r in results.items():
        print(f"  {tid:25s}  score={r.primary_metric_value:.3f}  n={r.n_samples}")
    print("=" * 60)
    print("All 4 task JSONs written. Schema: QualityResult (task_id, n_samples,")
    print("primary_metric_value, per_sample_results). DoD: PASS")


if __name__ == "__main__":
    main()
