"""Single benchmark run driver.

Usage:
  python scripts/run_bench.py --model qwen2.5-0.5b-instruct --quant Q4_K_M --task none
  python scripts/run_bench.py --model phi-3.5-mini-instruct --quant Q4_K_M --task mmlu_subset
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1]))

from src.config import settings
from src.registry.hardware import detect_hardware
from src.registry.models import get_model, get_quant


def _find_existing_run(model_id: str, quant_name: str, task_id: str | None) -> Path | None:
    """Return path to an existing result JSON matching (model, quant, task), or None."""
    for f in settings.results_dir.glob("*.json"):
        if f.name in ("aggregated.json",):
            continue
        try:
            data = json.loads(f.read_text())
            if (
                data.get("model", {}).get("id") == model_id
                and data.get("quant", {}).get("name") == quant_name
                and (data.get("task") or {}).get("id") == task_id
            ):
                return f
        except Exception:
            continue
    return None


def run_benchmark(
    model_id: str,
    quant_name: str,
    task_id: str | None,
    backend_name: str = "llama_cpp",
) -> Path:
    """Execute a full benchmark run and persist JSON. Returns path to result file.

    Skips and returns the existing path if a matching run JSON already exists.
    """
    existing = _find_existing_run(model_id, quant_name, task_id)
    if existing:
        print(f"[skip] Run already exists: {existing}")
        return existing

    from src.registry.downloader import download_gguf
    from src.benchmarks.memory import measure_peak_memory
    from src.benchmarks.power_mac import measure_energy_mac
    from src.benchmarks.power_analytical import estimate_energy_analytical, merge_energy_results
    from src.benchmarks.quality import run_quality_benchmark
    from src.benchmarks.throughput import run_throughput_benchmark, N_MEASURED
    from src.config import BenchmarkRun, settings
    from src.registry.hardware import detect_hardware
    from src.registry.models import get_model, get_quant
    from src.registry.tasks import get_task

    from src.registry.fingerprint import capture_fingerprint

    model = get_model(model_id)
    quant = get_quant(quant_name)
    hardware = detect_hardware()
    fingerprint = capture_fingerprint()

    gguf_path = download_gguf(model_id, quant_name)

    # Build backend
    if backend_name == "llama_cpp":
        from src.inference.llama_cpp_backend import LlamaCppBackend
        backend = LlamaCppBackend()
    else:
        from src.inference.mlx_backend import MLXBackend
        backend = MLXBackend()

    with backend:
        backend.load(gguf_path, n_ctx=2048, chat_template=model.chat_template)

        # Throughput + memory
        def _throughput():
            return run_throughput_benchmark(backend)

        throughput_result, mem_result = measure_peak_memory(_throughput)

        # Energy: measured (powermetrics + baseline) + analytical (MAC count)
        avg_output_tokens = int(throughput_result.median_tpot_ms and
                                256 * N_MEASURED or 256)  # approx tokens for KPI
        def _energy_pass():
            return run_throughput_benchmark(backend)

        _, measured_energy = measure_energy_mac(_energy_pass, tokens_generated=avg_output_tokens)
        analytical_energy = estimate_energy_analytical(
            params_b=model.params_b,
            n_tokens=avg_output_tokens,
        )
        energy_result = merge_energy_results(measured_energy, analytical_energy)

        # Quality (optional)
        quality_result = None
        task_spec = None
        if task_id:
            task_spec = get_task(task_id)
            task_obj = _build_task(task_spec)
            quality_result = run_quality_benchmark(backend, task_obj, task_spec)

    run = BenchmarkRun(
        model=model,
        quant=quant,
        hardware=hardware,
        fingerprint=fingerprint,
        task=task_spec,
        throughput=throughput_result,
        memory=mem_result,
        energy=energy_result,
        quality=quality_result,
    )

    settings.results_dir.mkdir(parents=True, exist_ok=True)
    out = settings.results_dir / f"{run.run_id}.json"
    out.write_text(run.model_dump_json(indent=2))
    return out


def _build_task(task_spec):
    from src.tasks.instruction_follow import InstructionFollowTask
    from src.tasks.json_following import JSONFollowingTask
    from src.tasks.mmlu_subset import MMLUSubsetTask
    from src.tasks.ragas_industrial import RagasIndustrialTask

    mapping = {
        "ragas_industrial": lambda: RagasIndustrialTask(task_spec.dataset_path),
        "mmlu_subset": lambda: MMLUSubsetTask(task_spec.dataset_path, task_spec.n_samples),
        "json_following": lambda: JSONFollowingTask(task_spec.dataset_path),
        "instruction_follow": lambda: InstructionFollowTask(task_spec.dataset_path),
    }
    factory = mapping.get(task_spec.id)
    if factory is None:
        raise ValueError(f"Unknown task: {task_spec.id}")
    return factory()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a single tiny-llm benchmark")
    parser.add_argument("--model", required=True)
    parser.add_argument("--quant", required=True)
    parser.add_argument("--task", default=None)
    parser.add_argument("--backend", default="llama_cpp", choices=["llama_cpp", "mlx"])
    args = parser.parse_args()

    task_id = args.task if args.task and args.task != "none" else None
    out = run_benchmark(args.model, args.quant, task_id, args.backend)
    print(f"Result saved: {out}")


if __name__ == "__main__":
    main()
