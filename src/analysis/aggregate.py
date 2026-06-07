"""Aggregate per-run JSON files into a single pandas DataFrame / Parquet."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.config import settings


def aggregate_results(results_dir: Path = settings.results_dir) -> pd.DataFrame:
    """Scan results/*.json, parse BenchmarkRun objects, flatten into DataFrame.

    Saves aggregated.parquet to results_dir. Returns the DataFrame.
    Never overwrites existing run JSONs.
    """
    import json

    _SKIP = {"aggregated.json", "speculative_decoding.json", "mcu_benchmark.json"}
    run_files = sorted(results_dir.glob("*.json"))
    rows = []
    for f in run_files:
        if f.name in _SKIP:
            continue
        try:
            data = json.loads(f.read_text())
            if "run_id" not in data:
                continue  # skip non-BenchmarkRun JSONs (quality_*.json etc.)
            rows.append(_run_to_row(data))
        except Exception as e:
            import warnings

            warnings.warn(f"Skipping {f}: {e}", stacklevel=2)

    df = pd.DataFrame(rows)
    if not df.empty:
        out = results_dir / "aggregated.parquet"
        df.to_parquet(out, index=False)
    return df


def _run_to_row(run: dict) -> dict:
    """Flatten a BenchmarkRun dict into a single-level row dict."""
    row: dict = {
        "run_id": run.get("run_id", ""),
        "timestamp": run.get("timestamp", ""),
        "model_id": run.get("model", {}).get("id", ""),
        "model_params_b": run.get("model", {}).get("params_b", None),
        "quant_name": run.get("quant", {}).get("name", ""),
        "quant_bits": run.get("quant", {}).get("bits", None),
        "hardware_id": run.get("hardware", {}).get("id", ""),
        "hardware_ram_gb": run.get("hardware", {}).get("ram_gb", None),
        "task_id": (run.get("task") or {}).get("id", None),
    }
    t = run.get("throughput", {})
    row.update(
        {
            "tps_median": t.get("median_tok_per_s"),
            "tps_iqr": t.get("iqr_tok_per_s"),
            "tps_min": t.get("min_tok_per_s"),
            "tps_max": t.get("max_tok_per_s"),
            "ttft_ms_median": t.get("median_ttft_ms"),
            "tpot_ms_median": t.get("median_tpot_ms"),
            "n_measured": t.get("n_measured"),
        }
    )
    m = run.get("memory", {})
    row.update(
        {
            "peak_rss_mb": m.get("peak_rss_mb"),
            "peak_unified_mb": m.get("peak_unified_mb"),
        }
    )
    # Support both `energy` (current) and legacy `power` field
    e = run.get("energy") or run.get("power") or {}
    row.update(
        {
            "measured_joules_per_query": e.get("measured_joules_per_query")
            or e.get("joules_per_query"),
            "measured_tokens_per_joule": e.get("measured_tokens_per_joule"),
            "estimated_joules_per_query": e.get("estimated_joules_per_query"),
            "estimated_tokens_per_joule": e.get("estimated_tokens_per_joule"),
        }
    )
    q = run.get("quality") or {}
    row.update(
        {
            "quality_score": q.get("primary_metric_value"),
            "quality_n_samples": q.get("n_samples"),
        }
    )
    return row
