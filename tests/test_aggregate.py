"""Tests for results aggregation."""

from __future__ import annotations

import json
from pathlib import Path

from src.analysis.aggregate import _run_to_row, aggregate_results


def _make_run_dict(tmp_path: Path) -> dict:
    return {
        "run_id": "abc123",
        "timestamp": "2024-01-01T00:00:00",
        "model": {
            "id": "qwen2.5-0.5b-instruct",
            "params_b": 0.5,
            "context_len": 32768,
            "hf_gguf_repo": "x",
            "chat_template": "chatml",
        },
        "quant": {
            "name": "Q4_K_M",
            "scheme": "gguf",
            "bits": 4.5,
            "file_suffix": "Q4_K_M.gguf",
            "extra": {},
        },
        "hardware": {
            "id": "mac",
            "family": "apple_silicon",
            "cores": 8,
            "ram_gb": 16.0,
            "can_run_powermetrics": True,
        },
        "fingerprint": {
            "host_id": "mac",
            "family": "apple_silicon",
            "cores": 8,
            "ram_gb": 16.0,
            "omp_threads": 8,
            "python_version": "3.11.0",
            "os_release": "macOS-14",
        },
        "task": None,
        "throughput": {
            "n_warmup": 5,
            "n_measured": 50,
            "median_tok_per_s": 42.0,
            "iqr_tok_per_s": 5.0,
            "min_tok_per_s": 38.0,
            "max_tok_per_s": 48.0,
            "median_ttft_ms": 10.0,
            "median_tpot_ms": 5.0,
            "raw_samples": [42.0] * 50,
        },
        "memory": {"peak_rss_mb": 1024.0, "peak_unified_mb": None},
        "power": None,
        "energy": {
            "measured_joules_per_query": 1.2,
            "measured_tokens_per_joule": 350.0,
            "estimated_joules_per_query": 0.05,
            "estimated_tokens_per_joule": 8400.0,
            "estimation_method": "powermetrics_mac+cmsis_nn_lai2018",
            "notes": "test",
        },
        "quality": None,
    }


def test_run_to_row():
    row = _run_to_row(_make_run_dict(Path(".")))
    assert row["model_id"] == "qwen2.5-0.5b-instruct"
    assert row["tps_median"] == 42.0
    assert row["n_measured"] == 50
    assert row["quant_bits"] == 4.5
    assert row["measured_tokens_per_joule"] == 350.0
    assert row["estimated_tokens_per_joule"] == 8400.0


def test_aggregate_empty_dir(tmp_path):
    df = aggregate_results(tmp_path)
    assert df.empty


def test_aggregate_one_run(tmp_path):
    run = _make_run_dict(tmp_path)
    (tmp_path / "run1.json").write_text(json.dumps(run))
    df = aggregate_results(tmp_path)
    assert len(df) == 1
    assert df.iloc[0]["model_id"] == "qwen2.5-0.5b-instruct"
