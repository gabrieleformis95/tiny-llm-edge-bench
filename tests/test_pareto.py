"""Tests for Pareto front computation."""

from __future__ import annotations

import pandas as pd
import pytest

from src.analysis.pareto import pareto_front, all_pareto_fronts


def test_pareto_front_simple():
    # Point (1, 0.9) dominates (2, 0.8) — lower x, higher y
    df = pd.DataFrame({
        "latency_p95_ms": [1.0, 2.0, 3.0],
        "quality_score": [0.9, 0.8, 0.7],
    })
    front = pareto_front(df, "latency_p95_ms", "quality_score")
    # Only (1.0, 0.9) is Pareto-optimal
    assert len(front) == 1
    assert front.iloc[0]["latency_p95_ms"] == 1.0


def test_pareto_front_two_points_non_dominated():
    # (1, 0.5) and (2, 0.9) are both Pareto-optimal (trade-off)
    df = pd.DataFrame({
        "latency_p95_ms": [1.0, 2.0],
        "quality_score": [0.5, 0.9],
    })
    front = pareto_front(df, "latency_p95_ms", "quality_score")
    assert len(front) == 2


def test_pareto_front_empty_returns_empty():
    df = pd.DataFrame({"latency_p95_ms": [None], "quality_score": [None]})
    front = pareto_front(df, "latency_p95_ms", "quality_score")
    assert front.empty


def test_all_pareto_fronts_keys():
    df = pd.DataFrame({
        "tpot_ms_median": [10.0],
        "peak_rss_mb": [512.0],
        "estimated_joules_per_query": [0.5],
        "quality_score": [0.8],
    })
    fronts = all_pareto_fronts(df)
    assert set(fronts.keys()) == {"quality_vs_latency", "quality_vs_ram", "quality_vs_energy"}
