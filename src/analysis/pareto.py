"""Compute Pareto fronts from aggregated benchmark results."""

from __future__ import annotations

import pandas as pd


def pareto_front(df: pd.DataFrame, x_col: str, y_col: str) -> pd.DataFrame:
    """Return the Pareto-optimal subset of df minimizing x_col and maximizing y_col."""
    clean = df[[x_col, y_col]].dropna()
    if clean.empty:
        return df.iloc[0:0]

    indices = clean.index.tolist()
    dominated = set()
    for i in indices:
        xi, yi = clean.loc[i, x_col], clean.loc[i, y_col]
        for j in indices:
            if i == j:
                continue
            xj, yj = clean.loc[j, x_col], clean.loc[j, y_col]
            # j dominates i if xj <= xi and yj >= yi (strictly better in at least one)
            if xj <= xi and yj >= yi and (xj < xi or yj > yi):
                dominated.add(i)
                break

    optimal_indices = [i for i in indices if i not in dominated]
    return df.loc[optimal_indices].sort_values(x_col)


def all_pareto_fronts(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Return Pareto fronts for all three standard trade-offs.

    Uses columns produced by aggregate._run_to_row:
      latency  -> tpot_ms_median (ms per output token)
      ram      -> peak_rss_mb
      energy   -> estimated_joules_per_query (analytical, always present)
    """
    return {
        "quality_vs_latency": pareto_front(df, "tpot_ms_median", "quality_score"),
        "quality_vs_ram": pareto_front(df, "peak_rss_mb", "quality_score"),
        "quality_vs_energy": pareto_front(df, "estimated_joules_per_query", "quality_score"),
    }
