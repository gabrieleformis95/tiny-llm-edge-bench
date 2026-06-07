"""Generate README results section, HTML report, and matplotlib plots."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.config import settings

_REPORTS_DIR = settings.reports_dir
_README = Path("README.md")

_TABLE_START = "<!-- AUTO-GENERATED TABLE START -->"
_TABLE_END = "<!-- AUTO-GENERATED TABLE END -->"


def generate_report(
    df: pd.DataFrame,
    reports_dir: Path = _REPORTS_DIR,
    readme_path: Path = _README,
) -> None:
    """Generate all report artifacts from aggregated DataFrame."""
    import matplotlib
    matplotlib.use("Agg")

    reports_dir.mkdir(parents=True, exist_ok=True)

    _plot_pareto_quality_latency(df, reports_dir / "pareto_quality_latency.png")
    _plot_pareto_quality_ram(df, reports_dir / "pareto_quality_ram.png")
    _plot_pareto_quality_energy(df, reports_dir / "pareto_quality_energy.png")
    _plot_throughput_by_quant(df, reports_dir / "throughput_by_quant.png")
    _plot_quality_degradation(df, reports_dir / "quality_degradation.png")
    _plot_mcu_comparison(df, reports_dir / "mcu_comparison.png")

    table_md = _build_results_table(df)
    _update_readme_table(readme_path, table_md)
    _write_html(df, reports_dir / "index.html", reports_dir)


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def _plot_pareto_quality_latency(df: pd.DataFrame, out_path: Path) -> None:
    import matplotlib.pyplot as plt
    from src.analysis.pareto import pareto_front

    fig, ax = plt.subplots(figsize=(8, 5))
    sub = df.dropna(subset=["tpot_ms_median", "quality_score"])
    if sub.empty:
        ax.text(0.5, 0.5, "No quality data yet\n(run `make bench-all`)",
                ha="center", va="center", transform=ax.transAxes, fontsize=12, color="gray")
    else:
        for model_id, group in sub.groupby("model_id"):
            front = pareto_front(group, "tpot_ms_median", "quality_score")
            ax.plot(front["tpot_ms_median"], front["quality_score"], marker="o", label=model_id)
        ax.legend(fontsize=8)
    ax.set_xlabel("Token latency TPOT (ms/token)")
    ax.set_ylabel("Quality Score")
    ax.set_title("Pareto: Quality vs Latency")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _plot_pareto_quality_ram(df: pd.DataFrame, out_path: Path) -> None:
    import matplotlib.pyplot as plt
    from src.analysis.pareto import pareto_front

    fig, ax = plt.subplots(figsize=(8, 5))
    sub = df.dropna(subset=["peak_rss_mb", "quality_score"])
    if sub.empty:
        ax.text(0.5, 0.5, "No quality data yet\n(run `make bench-all`)",
                ha="center", va="center", transform=ax.transAxes, fontsize=12, color="gray")
    else:
        for model_id, group in sub.groupby("model_id"):
            front = pareto_front(group, "peak_rss_mb", "quality_score")
            ax.plot(front["peak_rss_mb"], front["quality_score"], marker="o", label=model_id)
        ax.legend(fontsize=8)
    ax.set_xlabel("Peak RAM (MB)")
    ax.set_ylabel("Quality Score")
    ax.set_title("Pareto: Quality vs RAM")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _plot_pareto_quality_energy(df: pd.DataFrame, out_path: Path) -> None:
    import matplotlib.pyplot as plt
    from src.analysis.pareto import pareto_front

    fig, ax = plt.subplots(figsize=(8, 5))
    sub = df.dropna(subset=["estimated_tokens_per_joule", "quality_score"])
    if sub.empty:
        ax.text(0.5, 0.5, "No quality data yet\n(run `make bench-all`)",
                ha="center", va="center", transform=ax.transAxes, fontsize=12, color="gray")
    else:
        for model_id, group in sub.groupby("model_id"):
            front = pareto_front(group, "estimated_joules_per_query", "quality_score")
            ax.plot(front["estimated_tokens_per_joule"], front["quality_score"],
                    marker="o", label=model_id)
        ax.legend(fontsize=8)
    ax.set_xlabel("Tokens/joule (estimated, higher=better)")
    ax.set_ylabel("Quality Score")
    ax.set_title("Pareto: Quality vs Energy Efficiency")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _plot_throughput_by_quant(df: pd.DataFrame, out_path: Path) -> None:
    import matplotlib.pyplot as plt
    import numpy as np

    sub = df.dropna(subset=["tps_median"])
    quant_order = ["Q2_K", "Q3_K_M", "Q4_K_M", "Q5_K_M", "Q8_0", "FP16"]
    quants = [q for q in quant_order if q in sub["quant_name"].values]
    if not quants:
        quants = sorted(sub["quant_name"].unique())
    models = sorted(sub["model_id"].unique())

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(quants))
    width = 0.8 / max(len(models), 1)
    for i, model in enumerate(models):
        vals = [sub[(sub["model_id"] == model) & (sub["quant_name"] == q)]["tps_median"].mean()
                for q in quants]
        ax.bar(x + i * width, vals, width, label=model)
    ax.set_xticks(x + width * len(models) / 2)
    ax.set_xticklabels(quants, rotation=30)
    ax.set_ylabel("Tokens/s (median)")
    ax.set_title("Throughput by Quantization")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _plot_quality_degradation(df: pd.DataFrame, out_path: Path) -> None:
    """Quality retention relative to the Q8_0 baseline (near-lossless proxy for FP16).

    FP16 is not benchmarked (not uniformly available across the model GGUF repos:
    phi-3.5-mini has no F16, qwen2.5-3b ships F16 sharded). Q8_0 is empirically
    near-lossless (<1% vs FP16) and is used as the 100% reference per (model, task).
    """
    import matplotlib.pyplot as plt

    sub = df.dropna(subset=["quant_bits", "quality_score"])
    fig, ax = plt.subplots(figsize=(8, 5))
    if sub.empty:
        ax.text(0.5, 0.5, "No quality data yet\n(run `make bench-all`)",
                ha="center", va="center", transform=ax.transAxes, fontsize=12, color="gray")
    else:
        # Per (model, task): normalize quality to that cell's Q8_0 score, then
        # average the retention across models at each quant level.
        rows = []
        for (_model, _task), grp in sub.groupby(["model_id", "task_id"]):
            base = grp[grp["quant_name"] == "Q8_0"]["quality_score"]
            if base.empty or base.iloc[0] == 0:
                continue
            ref = base.iloc[0]
            for _, r in grp.iterrows():
                rows.append((r["task_id"], r["quant_bits"], r["quality_score"] / ref))
        if rows:
            import pandas as _pd
            norm = _pd.DataFrame(rows, columns=["task_id", "quant_bits", "retention"])
            for task_id, group in norm.groupby("task_id"):
                agg = group.groupby("quant_bits")["retention"].mean().reset_index()
                agg = agg.sort_values("quant_bits")
                ax.plot(agg["quant_bits"], agg["retention"], marker="o",
                        label=str(task_id))
            ax.axhline(1.0, color="gray", linestyle="--", linewidth=1, alpha=0.6)
            ax.legend(fontsize=8)
    ax.set_xlabel("Quantization Bits")
    ax.set_ylabel("Quality retention (relative to Q8_0)")
    ax.set_title("Quality Degradation vs Quantization (baseline = Q8_0, near-lossless)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _plot_mcu_comparison(df: pd.DataFrame, out_path: Path) -> None:
    """Model size vs latency: Mac LLMs + Cortex-M4 LSTM on the same axes."""
    import matplotlib.pyplot as plt

    mcu_json = settings.results_dir / "mcu_benchmark.json"

    fig, ax = plt.subplots(figsize=(9, 5))

    # Mac LLMs: params_b (B) vs ms/token (1000 / tps)
    llm_sub = df.dropna(subset=["model_params_b", "tps_median"])
    if not llm_sub.empty:
        for _, row in llm_sub.iterrows():
            ms_per_tok = 1000.0 / row["tps_median"]
            label = f"{row['model_id'].split('-instruct')[0]} {row['quant_name']}"
            ax.scatter(row["model_params_b"] * 1e9, ms_per_tok,
                       marker="o", s=80, zorder=3)
            ax.annotate(label, (row["model_params_b"] * 1e9, ms_per_tok),
                        fontsize=7, textcoords="offset points", xytext=(5, 3))

    # Cortex-M4 LSTM (165K params, 13.5 ms/inference, ESTIMATED)
    if mcu_json.exists():
        mcu = json.loads(mcu_json.read_text())
        mcu_params = 165_000
        mcu_latency_ms = mcu.get("latency_us", 139404) / 1000.0
        sim = mcu.get("simulation_method", "")
        tag = "Renode sim" if "renode" in sim else "ESTIMATED"
        ax.scatter(mcu_params, mcu_latency_ms, marker="^", s=120, color="red", zorder=4)
        ax.annotate(f"LSTM-AE (Cortex-M4)\n[{tag}]",
                    (mcu_params, mcu_latency_ms),
                    fontsize=7, color="red", textcoords="offset points", xytext=(5, 3))

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Model parameters (log scale)")
    ax.set_ylabel("Latency ms/token or ms/inference (log scale)")
    ax.set_title("Mac LLMs vs Cortex-M4 LSTM: Model Size vs Latency")
    ax.grid(True, which="both", linestyle="--", alpha=0.4)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Results table + README update
# ---------------------------------------------------------------------------

def _build_results_table(df: pd.DataFrame) -> str:
    cols = ["model_id", "quant_name", "quant_bits", "tps_median",
            "tpot_ms_median", "peak_rss_mb", "quality_score",
            "estimated_joules_per_query"]
    sub = df[[c for c in cols if c in df.columns]].copy()
    sub = sub.dropna(subset=["tps_median"])

    if sub.empty:
        return "| No results yet — run `make bench-all` |\n|---|"

    agg = (
        sub.groupby(["model_id", "quant_name"], sort=False)
        .agg(
            quant_bits=("quant_bits", "first"),
            tps_median=("tps_median", "median"),
            tpot_ms_median=("tpot_ms_median", "median"),
            peak_rss_mb=("peak_rss_mb", "median"),
            quality_score=("quality_score", "mean"),
            estimated_joules_per_query=("estimated_joules_per_query", "first"),
        )
        .reset_index()
        .sort_values("tps_median", ascending=False)
    )

    header = ("| Model | Quant | Bits | tok/s (median) | TPOT (ms) "
              "| Peak RAM (MB) | Quality Score | J/query (est.) |")
    sep = "|---|---|---|---|---|---|---|---|"
    lines = [header, sep]

    def _fmt(v, fmt=".1f"):
        return f"{v:{fmt}}" if pd.notna(v) else "-"

    for _, row in agg.iterrows():
        lines.append(
            f"| {row['model_id']} | {row['quant_name']} "
            f"| {_fmt(row['quant_bits'], '.1f')} "
            f"| {_fmt(row['tps_median'], '.1f')} "
            f"| {_fmt(row['tpot_ms_median'], '.1f')} "
            f"| {_fmt(row['peak_rss_mb'], '.0f')} "
            f"| {_fmt(row['quality_score'], '.3f')} "
            f"| {_fmt(row['estimated_joules_per_query'], '.2f')} |"
        )
    lines.append("")
    lines.append("*Energy: ESTIMATED (analytical, Lai et al. 2018). "
                 "Quality: mean across tasks.*")
    return "\n".join(lines)


def _update_readme_table(readme_path: Path, table_md: str) -> None:
    if not readme_path.exists():
        return
    text = readme_path.read_text()
    if _TABLE_START not in text or _TABLE_END not in text:
        return
    before = text[: text.index(_TABLE_START) + len(_TABLE_START)]
    after = text[text.index(_TABLE_END):]
    readme_path.write_text(f"{before}\n{table_md}\n{after}")


# ---------------------------------------------------------------------------
# HTML report
# ---------------------------------------------------------------------------

def _write_html(df: pd.DataFrame, out_path: Path, reports_dir: Path) -> None:
    plots = [
        ("mcu_comparison.png", "Mac LLMs vs Cortex-M4 LSTM (model size vs latency)"),
        ("pareto_quality_latency.png", "Pareto: Quality vs Latency"),
        ("pareto_quality_ram.png", "Pareto: Quality vs RAM"),
        ("pareto_quality_energy.png", "Pareto: Quality vs Energy Efficiency"),
        ("throughput_by_quant.png", "Throughput by Quantization"),
        ("quality_degradation.png", "Quality Degradation vs Quantization (baseline = Q8_0)"),
        ("operator_breakdown.png", "Operator-Level Breakdown (Phi-3.5-mini)"),
    ]
    imgs = "".join(
        f"<h2>{title}</h2><img src=\"{name}\" style=\"max-width:100%;margin-bottom:2em\"><br>"
        for name, title in plots
        if (reports_dir / name).exists()
    )
    table_html = df.to_html(index=False, classes="table", border=0) if not df.empty else ""
    html = (
        "<!DOCTYPE html><html><head><meta charset='utf-8'>"
        "<title>tiny-llm-edge-bench report</title>"
        "<style>"
        "body{font-family:sans-serif;max-width:1400px;margin:auto;padding:1em}"
        "h1{border-bottom:2px solid #333}"
        ".table{border-collapse:collapse;width:100%;font-size:0.85em}"
        ".table td,.table th{border:1px solid #ccc;padding:4px 8px}"
        ".table tr:nth-child(even){background:#f5f5f5}"
        "</style></head>"
        "<body>"
        "<h1>tiny-llm-edge-bench</h1>"
        "<p>Reproducible benchmark suite for small open-source LLMs on edge hardware.</p>"
        f"{imgs}"
        f"<h2>Full Results</h2>{table_html}"
        "</body></html>"
    )
    out_path.write_text(html)
