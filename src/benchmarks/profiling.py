"""Operator-level decode time breakdown via memory-bandwidth model.

llama.cpp exposes only aggregate timing through llama_perf_context:
  t_eval_ms, t_p_eval_ms, n_eval, n_p_eval
No per-tensor timings are available from the Python bindings.

This module estimates per-operator fractions using a memory-bandwidth-bound
model for single-token decode on Apple Silicon:

  time(op) proportional to bytes_accessed(op) / bandwidth

Key insight: weight quantization reduces weight bytes ~4x (FP16 -> Q4);
KV cache reads (attention scores + softmax) stay in FP16 regardless.
Their fraction of total decode time grows as other ops get faster.

All output is labeled ESTIMATED.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional


@dataclass
class OperatorProfile:
    """Per-operator time fractions (sum to ~1.0). All values are ESTIMATED."""
    model_id: str
    quant_name: str
    quant_bits: float
    seq_len: int
    ffn_frac: float            # gate+up+down weight projections (SwiGLU)
    attn_qkv_frac: float       # QKV weight projections
    attn_output_frac: float    # attention output weight projection
    attn_scores_frac: float    # QK^T scores + softmax + V weighted sum (KV cache reads)
    layernorm_frac: float      # all layer norms (FP32, minor)
    total_ms_per_token: Optional[float] = None  # anchor from llama_perf_context
    method: str = "bandwidth_analytical"
    notes: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


# Architecture parameters: (n_layers, d_model, d_ffn, n_heads, n_kv_heads)
# Sources: official model cards and config.json on HuggingFace
_MODEL_ARCH: dict[str, dict] = {
    "phi-3.5-mini-instruct": {
        "n_layers": 32, "d_model": 3072, "d_ffn": 8192,
        "n_heads": 32, "n_kv_heads": 32,
    },
    "qwen2.5-0.5b-instruct": {
        "n_layers": 24, "d_model": 896, "d_ffn": 4864,
        "n_heads": 14, "n_kv_heads": 2,
    },
    "llama-3.2-1b-instruct": {
        "n_layers": 16, "d_model": 2048, "d_ffn": 8192,
        "n_heads": 32, "n_kv_heads": 8,
    },
    "qwen2.5-3b-instruct": {
        "n_layers": 36, "d_model": 2048, "d_ffn": 11008,
        "n_heads": 16, "n_kv_heads": 8,
    },
}


def estimate_operator_breakdown(
    model_id: str,
    quant_bits: float,
    quant_name: str = "",
    seq_len: int = 256,
    measured_ms_per_token: Optional[float] = None,
) -> OperatorProfile:
    """Estimate per-operator decode time breakdown from memory bandwidth model.

    For single-token autoregressive decode on memory-bandwidth-bound hardware
    (Apple Silicon M-series): time proportional to bytes accessed per step.

    Weight quantization reduces bytes per weight param:
      FP16 -> 2.0 B/param
      Q4_K_M (4.5 bits) -> 0.5625 B/param
    KV cache reads stay in FP16 (2 B/elem) regardless of weight quantization.
    """
    arch = _MODEL_ARCH.get(model_id)
    if arch is None:
        raise ValueError(f"Unknown model '{model_id}'. Add architecture to _MODEL_ARCH.")

    n_heads: int = arch["n_heads"]
    n_kv_heads: int = arch["n_kv_heads"]
    d_model: int = arch["d_model"]
    d_ffn: int = arch["d_ffn"]
    head_dim: int = d_model // n_heads

    bpw = quant_bits / 8  # bytes per weight param after quantization
    bpa = 2.0             # FP16 bytes per activation element (unchanged by quantization)

    # Weight-bound ops: benefit from quantization
    # Q, K, V projections: output dims = n_heads*head_dim, n_kv_heads*head_dim each
    b_qkv = (n_heads + 2 * n_kv_heads) * head_dim * d_model * bpw
    b_out = d_model * d_model * bpw
    # SwiGLU FFN: gate_proj + up_proj + down_proj
    b_ffn = 3 * d_model * d_ffn * bpw

    # KV cache reads: NOT reduced by weight quantization
    # For each decode step, read K and V caches of length seq_len
    b_kv = 2 * n_kv_heads * seq_len * head_dim * bpa

    # LayerNorm: FP32 scale+bias, 2 norms per layer (pre-attn and pre-FFN)
    b_ln = 2 * 4 * d_model  # FP32 = 4 bytes

    total = b_qkv + b_out + b_ffn + b_kv + b_ln

    return OperatorProfile(
        model_id=model_id,
        quant_name=quant_name or f"{quant_bits}bit",
        quant_bits=quant_bits,
        seq_len=seq_len,
        ffn_frac=b_ffn / total,
        attn_qkv_frac=b_qkv / total,
        attn_output_frac=b_out / total,
        attn_scores_frac=b_kv / total,
        layernorm_frac=b_ln / total,
        total_ms_per_token=measured_ms_per_token,
        method="bandwidth_analytical",
        notes=(
            f"ESTIMATED. Memory-bandwidth-bound model: time proportional to bytes accessed. "
            f"Weight: {bpw:.4f} B/param ({quant_bits} bits). "
            f"KV cache: FP16 ({bpa} B/elem), seq_len={seq_len}. "
            "llama_perf_context exposes only aggregate t_eval_ms/n_eval - "
            "no per-operator data available from llama.cpp Python bindings."
        ),
    )


def plot_operator_breakdown(profiles: list[OperatorProfile], out_path: Path) -> None:
    """Save stacked bar plot: % decode time per operator for each (model, quant) combo."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    operator_keys = [
        ("FFN (gate+up+down)", "ffn_frac", "#2563eb"),
        ("Attn QKV proj",      "attn_qkv_frac", "#16a34a"),
        ("Attn output proj",   "attn_output_frac", "#ca8a04"),
        ("Attn scores+softmax","attn_scores_frac", "#dc2626"),
        ("LayerNorm",          "layernorm_frac", "#9333ea"),
    ]

    x = np.arange(len(profiles))
    labels = [p.quant_name for p in profiles]

    fig, ax = plt.subplots(figsize=(7, 5))
    bottoms = np.zeros(len(profiles))

    for name, attr, color in operator_keys:
        vals = np.array([getattr(p, attr) * 100 for p in profiles])
        ax.bar(x, vals, bottom=bottoms, label=name, color=color, width=0.55, alpha=0.9)
        for i, (v, b) in enumerate(zip(vals, bottoms)):
            if v > 2.5:
                ax.text(
                    x[i], b + v / 2, f"{v:.1f}%",
                    ha="center", va="center", fontsize=8.5,
                    color="white", fontweight="bold",
                )
        bottoms += vals

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel("Estimated % of decode time", fontsize=9)
    ax.set_title(
        f"Operator time breakdown - {profiles[0].model_id}\n"
        f"seq_len={profiles[0].seq_len}, memory-bandwidth model",
        fontsize=10,
    )
    ax.legend(loc="upper right", fontsize=8, framealpha=0.8)
    ax.set_ylim(0, 118)
    ax.set_yticks([])

    ax.text(
        0.5, -0.10,
        "ESTIMATED (analytical). llama.cpp does not expose per-operator timings.",
        transform=ax.transAxes, ha="center", fontsize=7,
        style="italic", color="#6b7280",
    )

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


def _load_measured_tps_from_results(model_id: str, quant_name: str) -> Optional[float]:
    """Scan results/*.json for a matching run and return measured tok/s."""
    import json
    from src.config import settings

    for f in settings.results_dir.glob("*.json"):
        if f.name == "aggregated.json":
            continue
        try:
            data = json.loads(f.read_text())
            if (
                data.get("model", {}).get("id") == model_id
                and data.get("quant", {}).get("name") == quant_name
            ):
                tps = data.get("throughput", {}).get("median_tok_per_s")
                if tps:
                    return 1000.0 / tps  # convert to ms/token
        except Exception:
            continue
    return None


if __name__ == "__main__":
    # Generate FP16 vs Q4_K_M comparison for Phi-3.5-mini
    model_id = "phi-3.5-mini-instruct"
    seq_len = 256

    measured_ms = _load_measured_tps_from_results(model_id, "Q4_K_M")

    profiles = [
        estimate_operator_breakdown(model_id, 16.0, "FP16", seq_len=seq_len),
        estimate_operator_breakdown(
            model_id, 4.5, "Q4_K_M", seq_len=seq_len,
            measured_ms_per_token=measured_ms,
        ),
    ]

    from pathlib import Path
    out = Path("reports/operator_breakdown.png")
    plot_operator_breakdown(profiles, out)

    for p in profiles:
        print(
            f"{p.quant_name}: FFN={p.ffn_frac*100:.1f}%  "
            f"Attn_scores={p.attn_scores_frac*100:.1f}%  "
            f"QKV={p.attn_qkv_frac*100:.1f}%  "
            f"Out={p.attn_output_frac*100:.1f}%"
        )
