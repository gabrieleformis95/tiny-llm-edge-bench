"""Tests for operator-level profiling (analytical bandwidth model)."""

from __future__ import annotations

import pytest

from src.benchmarks.profiling import OperatorProfile, estimate_operator_breakdown


def test_fracs_sum_to_one():
    p = estimate_operator_breakdown("phi-3.5-mini-instruct", 4.5, "Q4_K_M")
    total = (
        p.ffn_frac + p.attn_qkv_frac + p.attn_output_frac + p.attn_scores_frac + p.layernorm_frac
    )
    assert abs(total - 1.0) < 1e-9


def test_ffn_dominant():
    p = estimate_operator_breakdown("phi-3.5-mini-instruct", 4.5, "Q4_K_M")
    assert p.ffn_frac > 0.5, "FFN should dominate for Phi-3.5-mini"


def test_q4_scores_frac_larger_than_fp16():
    fp16 = estimate_operator_breakdown("phi-3.5-mini-instruct", 16.0, "FP16")
    q4 = estimate_operator_breakdown("phi-3.5-mini-instruct", 4.5, "Q4_K_M")
    assert q4.attn_scores_frac > fp16.attn_scores_frac, (
        "Q4 weight quantization should increase attn_scores fraction "
        "(KV cache reads are not weight-quantized)"
    )


def test_unknown_model_raises():
    with pytest.raises(ValueError, match="Unknown model"):
        estimate_operator_breakdown("nonexistent-model", 4.5)


def test_all_registered_models():
    for model_id in [
        "phi-3.5-mini-instruct",
        "qwen2.5-0.5b-instruct",
        "llama-3.2-1b-instruct",
        "qwen2.5-3b-instruct",
    ]:
        p = estimate_operator_breakdown(model_id, 4.5)
        assert isinstance(p, OperatorProfile)
        assert p.ffn_frac > 0


def test_measured_ms_stored():
    p = estimate_operator_breakdown("phi-3.5-mini-instruct", 4.5, measured_ms_per_token=58.8)
    assert p.total_ms_per_token == pytest.approx(58.8)
