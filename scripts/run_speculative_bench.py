"""Speculative decoding experiment: Qwen2.5-0.5B (draft) -> Qwen2.5-3B (target).

Algorithm (greedy / temperature=0):
  1. Draft model autoregressively generates k tokens from current context.
  2. Target model scores (context + k draft tokens) in one forward pass.
  3. For each position i=0..k-1:
     - Accept draft token if argmax(target_logits[i]) == draft_token[i].
     - On first mismatch: use target argmax as correction; stop scanning.
  4. If all k accepted: append target argmax at position k (bonus token).
  5. Repeat from accepted prefix.

Metrics:
  - acceptance_rate: fraction of draft tokens accepted across all target calls
  - speculative_tps: output tokens / wall time (speculative loop)
  - baseline_tps: output tokens / wall time (standard autoregressive, target only)
  - speedup: speculative_tps / baseline_tps

DoD: one result JSON + README "Speculative decoding" subsection.
Reference: Leviathan et al. 2023, "Fast Inference from Transformers via Speculative Decoding."
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parents[1]))

from src.config import settings
from src.registry.downloader import download_gguf


# Standard prompt set for measurement (20-token median input)
_PROMPTS = [
    "Explain the difference between precision and recall in machine learning.",
    "What causes thermal throttling in embedded processors?",
    "Describe the main advantages of LSTM over vanilla RNN.",
    "What is the purpose of layer normalization in transformers?",
    "Explain quantization-aware training versus post-training quantization.",
]
_N_WARMUP = 2
_N_MEASURE = 5   # measured runs (one per prompt)
_DRAFT_K = 4     # draft tokens per target pass


def _argmax_logits(llm, n_vocab: int) -> int:
    """Return the argmax token from the last forward pass (temperature=0)."""
    import ctypes
    import llama_cpp.llama_cpp as lc
    logits_ptr = lc.llama_get_logits(llm._ctx.ctx)
    logits = np.frombuffer(
        (ctypes.c_float * n_vocab).from_address(ctypes.addressof(logits_ptr.contents)),
        dtype=np.float32,
    ).copy()
    return int(np.argmax(logits))


def _argmax_logits_ith(llm, pos: int, n_vocab: int) -> int:
    """Return argmax logits at batch position `pos` after a multi-token eval."""
    import ctypes
    import llama_cpp.llama_cpp as lc
    logits_ptr = lc.llama_get_logits_ith(llm._ctx.ctx, pos)
    logits = np.frombuffer(
        (ctypes.c_float * n_vocab).from_address(ctypes.addressof(logits_ptr.contents)),
        dtype=np.float32,
    ).copy()
    return int(np.argmax(logits))


def _load_model(model_id: str, quant_name: str, n_ctx: int = 512):
    """Download if needed and load a model."""
    from llama_cpp import Llama

    gguf_path = download_gguf(model_id, quant_name)
    llm = Llama(
        model_path=str(gguf_path),
        n_ctx=n_ctx,
        n_gpu_layers=0,   # CPU only to avoid Metal crash on multi-model
        verbose=False,
        logits_all=False,  # logits_all=True shifts llama_get_logits to position 0, not last
    )
    return llm


def _tokenize(llm, text: str) -> list[int]:
    return llm.tokenize(text.encode(), add_bos=True, special=True)


def _speculative_decode(
    draft, target, prompt_tokens: list[int], max_new_tokens: int, k: int = _DRAFT_K
) -> tuple[list[int], int, int]:
    """Run speculative decoding. Returns (output_tokens, n_accepted, n_total_draft).

    With logits_all=False, llama_get_logits always returns the last evaluated position.
    Target verification is done incrementally (k sequential single-token evals) rather
    than in one batch. This correctly measures acceptance_rate; speedup reflects CPU
    single-model constraints rather than ideal batch speedup.
    """
    n_vocab_d = draft.n_vocab()
    n_vocab_t = target.n_vocab()
    context = list(prompt_tokens)
    output_tokens = []
    n_accepted = 0
    n_total_draft = 0

    draft.reset()
    draft.eval(context)
    target.reset()
    target.eval(context)

    while len(output_tokens) < max_new_tokens:
        remaining = max_new_tokens - len(output_tokens)
        step_k = min(k, remaining)

        # --- Draft phase ---
        draft_tokens = []
        for _ in range(step_k):
            tok = _argmax_logits(draft, n_vocab_d)
            draft_tokens.append(tok)
            draft.eval([tok])
        n_total_draft += step_k

        # --- Target verification: incremental (one forward pass per draft token) ---
        # Target KV is at current context from previous iteration (or init).
        # _argmax_logits reads the last position, which is what comes after context.
        accept_idx = 0
        for j, dtok in enumerate(draft_tokens):
            target_tok = _argmax_logits(target, n_vocab_t)
            if target_tok == dtok:
                accept_idx = j + 1
                target.eval([dtok])
            else:
                break

        n_accepted += accept_idx

        accepted = draft_tokens[:accept_idx]
        output_tokens.extend(accepted)
        context.extend(accepted)

        # Bonus or correction token (target is already at the right state)
        extra_tok = _argmax_logits(target, n_vocab_t)
        output_tokens.append(extra_tok)
        context.append(extra_tok)
        target.eval([extra_tok])

        # Resync draft to current context
        draft.reset()
        draft.eval(context)

        if output_tokens and output_tokens[-1] == draft.token_eos():
            break

    return output_tokens, n_accepted, n_total_draft


def _baseline_decode(target, prompt_tokens: list[int], max_new_tokens: int) -> list[int]:
    """Standard autoregressive decoding with target only."""
    n_vocab = target.n_vocab()
    context = list(prompt_tokens)
    output_tokens = []

    target.reset()
    target.eval(context)

    for _ in range(max_new_tokens):
        tok = _argmax_logits(target, n_vocab)
        output_tokens.append(tok)
        if tok == target.token_eos():
            break
        target.eval([tok])

    return output_tokens


def _measure(
    draft_model_id: str = "qwen2.5-0.5b-instruct",
    target_model_id: str = "qwen2.5-3b-instruct",
    quant_name: str = "Q4_K_M",
    max_new_tokens: int = 50,
    k: int = _DRAFT_K,
) -> dict:
    print(f"Loading draft model: {draft_model_id} {quant_name}")
    draft = _load_model(draft_model_id, quant_name)
    print(f"Loading target model: {target_model_id} {quant_name}")
    target = _load_model(target_model_id, quant_name)

    # Warmup both models
    for prompt in _PROMPTS[:_N_WARMUP]:
        toks = _tokenize(target, prompt)
        _baseline_decode(target, toks, max_new_tokens=10)
        _baseline_decode(draft, _tokenize(draft, prompt), max_new_tokens=10)

    # Baseline measurement: target only
    baseline_tokens = 0
    baseline_start = time.perf_counter()
    for prompt in _PROMPTS[:_N_MEASURE]:
        toks = _tokenize(target, prompt)
        out = _baseline_decode(target, toks, max_new_tokens)
        baseline_tokens += len(out)
    baseline_elapsed = time.perf_counter() - baseline_start
    baseline_tps = baseline_tokens / baseline_elapsed if baseline_elapsed > 0 else 0

    # Speculative measurement
    spec_tokens = 0
    total_accepted = 0
    total_draft = 0
    spec_start = time.perf_counter()
    for prompt in _PROMPTS[:_N_MEASURE]:
        toks = _tokenize(target, prompt)
        out, n_acc, n_draft = _speculative_decode(draft, target, toks, max_new_tokens, k)
        spec_tokens += len(out)
        total_accepted += n_acc
        total_draft += n_draft
    spec_elapsed = time.perf_counter() - spec_start
    spec_tps = spec_tokens / spec_elapsed if spec_elapsed > 0 else 0

    acceptance_rate = total_accepted / total_draft if total_draft > 0 else 0
    speedup = spec_tps / baseline_tps if baseline_tps > 0 else 0

    return {
        "draft_model": draft_model_id,
        "target_model": target_model_id,
        "quant": quant_name,
        "draft_k": k,
        "n_prompts": _N_MEASURE,
        "max_new_tokens": max_new_tokens,
        "baseline_tps": round(baseline_tps, 2),
        "speculative_tps": round(spec_tps, 2),
        "speedup": round(speedup, 3),
        "acceptance_rate": round(acceptance_rate, 3),
        "total_draft_tokens": total_draft,
        "total_accepted_tokens": total_accepted,
        "notes": (
            f"Greedy (argmax) acceptance. k={k} draft tokens/pass. "
            "n_gpu_layers=0 (CPU only, avoids Metal multi-model crash). "
            "Target verification is incremental (k sequential single-token evals); "
            "acceptance_rate is exact but speedup does not reflect ideal batch speedup. "
            "Reference: Leviathan et al. 2023, 'Fast Inference from Transformers via Speculative Decoding.'"
        ),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--draft", default="qwen2.5-0.5b-instruct")
    parser.add_argument("--target", default="qwen2.5-3b-instruct")
    parser.add_argument("--quant", default="Q4_K_M")
    parser.add_argument("--max-tokens", type=int, default=50)
    parser.add_argument("--k", type=int, default=4, help="Draft tokens per pass")
    parser.add_argument("--out", default="results/speculative_decoding.json")
    args = parser.parse_args()

    result = _measure(args.draft, args.target, args.quant, args.max_tokens, args.k)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2))
    print(f"\nResult saved: {out_path}")
    print(f"Baseline:    {result['baseline_tps']:.2f} tok/s")
    print(f"Speculative: {result['speculative_tps']:.2f} tok/s")
    print(f"Speedup:     {result['speedup']:.3f}x")
    print(f"Acceptance:  {result['acceptance_rate']*100:.1f}%")


if __name__ == "__main__":
    main()
