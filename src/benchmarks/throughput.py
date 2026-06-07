"""Measure throughput: tok/s, TTFT, TPOT over a standard prompt suite.

Protocol: 5 warmup runs (discarded) + 50 measured runs with 2 s inter-run
cooldown to reduce thermal correlation.
Prompts are cycled from the pool (5 short + 10 medium + 5 long = 20 prompts).
Reports median, IQR, std, min, max (not p95 -- N=50 too small for tail stats).
"""

from __future__ import annotations

import itertools
import json
import time
import warnings
from pathlib import Path

import numpy as np

from src.config import ThroughputResult
from src.inference.base import InferenceBackend

_PROMPTS_PATH = Path(__file__).parents[2] / "data" / "prompts" / "standard.json"

N_WARMUP = 5
N_MEASURED = 50
COOLDOWN_S = 2  # inter-run sleep to reduce thermal correlation
NOISE_THRESHOLD = 0.15  # warn if IQR/median exceeds this


def run_throughput_benchmark(
    backend: InferenceBackend,
    prompts_path: Path = _PROMPTS_PATH,
    max_tokens: int = 256,
) -> ThroughputResult:
    """5 warmup + 50 measured runs over the standard prompt pool (cycled).

    Warmup results are discarded. Reports median, IQR, std, min, max over tok/s.
    """
    with open(prompts_path) as f:
        prompts = [p["prompt"] for p in json.load(f)]

    prompt_cycle = itertools.cycle(prompts)

    # Warmup: discard
    for _ in range(N_WARMUP):
        backend.generate(next(prompt_cycle), max_tokens=max_tokens)

    # Measured runs with inter-run cooldown
    tps_samples: list[float] = []
    ttft_samples: list[float] = []
    tpot_samples: list[float] = []

    for i in range(N_MEASURED):
        if i > 0:
            time.sleep(COOLDOWN_S)
        prompt = next(prompt_cycle)
        t0 = time.perf_counter()
        result = backend.generate(prompt, max_tokens=max_tokens)
        elapsed_s = time.perf_counter() - t0

        output_tokens = result.total_tokens - result.prompt_tokens
        tps = output_tokens / elapsed_s if elapsed_s > 0 else 0.0
        tps_samples.append(tps)
        if result.ttft_ms is not None:
            ttft_samples.append(result.ttft_ms)
        tpot_samples.append(result.tpot_ms)

    tps_arr = np.array(tps_samples)
    median = float(np.median(tps_arr))
    q25, q75 = np.percentile(tps_arr, [25, 75])
    iqr = float(q75 - q25)

    if median > 0 and iqr / median > NOISE_THRESHOLD:
        warnings.warn(
            f"Throughput measurement is noisy: IQR/median = {iqr / median:.2f} "
            f"(threshold {NOISE_THRESHOLD}). Consider re-running or investigating "
            "thermal throttling.",
            stacklevel=2,
        )

    return ThroughputResult(
        n_warmup=N_WARMUP,
        n_measured=N_MEASURED,
        median_tok_per_s=median,
        iqr_tok_per_s=iqr,
        std_tok_per_s=float(np.std(tps_arr)),
        min_tok_per_s=float(tps_arr.min()),
        max_tok_per_s=float(tps_arr.max()),
        median_ttft_ms=float(np.median(ttft_samples)) if ttft_samples else None,
        median_tpot_ms=float(np.median(tpot_samples)),
        raw_samples=tps_samples,
    )
