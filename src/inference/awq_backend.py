"""AWQ-INT4 inference backend via autoawq.

Platform note: autoawq requires CUDA. On Apple Silicon (MPS / CPU-only),
this backend raises PlatformError at load() time. Run on a CUDA-capable
Linux machine or via Docker (see Dockerfile). The backend is fully implemented
so results are reproducible on Linux; the Apple Silicon limitation is documented
in README section "GGUF vs AWQ".
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from src.inference.base import GenerationResult, InferenceBackend


class PlatformError(RuntimeError):
    """Raised when autoawq is unavailable on the current platform."""


class AWQBackend(InferenceBackend):
    """Wraps AutoAWQ for INT4 activation-aware quantized inference.

    Expects a directory produced by scripts/quantize_awq.py or a pre-quantized
    HF repo (e.g. TheBloke/*-AWQ). Exposes the same InferenceBackend interface
    as LlamaCppBackend for drop-in comparison.
    """

    def __init__(self) -> None:
        self._model: Any | None = None
        self._tokenizer: Any | None = None
        self._model_path: str | None = None

    def load(
        self,
        gguf_path: Path,
        n_ctx: int = 2048,
        n_gpu_layers: int = -1,
        chat_template: str | None = None,
    ) -> None:
        """Load an AWQ model from a local directory or HF repo id.

        gguf_path is reused as the model path / repo id for interface compatibility.
        For AWQ models this should point to the quantized directory, not a .gguf file.
        """
        try:
            import torch  # noqa: F401  (availability check; re-imported where used)
            from awq import AutoAWQForCausalLM  # type: ignore[import]
            from transformers import AutoTokenizer  # type: ignore[import]
        except ImportError as exc:
            raise PlatformError(
                "autoawq not installed or CUDA unavailable. "
                "AWQ inference requires a CUDA-capable GPU. "
                "On Apple Silicon, use LlamaCppBackend (GGUF Q4_K_M) instead. "
                "To run AWQ on Linux: pip install autoawq && "
                "python scripts/quantize_awq.py --model phi-3.5-mini-instruct"
            ) from exc

        model_path = str(gguf_path)
        self._model_path = model_path
        self._tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self._model = AutoAWQForCausalLM.from_quantized(
            model_path,
            fuse_layers=True,
            trust_remote_code=True,
            safetensors=True,
        )

    def generate(
        self,
        prompt: str,
        max_tokens: int = 512,
        temperature: float = 0.0,
        seed: int = 42,
    ) -> GenerationResult:
        if self._model is None or self._tokenizer is None:
            raise RuntimeError("Model not loaded. Call load() first.")

        import torch

        inputs = self._tokenizer(prompt, return_tensors="pt").to(self._model.device)
        prompt_tokens = inputs["input_ids"].shape[-1]

        t0 = time.perf_counter()
        with torch.inference_mode():
            output_ids = self._model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                do_sample=temperature > 0.0,
                temperature=temperature if temperature > 0.0 else None,
            )
        t_total = (time.perf_counter() - t0) * 1000

        completion_ids = output_ids[0][prompt_tokens:]
        text = self._tokenizer.decode(completion_ids, skip_special_tokens=True)
        completion_tokens = len(completion_ids)
        tpot = t_total / max(completion_tokens, 1)

        return GenerationResult(
            text=text,
            ttft_ms=None,
            tpot_ms=tpot,
            total_tokens=prompt_tokens + completion_tokens,
            prompt_tokens=prompt_tokens,
        )

    def unload(self) -> None:
        self._model = None
        self._tokenizer = None
