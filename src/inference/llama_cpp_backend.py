"""llama-cpp-python inference backend (Metal on Apple Silicon)."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Optional

from src.inference.base import GenerationResult, InferenceBackend


class LlamaCppBackend(InferenceBackend):
    """Wraps llama-cpp-python Llama for GGUF inference.

    On macOS with Apple Silicon, compiled with GGML_METAL=on, all layers
    are offloaded to the GPU by default (n_gpu_layers=-1).
    """

    def __init__(self) -> None:
        self._model: Optional[object] = None  # llama_cpp.Llama at runtime
        self._n_gpu_layers: int = -1
        self._chat_template: Optional[str] = None

    def load(
        self,
        gguf_path: Path,
        n_ctx: int = 2048,
        n_gpu_layers: int = -1,
        chat_template: Optional[str] = None,
    ) -> None:
        """Load a GGUF model via llama-cpp-python."""
        try:
            from llama_cpp import Llama  # type: ignore[import]
        except ImportError as e:
            raise ImportError(
                "llama-cpp-python not installed. Install with: "
                'CMAKE_ARGS="-DGGML_METAL=on" pip install llama-cpp-python'
            ) from e

        self._n_gpu_layers = n_gpu_layers
        self._chat_template = chat_template
        self._model = Llama(
            model_path=str(gguf_path),
            n_ctx=n_ctx,
            n_gpu_layers=n_gpu_layers,
            verbose=False,
        )

    def generate(
        self,
        prompt: str,
        max_tokens: int = 512,
        temperature: float = 0.0,
        seed: int = 42,
    ) -> GenerationResult:
        """Run llama.cpp inference with timing."""
        if self._model is None:
            raise RuntimeError("Model not loaded. Call load() first.")

        from src.inference.chat_templates import format_chat
        prompt = format_chat(prompt, self._chat_template)

        t0 = time.perf_counter()
        output = self._model(
            prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_k=1 if temperature == 0.0 else 40,
            seed=seed,
            echo=False,
        )
        t_total = (time.perf_counter() - t0) * 1000  # ms

        text = output["choices"][0]["text"]
        usage = output.get("usage", {})
        prompt_tokens = usage.get("prompt_tokens", 0)
        completion_tokens = usage.get("completion_tokens", len(text.split()))

        # TPOT = total generation time / output tokens.
        # TTFT is not measurable without streaming mode; set to None.
        tpot = t_total / max(completion_tokens, 1)

        return GenerationResult(
            text=text,
            ttft_ms=None,
            tpot_ms=tpot,
            total_tokens=prompt_tokens + completion_tokens,
            prompt_tokens=prompt_tokens,
        )

    def unload(self) -> None:
        """Free model from memory."""
        self._model = None
