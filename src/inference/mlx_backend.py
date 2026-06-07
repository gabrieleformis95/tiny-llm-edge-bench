"""MLX inference backend for Apple Silicon (upper-bound comparison)."""

from __future__ import annotations

from pathlib import Path

from src.inference.base import GenerationResult, InferenceBackend


class MLXBackend(InferenceBackend):
    """Wraps mlx-lm for native Apple Silicon inference.

    Used as upper-bound performance comparison against llama-cpp-python + Metal.
    Only available on macOS arm64 with mlx-lm installed.
    """

    def __init__(self) -> None:
        self._model: object | None = None
        self._tokenizer: object | None = None

    def load(
        self,
        gguf_path: Path,
        n_ctx: int = 2048,
        n_gpu_layers: int = -1,
        chat_template: str | None = None,
    ) -> None:
        """Load an MLX model from HF repo derived from gguf_path parent directory name."""
        try:
            from mlx_lm import load as mlx_load  # type: ignore[import]
        except ImportError as e:
            raise ImportError("mlx-lm not installed. Install with: pip install mlx-lm") from e

        # gguf_path parent contains the model id used as HF repo
        hf_repo = str(gguf_path)
        self._model, self._tokenizer = mlx_load(hf_repo)

    def generate(
        self,
        prompt: str,
        max_tokens: int = 512,
        temperature: float = 0.0,
        seed: int = 42,
    ) -> GenerationResult:
        """Run mlx_lm.generate with timing."""
        if self._model is None:
            raise RuntimeError("Model not loaded.")

        import time

        try:
            from mlx_lm import generate as mlx_generate  # type: ignore[import]
        except ImportError as e:
            raise ImportError("mlx-lm not installed") from e

        t0 = time.perf_counter()
        text = mlx_generate(
            self._model,
            self._tokenizer,
            prompt=prompt,
            max_tokens=max_tokens,
            temp=temperature,
            verbose=False,
        )
        t_total = (time.perf_counter() - t0) * 1000
        tokens = len(text.split())
        tpot = t_total / max(tokens, 1)

        return GenerationResult(
            text=text,
            ttft_ms=None,
            tpot_ms=tpot,
            total_tokens=tokens,
            prompt_tokens=0,
        )

    def unload(self) -> None:
        """Clear MLX model from unified memory."""
        self._model = None
        self._tokenizer = None
