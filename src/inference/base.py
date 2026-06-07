"""Abstract inference backend interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path


@dataclass
class GenerationResult:
    text: str
    ttft_ms: float | None  # None unless backend uses streaming mode
    tpot_ms: float  # time-per-output-token in ms
    total_tokens: int
    prompt_tokens: int


class InferenceBackend(ABC):
    """Common interface for llama.cpp and MLX backends."""

    @abstractmethod
    def load(
        self,
        gguf_path: Path,
        n_ctx: int = 2048,
        n_gpu_layers: int = -1,
        chat_template: str | None = None,
    ) -> None:
        """Load model weights into memory."""
        raise NotImplementedError

    @abstractmethod
    def generate(
        self,
        prompt: str,
        max_tokens: int = 512,
        temperature: float = 0.0,
        seed: int = 42,
    ) -> GenerationResult:
        """Run inference and return a GenerationResult."""
        raise NotImplementedError

    @abstractmethod
    def unload(self) -> None:
        """Release model weights from memory."""
        raise NotImplementedError

    def __enter__(self) -> InferenceBackend:
        return self

    def __exit__(self, *_: object) -> None:
        self.unload()
