"""Task protocol: any evaluation task must implement this interface."""

from __future__ import annotations

from typing import Iterator, Protocol, runtime_checkable


@runtime_checkable
class Task(Protocol):
    task_id: str

    def iter_samples(self) -> Iterator[dict]:
        """Yield one sample dict per evaluation example.

        Each dict must contain at least:
          - "prompt": str  - the full prompt to send to the LLM
          - "reference": str | list[str]  - expected output for scoring
        """
        ...

    def score(self, prediction: str, reference: str | list[str]) -> float:
        """Return a score in [0, 1] for a single prediction."""
        ...

    def build_prompt(self, sample: dict) -> str:
        """Convert a raw sample dict into a model-ready prompt string."""
        ...
