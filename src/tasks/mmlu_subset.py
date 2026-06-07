"""200-question MMLU subset task.

Scoring: forced single-letter (A/B/C/D) output, or log-prob based if available.
Falls back to first-token comparison if logits_all is unreliable.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator


class MMLUSubsetTask:
    task_id = "mmlu_subset"

    CHOICES = ["A", "B", "C", "D"]

    def __init__(self, dataset_path: Path, n_samples: int = 200) -> None:
        self.dataset_path = dataset_path
        self.n_samples = n_samples

    def iter_samples(self) -> Iterator[dict]:
        """Yield {id, question, choices, reference} dicts."""
        with open(self.dataset_path) as f:
            data = json.load(f)
        for item in data[: self.n_samples]:
            yield {
                "id": item.get("id", item["question"][:30]),
                "question": item["question"],
                "choices": item["choices"],  # list of 4 strings
                "reference": item["answer"],  # "A", "B", "C", or "D"
            }

    def build_prompt(self, sample: dict) -> str:
        """Format MMLU question with A/B/C/D choices."""
        choices_text = "\n".join(
            f"{letter}. {text}"
            for letter, text in zip(self.CHOICES, sample["choices"])
        )
        return (
            f"The following is a multiple-choice question. "
            f"Answer with a single letter: A, B, C, or D.\n\n"
            f"Question: {sample['question']}\n{choices_text}\nAnswer:"
        )

    def score(self, prediction: str, reference: str | list[str]) -> float:
        """Return 1.0 if first non-whitespace uppercase letter matches reference."""
        ref = reference if isinstance(reference, str) else str(reference[0])
        pred = prediction.strip()
        if not pred:
            return 0.0
        first_letter = pred[0].upper()
        return 1.0 if first_letter == ref.upper() else 0.0
