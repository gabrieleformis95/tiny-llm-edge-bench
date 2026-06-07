"""Download a 200-question MMLU subset from HuggingFace and save to data/golden/.

Uses the Cais/mmlu dataset (test split). Samples 200 questions evenly across
a fixed selection of subjects relevant to technical/STEM domains.

Usage:
  python scripts/download_mmlu_subset.py
"""

from __future__ import annotations

import json
import random
from pathlib import Path

OUT_PATH = Path(__file__).parents[1] / "data" / "golden" / "mmlu_subset.json"

SUBJECTS = [
    "abstract_algebra",
    "college_computer_science",
    "college_mathematics",
    "college_physics",
    "computer_security",
    "electrical_engineering",
    "elementary_mathematics",
    "high_school_computer_science",
    "high_school_physics",
    "machine_learning",
]

N_TOTAL = 200
N_PER_SUBJECT = N_TOTAL // len(SUBJECTS)  # 20 per subject

CHOICES_LABELS = ["A", "B", "C", "D"]


def main() -> None:
    try:
        from datasets import load_dataset  # type: ignore[import]
    except ImportError as exc:
        raise ImportError('pip install "tiny-llm-edge-bench[data]"') from exc

    rows: list[dict] = []
    rng = random.Random(42)

    for subject in SUBJECTS:
        ds = load_dataset("cais/mmlu", subject, split="test")
        items = list(ds)
        sample = rng.sample(items, min(N_PER_SUBJECT, len(items)))
        for i, item in enumerate(sample):
            rows.append(
                {
                    "id": f"{subject}_{i:03d}",
                    "question": item["question"],
                    "choices": item["choices"],
                    "answer": CHOICES_LABELS[item["answer"]],
                    "subject": subject,
                }
            )

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(rows, indent=2, ensure_ascii=False))
    print(f"Saved {len(rows)} questions to {OUT_PATH}")


if __name__ == "__main__":
    main()
