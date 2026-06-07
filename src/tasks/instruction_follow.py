"""IFEval-style instruction following task.

30 prompts with length and format constraints.
Metric: compliance_rate (fraction of prompts where all constraints are satisfied).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator


class InstructionFollowTask:
    task_id = "instruction_follow"

    def __init__(self, dataset_path: Path) -> None:
        self.dataset_path = dataset_path

    def iter_samples(self) -> Iterator[dict]:
        """Yield {id, prompt, constraints} dicts."""
        with open(self.dataset_path) as f:
            data = json.load(f)
        for item in data:
            yield {
                "id": item["id"],
                "prompt": item["prompt"],
                "constraints": item["constraints"],
                "reference": json.dumps(item["constraints"]),
            }

    def build_prompt(self, sample: dict) -> str:
        """Return the raw prompt (constraints are already embedded in it)."""
        return sample["prompt"]

    def score(self, prediction: str, reference: str | list[str]) -> float:
        """Return fraction of constraints satisfied (0.0 to 1.0)."""
        if isinstance(reference, str):
            try:
                constraints = json.loads(reference)
            except json.JSONDecodeError:
                return 0.0
        else:
            constraints = list(reference)

        results = self._check_constraints(prediction, constraints)
        if not results:
            return 1.0
        return sum(results) / len(results)

    def _check_constraints(self, text: str, constraints: list[dict]) -> list[bool]:
        """Evaluate each constraint against text."""
        import re

        results = []
        words = text.split()
        sentences = [s.strip() for s in re.split(r"[.!?]+", text) if s.strip()]

        for c in constraints:
            kind = c.get("type", "")
            if kind == "word_count_max":
                results.append(len(words) <= c["value"])
            elif kind == "word_count_min":
                results.append(len(words) >= c["value"])
            elif kind == "sentence_count":
                results.append(len(sentences) == c["value"])
            elif kind == "line_count":
                lines = [l for l in text.splitlines() if l.strip()]
                results.append(len(lines) == c["value"])
            elif kind == "format":
                fmt = c["value"]
                if fmt == "numbered_list":
                    results.append(bool(re.search(r"^\s*\d+\.", text, re.MULTILINE)))
                elif fmt == "bullet_list":
                    results.append(bool(re.search(r"^\s*[-*]", text, re.MULTILINE)))
                else:
                    results.append(True)
            elif kind == "item_count":
                # Count bullet/numbered items
                items = re.findall(r"^\s*(?:\d+\.|[-*])\s+\S", text, re.MULTILINE)
                n = len(items)
                ok = True
                if "min" in c:
                    ok = ok and n >= c["min"]
                if "max" in c:
                    ok = ok and n <= c["max"]
                if "value" in c:
                    ok = ok and n == c["value"]
                results.append(ok)
            else:
                results.append(True)
        return results
