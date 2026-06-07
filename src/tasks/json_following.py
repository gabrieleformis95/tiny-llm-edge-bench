"""JSON structured output adherence task.

20 prompts asking for a JSON object with a specific schema.
Metric: json_validity rate (lenient parser handles markdown fences).
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Iterator


def _extract_first_object(text: str) -> str | None:
    """Return the first balanced {...} block, correctly handling strings and escapes."""
    start = text.find("{")
    if start < 0:
        return None
    depth = 0
    in_string = False
    i = start
    while i < len(text):
        ch = text[i]
        if in_string:
            if ch == "\\":
                i += 2
                continue
            if ch == '"':
                in_string = False
        else:
            if ch == '"':
                in_string = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return text[start : i + 1]
        i += 1
    return None


def parse_json_lenient(text: str) -> dict | None:
    """Extract and parse the first JSON object from a model response.

    Handles: markdown fences, preamble text, multiple objects, nested objects.
    """
    text = text.strip()

    # 1. Markdown fence: extract object inside ``` ... ```
    fence_match = re.search(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL)
    if fence_match:
        candidate = _extract_first_object(fence_match.group(1))
        if candidate:
            try:
                return json.loads(candidate)
            except json.JSONDecodeError:
                pass

    # 2. Clean JSON with no preamble
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # 3. Extract first balanced {...} (handles preamble / multiple objects)
    candidate = _extract_first_object(text)
    if candidate:
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            pass

    return None


class JSONFollowingTask:
    task_id = "json_following"

    def __init__(self, dataset_path: Path) -> None:
        self.dataset_path = dataset_path

    def iter_samples(self) -> Iterator[dict]:
        """Yield {id, instruction, required_keys} dicts."""
        with open(self.dataset_path) as f:
            data = json.load(f)
        for item in data:
            yield {
                "id": item["id"],
                "instruction": item["instruction"],
                "required_keys": item["required_keys"],
                "reference": json.dumps(item["required_keys"]),  # for score()
            }

    def build_prompt(self, sample: dict) -> str:
        """Return prompt with instruction to output JSON."""
        return (
            f"Output ONLY a valid JSON object, no explanation, no markdown fences.\n\n"
            f"Instruction: {sample['instruction']}"
        )

    def score(self, prediction: str, reference: str | list[str]) -> float:
        """Return 1.0 if output is valid JSON with all required top-level keys."""
        required_keys: list[str]
        if isinstance(reference, str):
            try:
                required_keys = json.loads(reference)
            except json.JSONDecodeError:
                required_keys = []
        else:
            required_keys = list(reference)

        parsed = parse_json_lenient(prediction)
        if parsed is None:
            return 0.0
        missing = [k for k in required_keys if k not in parsed]
        if missing:
            return (len(required_keys) - len(missing)) / len(required_keys)
        return 1.0
