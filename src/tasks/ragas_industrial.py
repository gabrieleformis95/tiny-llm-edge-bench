"""RAG-grounded QA task using the ragas_golden.json from predictive-maintenance-copilot.

Default metric: faithfulness via ROUGE-L F1 (lexical overlap with the grounded
reference) - this is what the published benchmark matrix uses.
Optional metric: RAGAS-style LLM-as-judge faithfulness (Groq Llama-3.3-70B),
used only when GROQ_API_KEY is set; otherwise ROUGE-L is the score.
ROUGE-L is always recorded in last_diagnostics.
"""

from __future__ import annotations

import hashlib
import json as _json
import re
import urllib.request
from pathlib import Path
from typing import Iterator

from src.config import settings

_CACHE_PATH = Path("data/golden/judge_cache.json")


# ---------------------------------------------------------------------------
# Cache helpers
# ---------------------------------------------------------------------------

def _load_cache() -> dict:
    if _CACHE_PATH.exists():
        try:
            return _json.loads(_CACHE_PATH.read_text())
        except Exception:
            return {}
    return {}


def _save_cache(cache: dict) -> None:
    _CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    _CACHE_PATH.write_text(_json.dumps(cache, indent=2))


# ---------------------------------------------------------------------------
# Groq API
# ---------------------------------------------------------------------------

# Minimum seconds between Groq calls. Free-tier token/minute caps are low; pacing
# proactively avoids 429 storms. Raise via _set_call_interval() for batch jobs.
_MIN_CALL_INTERVAL = 0.0
_last_call_ts = 0.0


def _set_call_interval(seconds: float) -> None:
    global _MIN_CALL_INTERVAL
    _MIN_CALL_INTERVAL = seconds


def _groq_call(messages: list[dict], max_tokens: int = 256) -> str:
    import time
    global _last_call_ts
    gap = time.time() - _last_call_ts
    if gap < _MIN_CALL_INTERVAL:
        time.sleep(_MIN_CALL_INTERVAL - gap)
    _last_call_ts = time.time()

    payload = _json.dumps({
        "model": "llama-3.3-70b-versatile",
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": 0.0,
    }).encode()
    req = urllib.request.Request(
        "https://api.groq.com/openai/v1/chat/completions",
        data=payload,
        headers={
            "Authorization": f"Bearer {settings.groq_api_key}",
            "Content-Type": "application/json",
            # Groq is behind Cloudflare, which 403s the default Python-urllib UA.
            "User-Agent": "tiny-llm-edge-bench/1.0",
        },
    )
    import time
    for attempt in range(10):
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                data = _json.loads(resp.read())
            return data["choices"][0]["message"]["content"].strip()
        except urllib.error.HTTPError as e:
            # Back off on rate limit (429) / transient server errors (5xx).
            if e.code in (429, 500, 502, 503) and attempt < 9:
                retry_after = e.headers.get("retry-after")
                # Cap the wait: a daily-quota 429 can carry a retry-after of hours.
                wait = min(float(retry_after), 60.0) if retry_after else min(2 ** attempt, 30)
                time.sleep(wait)
                continue
            raise
        except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
            # Network/read timeouts are transient: retry with backoff.
            if attempt < 9:
                time.sleep(min(2 ** attempt, 30))
                continue
            raise


# ---------------------------------------------------------------------------
# RAGAS faithfulness judge
# ---------------------------------------------------------------------------

def _judge_faithfulness(answer: str, contexts: list[str]) -> float:
    """Fraction of atomic statements in answer supported by contexts.

    Two-step: decompose answer -> atomic statements, then verify each.
    Results are cached in data/golden/judge_cache.json.
    """
    cache_key = hashlib.sha256(
        (answer + "|||" + "|||".join(contexts)).encode()
    ).hexdigest()
    cache = _load_cache()
    if cache_key in cache:
        return cache[cache_key]

    ctx_block = "\n".join(f"[{i+1}] {c}" for i, c in enumerate(contexts))

    # Step 1: decompose
    raw = _groq_call([{"role": "user", "content": (
        "Break the following answer into a numbered list of atomic factual statements. "
        "One statement per line. No bullets, no explanation.\n\nAnswer: " + answer[:1000]
    )}], max_tokens=300)

    statements = []
    for line in raw.splitlines():
        line = re.sub(r"^\d+[\.\)]\s*", "", line.strip())
        if line:
            statements.append(line)

    if not statements:
        cache[cache_key] = 0.0
        _save_cache(cache)
        return 0.0

    # Step 2: verify all statements in a single batched call (stays under the
    # free-tier token/minute budget; one call per statement blows it on long answers).
    numbered = "\n".join(f"{i+1}. {s}" for i, s in enumerate(statements))
    verdict = _groq_call([{"role": "user", "content": (
        f"Contexts:\n{ctx_block}\n\n"
        f"Statements:\n{numbered}\n\n"
        "For EACH numbered statement, decide if it is fully supported by the contexts "
        "above. Reply with one line per statement in the form '<number>: SUPPORTED' or "
        "'<number>: UNSUPPORTED'. No other text."
    )}], max_tokens=20 * len(statements) + 50)

    supported = 0
    for line in verdict.splitlines():
        up = line.upper()
        if "UNSUPPORTED" in up:
            continue
        if "SUPPORTED" in up:
            supported += 1

    result = supported / len(statements)
    cache[cache_key] = result
    _save_cache(cache)
    return result


# ---------------------------------------------------------------------------
# ROUGE-L (kept as secondary diagnostic)
# ---------------------------------------------------------------------------

def _rouge_l_f1(prediction: str, reference: str) -> float:
    pred_tokens = prediction.lower().split()
    ref_tokens = reference.lower().split()
    if not pred_tokens or not ref_tokens:
        return 0.0
    m, n = len(ref_tokens), len(pred_tokens)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if ref_tokens[i - 1] == pred_tokens[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    lcs = dp[m][n]
    recall = lcs / m
    precision = lcs / n
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


# ---------------------------------------------------------------------------
# Task
# ---------------------------------------------------------------------------

class RagasIndustrialTask:
    task_id = "ragas_industrial"

    def __init__(self, dataset_path: Path) -> None:
        self.dataset_path = dataset_path
        self._samples: list[dict] = []
        self._current_contexts: list[str] = []
        self.last_diagnostics: dict = {}

    def _load(self) -> None:
        with open(self.dataset_path) as f:
            self._samples = _json.load(f)

    def iter_samples(self) -> Iterator[dict]:
        if not self._samples:
            self._load()
        for s in self._samples:
            self._current_contexts = s.get("contexts", [])
            yield {
                "id": s.get("id", s.get("question", "")[:40]),
                "question": s["question"],
                "contexts": self._current_contexts,
                "reference": s.get("ground_truth", s.get("answer", "")),
            }

    def build_prompt(self, sample: dict) -> str:
        context_block = "\n\n".join(
            f"[Context {i+1}]: {c}" for i, c in enumerate(sample["contexts"])
        )
        return (
            "You are a helpful assistant. Answer the question using ONLY the provided context.\n\n"
            f"{context_block}\n\n"
            f"Question: {sample['question']}\n"
            "Answer:"
        )

    def score(self, prediction: str, reference: str | list[str]) -> float:
        """Default: ROUGE-L F1 (published metric). If GROQ_API_KEY + contexts are
        present, returns the optional LLM-as-judge faithfulness instead.
        ROUGE-L is always stored in last_diagnostics.
        """
        ref = reference if isinstance(reference, str) else " ".join(reference)
        rouge_l = _rouge_l_f1(prediction, ref)
        self.last_diagnostics = {"rouge_l": rouge_l}

        if settings.groq_api_key and self._current_contexts:
            try:
                judge_score = _judge_faithfulness(prediction, self._current_contexts)
                self.last_diagnostics["judge_faithfulness"] = judge_score
                return judge_score
            except Exception as exc:
                self.last_diagnostics["judge_error"] = str(exc)

        return rouge_l
