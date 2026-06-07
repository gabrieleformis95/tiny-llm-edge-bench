"""Human calibration of the RAGAS / ROUGE-L judge.

Workflow:
  1. Generate model predictions for all golden examples (cached to
     data/golden/predicted_responses.json after first run).
  2. Interactive CLI: present each example, collect human 1-5 rating.
  3. Compute Spearman r(human, ROUGE-L) and save data/golden/human_ratings.csv.

Usage:
  python src/tasks/ragas_human_audit.py                      # generate + rate
  python src/tasks/ragas_human_audit.py --skip-generate      # rate only (needs cache)
  python src/tasks/ragas_human_audit.py --skip-rate          # generate only
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).parents[2]
sys.path.insert(0, str(ROOT))

from src.tasks.ragas_industrial import RagasIndustrialTask, _rouge_l_f1

_GOLDEN = ROOT / "data/golden/ragas_golden.json"
_PREDICTIONS_CACHE = ROOT / "data/golden/predicted_responses.json"
_RATINGS_CSV = ROOT / "data/golden/human_ratings.csv"
_DEFAULT_MODEL = "phi-3.5-mini-instruct"
_DEFAULT_QUANT = "Q4_K_M"
_MAX_NEW_TOKENS = 150


def _generate_predictions(model_id: str, quant: str) -> list[dict]:
    from src.registry.downloader import download_gguf
    from llama_cpp import Llama

    task = RagasIndustrialTask(_GOLDEN)
    samples = list(task.iter_samples())

    print(f"Loading {model_id} {quant}...")
    gguf_path = download_gguf(model_id, quant)
    llm = Llama(
        model_path=str(gguf_path),
        n_ctx=2048,
        n_gpu_layers=-1,
        verbose=False,
    )

    results = []
    for i, sample in enumerate(samples):
        print(f"  [{i+1}/{len(samples)}] {sample['question'][:70]}...")
        prompt = task.build_prompt(sample)
        resp = llm(prompt, max_tokens=_MAX_NEW_TOKENS, temperature=0.0, echo=False)
        prediction = resp["choices"][0]["text"].strip()
        rouge_l = _rouge_l_f1(prediction, sample["reference"])
        results.append({
            "id": sample["id"],
            "question": sample["question"],
            "reference": sample["reference"],
            "prediction": prediction,
            "rouge_l": round(rouge_l, 4),
        })
        print(f"         ROUGE-L: {rouge_l:.3f}")

    del llm
    return results


def _rating_session(predictions: list[dict]) -> list[dict]:
    print()
    print("=" * 64)
    print("RAGAS Human Calibration")
    print("Rate prediction faithfulness vs reference:")
    print("  1=poor  2=fair  3=adequate  4=good  5=excellent")
    print("=" * 64)

    rated = []
    for i, p in enumerate(predictions):
        print(f"\n--- Example {i+1}/{len(predictions)} ---")
        print(f"Question : {p['question']}")
        print(f"\nReference: {p['reference'][:350]}")
        print(f"\nPredicted: {p['prediction'][:400]}")
        print(f"\n[auto ROUGE-L: {p['rouge_l']:.3f}]")

        while True:
            try:
                raw = input("\nFaithfulness rating (1-5, or q to quit): ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nAborted.")
                sys.exit(0)
            if raw.lower() == "q":
                print("Quitting.")
                sys.exit(0)
            if raw in {"1", "2", "3", "4", "5"}:
                break
            print("  Enter a single digit 1-5.")

        rated.append({**p, "human_rating": int(raw)})

    return rated


def _spearman(rated: list[dict]) -> float:
    from scipy.stats import spearmanr
    human = [r["human_rating"] for r in rated]
    auto = [r["rouge_l"] for r in rated]
    r, _ = spearmanr(human, auto)
    return float(r)


def main() -> float | None:
    parser = argparse.ArgumentParser(description="Human audit of RAGAS judge")
    parser.add_argument("--model", default=_DEFAULT_MODEL)
    parser.add_argument("--quant", default=_DEFAULT_QUANT)
    parser.add_argument("--skip-generate", action="store_true",
                        help="Skip inference; load predictions from cache")
    parser.add_argument("--skip-rate", action="store_true",
                        help="Generate predictions only, do not rate")
    args = parser.parse_args()

    # --- Step 1: predictions ---
    if _PREDICTIONS_CACHE.exists() or args.skip_generate:
        if not _PREDICTIONS_CACHE.exists():
            print("ERROR: no predictions cache, remove --skip-generate")
            sys.exit(1)
        print(f"Loading cached predictions: {_PREDICTIONS_CACHE}")
        predictions = json.loads(_PREDICTIONS_CACHE.read_text())
    else:
        predictions = _generate_predictions(args.model, args.quant)
        _PREDICTIONS_CACHE.write_text(json.dumps(predictions, indent=2))
        print(f"Saved predictions: {_PREDICTIONS_CACHE}")

    if args.skip_rate:
        return None

    # --- Step 2: human ratings ---
    rated = _rating_session(predictions)

    # --- Step 3: save CSV ---
    _RATINGS_CSV.parent.mkdir(parents=True, exist_ok=True)
    fields = ["id", "question", "reference", "prediction", "rouge_l", "human_rating"]
    with open(_RATINGS_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rated)
    print(f"\nSaved: {_RATINGS_CSV}")

    # --- Step 4: Spearman ---
    r = _spearman(rated)
    n = len(rated)
    print(f"\nSpearman r(human, ROUGE-L) = {r:.3f}  (N={n})")
    if abs(r) < 0.5:
        print(
            "  NOTE: |r| < 0.5 -- ROUGE-L is a weak proxy for human faithfulness "
            "judgment on this industrial domain."
        )

    return r


if __name__ == "__main__":
    main()
