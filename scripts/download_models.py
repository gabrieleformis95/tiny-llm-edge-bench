"""Download GGUFs from HuggingFace Hub to local cache.

Usage:
  python scripts/download_models.py --model qwen2.5-0.5b-instruct --quant Q4_K_M
  python scripts/download_models.py --all  # download everything in registry
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1]))

from src.registry.downloader import download_gguf
from src.registry.models import load_models, load_quants


def main() -> None:
    parser = argparse.ArgumentParser(description="Download GGUF models from HF")
    parser.add_argument("--model", default=None, help="Model id")
    parser.add_argument("--quant", default=None, help="Quant name")
    parser.add_argument("--all", action="store_true", help="Download all combinations")
    args = parser.parse_args()

    if args.all:
        for model in load_models():
            for quant in load_quants():
                download_gguf(model.id, quant.name)
    elif args.model and args.quant:
        path = download_gguf(args.model, args.quant)
        print(f"Downloaded to {path}")
    else:
        parser.error("Provide --model + --quant or --all")


if __name__ == "__main__":
    main()
