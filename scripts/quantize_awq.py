"""Download an FP16 model from HF and quantize it to AWQ-INT4.

PLATFORM NOTE: autoawq requires a CUDA GPU. This script will fail on
Apple Silicon. Run it on a Linux box with a GPU, or via the provided
Dockerfile:
  docker build -t tiny-bench-cuda -f Dockerfile .
  docker run --gpus all tiny-bench-cuda python scripts/quantize_awq.py --model phi-3.5-mini-instruct

Usage:
  python scripts/quantize_awq.py --model phi-3.5-mini-instruct
  python scripts/quantize_awq.py --model qwen2.5-3b-instruct --group-size 64
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1]))

# HF repo for FP16 base models (non-GGUF)
_FP16_REPOS = {
    "phi-3.5-mini-instruct": "microsoft/Phi-3.5-mini-instruct",
    "qwen2.5-0.5b-instruct": "Qwen/Qwen2.5-0.5B-Instruct",
    "qwen2.5-3b-instruct": "Qwen/Qwen2.5-3B-Instruct",
    "llama-3.2-1b-instruct": "meta-llama/Llama-3.2-1B-Instruct",
}


def quantize(model_id: str, group_size: int, output_dir: Path) -> None:
    try:
        from awq import AutoAWQForCausalLM  # type: ignore[import]
        from transformers import AutoTokenizer  # type: ignore[import]
    except ImportError as exc:
        sys.exit(
            "ERROR: autoawq not installed or CUDA unavailable.\n"
            "  AWQ quantization requires a CUDA GPU.\n"
            "  On Apple Silicon, use GGUF Q4_K_M from llama.cpp instead.\n"
            f"  Original error: {exc}"
        )

    hf_repo = _FP16_REPOS.get(model_id)
    if hf_repo is None:
        sys.exit(f"ERROR: no FP16 HF repo configured for {model_id!r}")

    quant_config = {
        "zero_point": True,
        "q_group_size": group_size,
        "w_bit": 4,
        "version": "GEMM",
    }

    print(f"Loading {hf_repo} ...")
    model = AutoAWQForCausalLM.from_pretrained(hf_repo, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(hf_repo, trust_remote_code=True)

    print(f"Quantizing to AWQ-INT4 (group_size={group_size}) ...")
    model.quantize(tokenizer, quant_config=quant_config)

    output_dir.mkdir(parents=True, exist_ok=True)
    model.save_quantized(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    print(f"Saved AWQ model to {output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Quantize a model to AWQ-INT4")
    parser.add_argument("--model", required=True, choices=list(_FP16_REPOS), help="Model id")
    parser.add_argument("--group-size", type=int, default=128, help="AWQ group size (default 128)")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory (default: ~/.cache/tiny-llm-edge-bench/models/<model>/AWQ-INT4)",
    )
    args = parser.parse_args()

    from src.config import settings
    out = args.output_dir or (settings.gguf_cache_dir / args.model / "AWQ-INT4")
    quantize(args.model, args.group_size, out)


if __name__ == "__main__":
    main()
