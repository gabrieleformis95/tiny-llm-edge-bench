#!/usr/bin/env bash
# Run the full model x quant x task matrix sequentially.
# Skips any run whose result JSON already exists (idempotent).
# Deletes each GGUF after all tasks for that (model, quant) pair complete.
set -euo pipefail

GGUF_CACHE="${HOME}/.cache/tiny-llm-edge-bench/models"

MODELS=(
  "qwen2.5-0.5b-instruct"
  "llama-3.2-1b-instruct"
  "qwen2.5-3b-instruct"
  "phi-3.5-mini-instruct"
)

QUANTS=(
  "Q8_0"
  "Q5_K_M"
  "Q4_K_M"
  "Q3_K_M"
  "Q2_K"
)

TASKS=(
  "none"
  "ragas_industrial"
  "mmlu_subset"
  "json_following"
  "instruction_follow"
)

for model in "${MODELS[@]}"; do
  for quant in "${QUANTS[@]}"; do
    # bartowski/Llama-3.2-1B-Instruct-GGUF does not publish Q3_K_M or Q2_K
    if [[ "$model" == "llama-3.2-1b-instruct" && ("$quant" == "Q3_K_M" || "$quant" == "Q2_K") ]]; then
      continue
    fi
    for task in "${TASKS[@]}"; do
      echo "==> $model x $quant x $task"
      # run_bench.py prints "[skip]" and exits 0 if a matching JSON already exists
      .venv/bin/python scripts/run_bench.py \
        --model "$model" \
        --quant "$quant" \
        --task "$task" \
        || echo "[WARN] run failed: $model $quant $task -- continuing"
    done
    # Free disk space: delete GGUF after all tasks for this (model, quant) are done
    gguf_dir="${GGUF_CACHE}/${model}/${quant}"
    if [[ -d "$gguf_dir" ]]; then
      echo "[cleanup] Removing ${gguf_dir}"
      rm -rf "$gguf_dir"
    fi
  done
done

echo "==> Aggregating results..."
.venv/bin/python -m src.cli aggregate
echo "Done."
