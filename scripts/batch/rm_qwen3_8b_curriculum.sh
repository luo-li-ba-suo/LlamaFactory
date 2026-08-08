#!/usr/bin/env bash
# Run all nine Qwen3-8B SfL margin curriculum experiments sequentially on GPUs 0-7.
set -euo pipefail

cd "$(dirname "$0")/../.."

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export FORCE_TORCHRUN=1
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

run() {
  local config="$1"
  export WANDB_RUN_GROUP="${config%.yaml}"
  echo "[$(date -Is)] Starting ${config}"
  llamafactory-cli train "$config"
}

run examples/train_sfl/curriculum_qwen3_8b_sfl_margin_full_original.yaml
run examples/train_sfl/curriculum_qwen3_8b_sfl_margin_full_balanced.yaml
run examples/train_sfl/curriculum_qwen3_8b_sfl_margin_b1.yaml
run examples/train_sfl/curriculum_qwen3_8b_sfl_margin_b2.yaml
run examples/train_sfl/curriculum_qwen3_8b_sfl_margin_b3.yaml
run examples/train_sfl/curriculum_qwen3_8b_sfl_margin_b4.yaml
run examples/train_sfl/curriculum_qwen3_8b_sfl_margin_b5.yaml
run examples/train_sfl/curriculum_qwen3_8b_sfl_margin_b3_b4.yaml
run examples/train_sfl/curriculum_qwen3_8b_sfl_margin_b2_b3_b4.yaml
