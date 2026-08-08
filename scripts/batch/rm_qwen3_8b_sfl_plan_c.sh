#!/usr/bin/env bash
# Run the three Plan C Qwen3-8B SfL curriculum groups sequentially on GPUs 0-7.
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

run examples/train_sfl/curriculum_qwen3_8b_sfl_plan_c_g1_raw_random.yaml
run examples/train_sfl/curriculum_qwen3_8b_sfl_plan_c_g2_teacher_filtered_random.yaml
run examples/train_sfl/curriculum_qwen3_8b_sfl_plan_c_g3_teacher_filtered_active.yaml
