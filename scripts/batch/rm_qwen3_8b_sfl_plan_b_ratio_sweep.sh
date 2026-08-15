#!/usr/bin/env bash
# Run the remaining Plan B G2:G3_2 ratio ablations sequentially on eight GPUs.
set -euo pipefail

cd /workspace/projects/LlamaFactory
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export FORCE_TORCHRUN=1
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

run() {
  local config="$1"
  export WANDB_RUN_GROUP="${config%.yaml}"
  echo "[$(date -Is)] Starting ${config}"
  llamafactory-cli train "${config}"
}

run examples/train_sfl/curriculum_qwen3_8b_sfl_plan_b_g2_g3_2_ratio_1_2.yaml
run examples/train_sfl/curriculum_qwen3_8b_sfl_plan_b_g2_g3_2_ratio_1_3.yaml
