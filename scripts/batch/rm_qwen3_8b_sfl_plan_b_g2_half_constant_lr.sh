#!/usr/bin/env bash
# Reproduce the G2_half + G3_2 Plan B experiment at three constant learning rates.
set -euo pipefail

cd /workspace/projects/LlamaFactory
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export FORCE_TORCHRUN=1
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

run() {
  local config="$1"
  export WANDB_RUN_GROUP="g2_half_plus_g3_2_constant_lr"
  echo "[$(date -Is)] Starting ${config}"
  llamafactory-cli train "${config}"
}

run examples/train_sfl/curriculum_qwen3_8b_sfl_plan_b_g2_half_plus_g3_2_constant_lr_1e5.yaml
run examples/train_sfl/curriculum_qwen3_8b_sfl_plan_b_g2_half_plus_g3_2_constant_lr_2e5.yaml
run examples/train_sfl/curriculum_qwen3_8b_sfl_plan_b_g2_half_plus_g3_2_constant_lr_3e5.yaml
