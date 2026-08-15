#!/usr/bin/env bash
# Run the two batch-size variants strictly in sequence on the same eight GPUs.
set -euo pipefail

cd /workspace/projects/LlamaFactory

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export FORCE_TORCHRUN=1
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export WANDB_RUN_GROUP="g2_g3_triple_constant_lr_warmup_batch_sweep"

run() {
  local config="$1"
  echo "[$(date -Is)] Starting ${config}"
  llamafactory-cli train "${config}"
  echo "[$(date -Is)] Finished ${config}"
}

run examples/train_sfl/curriculum_qwen3_8b_sfl_plan_b_g2_plus_g3_triple_constant_lr_5e5.yaml
run examples/train_sfl/curriculum_qwen3_8b_sfl_plan_b_g2_plus_g3_triple_constant_lr_5e5_bs64.yaml
