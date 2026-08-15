#!/usr/bin/env bash
# Resume the interrupted G3-triple:G2 sweep after the 0.75 preprocessing failure.
set -euo pipefail

cd /workspace/projects/LlamaFactory
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export FORCE_TORCHRUN=1
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

run() {
  local config="$1"
  export WANDB_RUN_GROUP="g3_triple_g2_sweep"
  echo "[$(date -Is)] Starting ${config}"
  llamafactory-cli train "${config}"
}

run examples/train_sfl/curriculum_qwen3_8b_sfl_plan_b_g3_triple_plus_g2_0_75.yaml
run examples/train_sfl/curriculum_qwen3_8b_sfl_plan_b_g3_triple_plus_g2_1_0.yaml
