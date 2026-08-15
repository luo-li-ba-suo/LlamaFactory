#!/usr/bin/env bash
# Start the Plan B 1:1.5 ratio ablation once every GPU is idle.
set -euo pipefail

gpu_is_idle() {
  nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null \
    | awk '$1 > 100 {busy = 1} END {exit busy}'
}

while ! gpu_is_idle; do
  echo "[$(date -Is)] GPUs are busy; waiting 60 seconds."
  sleep 60
done

cd /workspace/projects/LlamaFactory
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export FORCE_TORCHRUN=1
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export WANDB_RUN_GROUP="PLAN_B_G2_G3_2_RATIO_1_1_5"

exec llamafactory-cli train examples/train_sfl/curriculum_qwen3_8b_sfl_plan_b_g2_g3_2_ratio_1_1_5.yaml
