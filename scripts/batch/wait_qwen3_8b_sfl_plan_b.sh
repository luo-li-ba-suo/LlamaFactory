#!/usr/bin/env bash
# Wait for all GPUs to be idle, then run the Plan B ablations sequentially.
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

run() {
  local config="$1"
  export WANDB_RUN_GROUP="${config%.yaml}"
  echo "[$(date -Is)] Starting ${config}"
  llamafactory-cli train "${config}"
}

run examples/train_sfl/curriculum_qwen3_8b_sfl_plan_b_g2_half_plus_g3_2.yaml
run examples/train_sfl/curriculum_qwen3_8b_sfl_plan_b_g2_plus_g3_2_nonpositive.yaml
