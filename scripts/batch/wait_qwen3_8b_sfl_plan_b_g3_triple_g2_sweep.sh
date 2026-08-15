#!/usr/bin/env bash
# Queue the G3 triple continuation and the five G3-triple:G2 ratio variants.
set -euo pipefail

cd /workspace/projects/LlamaFactory
while nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk '$1 > 100 { busy=1 } END { exit(busy ? 0 : 1) }'; do
  sleep 30
done

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export FORCE_TORCHRUN=1
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

run() {
  local config="$1"
  export WANDB_RUN_GROUP="g3_triple_g2_sweep"
  echo "[$(date -Is)] Starting ${config}"
  llamafactory-cli train "${config}"
}

run examples/train_sfl/curriculum_qwen3_8b_sfl_plan_b_g3_2_plus_g3_2_2_plus_g3_2_3.yaml
run examples/train_sfl/curriculum_qwen3_8b_sfl_plan_b_g3_triple_plus_g2_0_25.yaml
run examples/train_sfl/curriculum_qwen3_8b_sfl_plan_b_g3_triple_plus_g2_0_5.yaml
run examples/train_sfl/curriculum_qwen3_8b_sfl_plan_b_g3_triple_plus_g2_0_75.yaml
run examples/train_sfl/curriculum_qwen3_8b_sfl_plan_b_g3_triple_plus_g2_1_0.yaml
