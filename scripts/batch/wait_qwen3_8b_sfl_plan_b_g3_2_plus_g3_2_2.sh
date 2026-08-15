#!/usr/bin/env bash
# Start after all eight GPUs are idle; used to queue behind the seed-43 reproduction.
set -euo pipefail

cd /workspace/projects/LlamaFactory
while nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk '$1 > 100 { busy=1 } END { exit(busy ? 0 : 1) }'; do
  sleep 30
done

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export FORCE_TORCHRUN=1
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export WANDB_RUN_GROUP="g3_2_plus_g3_2_2"
echo "[$(date -Is)] Starting G3_2 + G3_2_2"
llamafactory-cli train examples/train_sfl/curriculum_qwen3_8b_sfl_plan_b_g3_2_plus_g3_2_2.yaml
