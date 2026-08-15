#!/usr/bin/env bash
# Run the G2 1.5x + G3_2 ablation after the ratio sweep succeeds twice.
set -euo pipefail

parent_pid="${1:?usage: $0 <ratio-sweep-pid> <ratio-sweep-log>}"
parent_log="${2:?usage: $0 <ratio-sweep-pid> <ratio-sweep-log>}"

while kill -0 "${parent_pid}" 2>/dev/null; do
  sleep 60
done

if [ "$(grep -Fc 'Training completed.' "${parent_log}" || true)" -ne 2 ]; then
  echo "Ratio sweep did not complete both runs successfully; not starting G2 1.5x ablation." >&2
  exit 1
fi

cd /workspace/projects/LlamaFactory
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export FORCE_TORCHRUN=1
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export WANDB_RUN_GROUP="PLAN_B_G2_ONE_AND_HALF_PLUS_G3_2"

exec llamafactory-cli train examples/train_sfl/curriculum_qwen3_8b_sfl_plan_b_g2_one_and_half_plus_g3_2.yaml
