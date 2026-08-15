#!/usr/bin/env bash
# Run the two Plan C half-positive experiments only after a prior run succeeds.
set -euo pipefail

prior_pid="${1:?usage: $0 <prior-pid> <prior-log>}"
prior_log="${2:?usage: $0 <prior-pid> <prior-log>}"

while kill -0 "${prior_pid}" 2>/dev/null; do
  sleep 60
done

if ! grep -Fq "Training completed." "${prior_log}"; then
  echo "Prior training did not complete successfully; not starting half-positive runs." >&2
  exit 1
fi

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

run examples/train_sfl/curriculum_qwen3_8b_sfl_plan_c_g2g3_2_plus_g2g3_3_pos_half.yaml
run examples/train_sfl/curriculum_qwen3_8b_sfl_plan_c_g2g3_2_plus_g3g3_3_pos_half.yaml
