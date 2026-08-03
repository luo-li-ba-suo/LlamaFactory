#!/usr/bin/env bash
# Sequential 14B RM experiments on GPUs 0-3, without DeepSpeed ZeRO.
# Runs one four-GPU job at a time and stops at the first failed job.
set -euo pipefail

cd "$(dirname "$0")/../.."

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export FORCE_TORCHRUN=1
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

run() {
  local run_group="$1"
  local config="$2"
  export WANDB_RUN_GROUP="$run_group"
  echo "[$(date -Is)] Starting ${config}"
  llamafactory-cli train "$config"
}

# Qwen3-14B SfL
run "HELPSTEER3_QWEN3_14B_SFL_BINARY_ENGLISH_LORA_R16" \
  examples/train_sfl/helpsteer3_sfl_binary_english_qwen3_14b.yaml
run "HELPSTEER3_QWEN3_14B_SFL_BINARY_MULTILINGUAL_LORA_R16" \
  examples/train_sfl/helpsteer3_sfl_binary_multilingual_qwen3_14b.yaml

# Qwen2.5-14B SfL
run "HELPSTEER3_QWEN2_5_14B_SFL_BINARY_ENGLISH_LORA_R16" \
  examples/train_sfl/helpsteer3_sfl_binary_english_14b.yaml
run "HELPSTEER3_QWEN2_5_14B_SFL_BINARY_MULTILINGUAL_LORA_R16" \
  examples/train_sfl/helpsteer3_sfl_binary_multilingual_14b.yaml

# Qwen3-14B BT
run "HELPSTEER3_QWEN3_14B_BT_ENGLISH_LORA_R16" \
  examples/train_lora/helpsteer3_bt_rm_english_qwen3_14b.yaml
run "HELPSTEER3_QWEN3_14B_BT_MULTILINGUAL_LORA_R16" \
  examples/train_lora/helpsteer3_bt_rm_multilingual_qwen3_14b.yaml
