#!/usr/bin/env bash
# Run Qwen3-14B SfL LoRA on GPUs 0-3, without DeepSpeed ZeRO.
# Launch in the background from the container with:
#   nohup bash scripts/batch/rm_qwen3_14b_sfl_lora.sh > logs/rm_qwen3_14b_sfl_lora.log 2>&1 &
set -euo pipefail

cd "$(dirname "$0")/../.."

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export FORCE_TORCHRUN=1
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

# --- SfL binary English ---
export WANDB_RUN_GROUP="HELPSTEER3_QWEN3_14B_SFL_BINARY_ENGLISH_LORA_R16"
llamafactory-cli train examples/train_sfl/helpsteer3_sfl_binary_english_qwen3_14b.yaml

# --- SfL binary MultiLingual ---
export WANDB_RUN_GROUP="HELPSTEER3_QWEN3_14B_SFL_BINARY_MULTILINGUAL_LORA_R16"
llamafactory-cli train examples/train_sfl/helpsteer3_sfl_binary_multilingual_qwen3_14b.yaml
