#!/usr/bin/env bash
# Follow-up queue for GPUs 0-3, without DeepSpeed ZeRO or process killing.
# Run after rm_qwen3_14b_sfl_lora.sh has completed.
set -euo pipefail

cd "$(dirname "$0")/../.."

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export FORCE_TORCHRUN=1
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

# --- Qwen2.5-14B: SfL binary English ---
export WANDB_RUN_GROUP="HELPSTEER3_QWEN2_5_14B_SFL_BINARY_ENGLISH_LORA_R16"
llamafactory-cli train examples/train_sfl/helpsteer3_sfl_binary_english_14b.yaml

# --- Qwen2.5-14B: SfL binary MultiLingual ---
export WANDB_RUN_GROUP="HELPSTEER3_QWEN2_5_14B_SFL_BINARY_MULTILINGUAL_LORA_R16"
llamafactory-cli train examples/train_sfl/helpsteer3_sfl_binary_multilingual_14b.yaml

# --- Qwen3-14B: BT English ---
export WANDB_RUN_GROUP="HELPSTEER3_QWEN3_14B_BT_ENGLISH_LORA_R16"
llamafactory-cli train examples/train_lora/helpsteer3_bt_rm_english_qwen3_14b.yaml

# --- Qwen3-14B: BT MultiLingual ---
export WANDB_RUN_GROUP="HELPSTEER3_QWEN3_14B_BT_MULTILINGUAL_LORA_R16"
llamafactory-cli train examples/train_lora/helpsteer3_bt_rm_multilingual_qwen3_14b.yaml
