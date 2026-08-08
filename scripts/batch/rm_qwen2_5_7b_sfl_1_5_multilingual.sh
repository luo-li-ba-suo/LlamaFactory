#!/usr/bin/env bash
# Qwen2.5-7B multilingual SfL 1-5 LoRA on GPUs 0-3.
set -euo pipefail

cd "$(dirname "$0")/../.."

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export FORCE_TORCHRUN=1
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export WANDB_RUN_GROUP="HELPSTEER3_QWEN2_5_7B_SFL_1_5_MULTILINGUAL_LORA_R16"

llamafactory-cli train examples/train_sfl/helpsteer3_sfl_1_5_multilingual.yaml
