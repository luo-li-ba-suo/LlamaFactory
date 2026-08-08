#!/usr/bin/env bash
# Qwen3-8B multilingual SfL 1-5 LoRA on all eight GPUs.
set -euo pipefail

cd "$(dirname "$0")/../.."

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export FORCE_TORCHRUN=1
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export WANDB_RUN_GROUP="HELPSTEER3_QWEN3_8B_SFL_1_5_MULTILINGUAL_LORA_R16"

llamafactory-cli train examples/train_sfl/helpsteer3_sfl_1_5_multilingual_qwen3_8b.yaml
