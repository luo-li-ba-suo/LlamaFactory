#!/bin/bash
# Export LoRA checkpoint (SfL/BT/RM) to full model
# Usage: bash scripts/export_lora.sh <ckpt_dir> <export_dir>

set -e

BASE_MODEL="/workspace/model/chat_model/Qwen2.5-7B-Instruct"
CKPT_DIR="${1:-saves/helpsteer2/sfl-lora}"
EXPORT_DIR="${2:-$CKPT_DIR/merged}"

echo "=== Export LoRA to Full Model ==="
echo "  Base model:  $BASE_MODEL"
echo "  Checkpoint:  $CKPT_DIR"
echo "  Export to:   $EXPORT_DIR"
echo ""

llamafactory-cli export \
  --model_name_or_path "$BASE_MODEL" \
  --adapter_name_or_path "$CKPT_DIR" \
  --template qwen \
  --finetuning_type lora \
  --export_dir "$EXPORT_DIR" \
  --export_size 2 \
  --export_device cpu \
  --trust_remote_code

# RM checkpoints have a value_head that the export CLI misses (it uses infer_args, stage defaults to sft)
if [ -f "$CKPT_DIR/value_head.safetensors" ]; then
    cp "$CKPT_DIR/value_head.safetensors" "$EXPORT_DIR/"
    echo "Copied value_head.safetensors"
fi

echo ""
echo "Done: $EXPORT_DIR"
du -sh "$EXPORT_DIR"
