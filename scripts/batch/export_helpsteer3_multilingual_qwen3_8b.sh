#!/usr/bin/env bash
# Export separate Qwen3-8B multilingual SfL and BT LoRA checkpoints.
set -euo pipefail

cd "$(dirname "$0")/../.."

BASE_MODEL="/workspace/model/Qwen3-8B"
SFL_ADAPTER="saves/helpsteer3/sfl-binary-multilingual-qwen3-8b"
BT_ADAPTER="/workspace/model/ckpt/reward_model/helpsteer3-bt-multilingual-qwen3-8b"
CKPT_ROOT="/workspace/model/ckpt/reward_model"

export_adapter() {
  local name="$1"
  local adapter="$2"
  local export_dir="$3"

  if [[ ! -f "$adapter/adapter_config.json" ]]; then
    echo "Missing adapter config: $adapter/adapter_config.json" >&2
    return 1
  fi
  if ! rg -q 'Qwen3-8B' "$adapter/adapter_config.json"; then
    echo "Adapter is not based on Qwen3-8B: $adapter" >&2
    return 1
  fi
  if [[ -e "$export_dir" ]]; then
    echo "Refusing to overwrite existing export: $export_dir" >&2
    return 1
  fi

  echo "=== Exporting $name ==="
  llamafactory-cli export \
    --model_name_or_path "$BASE_MODEL" \
    --adapter_name_or_path "$adapter" \
    --template qwen3 \
    --finetuning_type lora \
    --export_dir "$export_dir" \
    --export_size 2 \
    --export_device cpu \
    --trust_remote_code

  if [[ -f "$adapter/value_head.safetensors" ]]; then
    cp "$adapter/value_head.safetensors" "$export_dir/"
  fi

  test -f "$export_dir/config.json"
  test -f "$export_dir/tokenizer_config.json"
  echo "Completed: $export_dir"
}

export_adapter \
  "HelpSteer3 multilingual Qwen3 SfL binary" \
  "$SFL_ADAPTER" \
  "$CKPT_ROOT/helpsteer3-sfl-binary-multilingual-qwen3-8b"

export_adapter \
  "HelpSteer3 multilingual Qwen3 BT" \
  "$BT_ADAPTER" \
  "$CKPT_ROOT/helpsteer3-bt-multilingual-qwen3-8b"
