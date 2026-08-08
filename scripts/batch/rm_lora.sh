cd /workspace/projects/LlamaFactory

# ============================================================
# HelpSteer3: English + MultiLingual ×1
# ============================================================

# --- BT English ---
# pkill -9 -f launcher.py; pkill -9 -f torchrun
# WANDB_RUN_GROUP="HELPSTEER3_BT_ENGLISH_LORA_R16" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True   llamafactory-cli train examples/train_lora/helpsteer3_bt_rm_english.yaml

# # --- BT MultiLingual ---
# pkill -9 -f launcher.py; pkill -9 -f torchrun
# WANDB_RUN_GROUP="HELPSTEER3_BT_MULTILINGUAL_LORA_R16" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True   llamafactory-cli train examples/train_lora/helpsteer3_bt_rm_multilingual.yaml

# --- SfL binary English ---
pkill -9 -f launcher.py; pkill -9 -f torchrun
WANDB_RUN_GROUP="HELPSTEER3_SFL_BINARY_ENGLISH_LORA_R16" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True   llamafactory-cli train examples/train_sfl/helpsteer3_sfl_binary_english.yaml

# --- SfL binary MultiLingual ---
pkill -9 -f launcher.py; pkill -9 -f torchrun
WANDB_RUN_GROUP="HELPSTEER3_SFL_BINARY_MULTILINGUAL_LORA_R16" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True   llamafactory-cli train examples/train_sfl/helpsteer3_sfl_binary_multilingual.yaml

# ============================================================
# HelpSteer3 14B: English + MultiLingual ×1
# ============================================================

# --- BT English 14B ---
pkill -9 -f launcher.py; pkill -9 -f torchrun
WANDB_RUN_GROUP="HELPSTEER3_BT_ENGLISH_LORA_R16_14B" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True   llamafactory-cli train examples/train_lora/helpsteer3_bt_rm_english_14b.yaml

# --- BT MultiLingual 14B ---
pkill -9 -f launcher.py; pkill -9 -f torchrun
WANDB_RUN_GROUP="HELPSTEER3_BT_MULTILINGUAL_LORA_R16_14B" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True   llamafactory-cli train examples/train_lora/helpsteer3_bt_rm_multilingual_14b.yaml

# --- SfL binary English 14B ---
pkill -9 -f launcher.py; pkill -9 -f torchrun
WANDB_RUN_GROUP="HELPSTEER3_SFL_BINARY_ENGLISH_LORA_R16_14B" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True   llamafactory-cli train examples/train_sfl/helpsteer3_sfl_binary_english_14b.yaml

# --- SfL binary MultiLingual 14B ---
pkill -9 -f launcher.py; pkill -9 -f torchrun
WANDB_RUN_GROUP="HELPSTEER3_SFL_BINARY_MULTILINGUAL_LORA_R16_14B" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True   llamafactory-cli train examples/train_sfl/helpsteer3_sfl_binary_multilingual_14b.yaml

# ============================================================
# HelpSteer3: English bs4x (7B + 14B)
# ============================================================

# --- BT English 7B bs4x ---
pkill -9 -f launcher.py; pkill -9 -f torchrun
WANDB_RUN_GROUP="HELPSTEER3_BT_ENGLISH_LORA_R16_BS4X" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True   llamafactory-cli train examples/train_lora/helpsteer3_bt_rm_english_bs4x.yaml

# --- SfL binary English 7B bs4x ---
pkill -9 -f launcher.py; pkill -9 -f torchrun
WANDB_RUN_GROUP="HELPSTEER3_SFL_BINARY_ENGLISH_LORA_R16_BS4X" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True   llamafactory-cli train examples/train_sfl/helpsteer3_sfl_binary_english_bs4x.yaml

# --- BT English 14B bs4x ---
pkill -9 -f launcher.py; pkill -9 -f torchrun
WANDB_RUN_GROUP="HELPSTEER3_BT_ENGLISH_LORA_R16_14B_BS4X" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True   llamafactory-cli train examples/train_lora/helpsteer3_bt_rm_english_14b_bs4x.yaml

# --- SfL binary English 14B bs4x ---
pkill -9 -f launcher.py; pkill -9 -f torchrun
WANDB_RUN_GROUP="HELPSTEER3_SFL_BINARY_ENGLISH_LORA_R16_14B_BS4X" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True   llamafactory-cli train examples/train_sfl/helpsteer3_sfl_binary_english_14b_bs4x.yaml

# ============================================================
# Qwen3-8B: HelpSteer3 English + MultiLingual ×1
# ============================================================

# --- BT English Qwen3-8B ---
pkill -9 -f launcher.py; pkill -9 -f torchrun
WANDB_RUN_GROUP="HELPSTEER3_QWEN3_8B_BT_ENGLISH_LORA_R16" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True   llamafactory-cli train examples/train_lora/helpsteer3_bt_rm_english_qwen3_8b.yaml

# --- BT MultiLingual Qwen3-8B ---
pkill -9 -f launcher.py; pkill -9 -f torchrun
WANDB_RUN_GROUP="HELPSTEER3_QWEN3_8B_BT_MULTILINGUAL_LORA_R16" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True   llamafactory-cli train examples/train_lora/helpsteer3_bt_rm_multilingual_qwen3_8b.yaml

# --- SfL binary English Qwen3-8B ---
pkill -9 -f launcher.py; pkill -9 -f torchrun
WANDB_RUN_GROUP="HELPSTEER3_QWEN3_8B_SFL_BINARY_ENGLISH_LORA_R16" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True   llamafactory-cli train examples/train_sfl/helpsteer3_sfl_binary_english_qwen3_8b.yaml

# --- SfL binary MultiLingual Qwen3-8B ---
pkill -9 -f launcher.py; pkill -9 -f torchrun
WANDB_RUN_GROUP="HELPSTEER3_QWEN3_8B_SFL_BINARY_MULTILINGUAL_LORA_R16" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True   llamafactory-cli train examples/train_sfl/helpsteer3_sfl_binary_multilingual_qwen3_8b.yaml

# ============================================================
# Qwen3-14B: HelpSteer3 English + MultiLingual ×1
# ============================================================

# --- BT English Qwen3-14B ---
pkill -9 -f launcher.py; pkill -9 -f torchrun
WANDB_RUN_GROUP="HELPSTEER3_QWEN3_14B_BT_ENGLISH_LORA_R16" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True   llamafactory-cli train examples/train_lora/helpsteer3_bt_rm_english_qwen3_14b.yaml

# --- BT MultiLingual Qwen3-14B ---
pkill -9 -f launcher.py; pkill -9 -f torchrun
WANDB_RUN_GROUP="HELPSTEER3_QWEN3_14B_BT_MULTILINGUAL_LORA_R16" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True   llamafactory-cli train examples/train_lora/helpsteer3_bt_rm_multilingual_qwen3_14b.yaml

# --- SfL binary English Qwen3-14B ---
pkill -9 -f launcher.py; pkill -9 -f torchrun
WANDB_RUN_GROUP="HELPSTEER3_QWEN3_14B_SFL_BINARY_ENGLISH_LORA_R16" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True   llamafactory-cli train examples/train_sfl/helpsteer3_sfl_binary_english_qwen3_14b.yaml

# --- SfL binary MultiLingual Qwen3-14B ---
pkill -9 -f launcher.py; pkill -9 -f torchrun
WANDB_RUN_GROUP="HELPSTEER3_QWEN3_14B_SFL_BINARY_MULTILINGUAL_LORA_R16" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True   llamafactory-cli train examples/train_sfl/helpsteer3_sfl_binary_multilingual_qwen3_14b.yaml

# ============================================================
# Qwen3: HelpSteer3 English bs4x (8B + 14B)
# ============================================================

# --- BT English Qwen3-8B bs4x ---
pkill -9 -f launcher.py; pkill -9 -f torchrun
WANDB_RUN_GROUP="HELPSTEER3_QWEN3_8B_BT_ENGLISH_LORA_R16_BS4X" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True   llamafactory-cli train examples/train_lora/helpsteer3_bt_rm_english_qwen3_8b_bs4x.yaml

# --- SfL binary English Qwen3-8B bs4x ---
pkill -9 -f launcher.py; pkill -9 -f torchrun
WANDB_RUN_GROUP="HELPSTEER3_QWEN3_8B_SFL_BINARY_ENGLISH_LORA_R16_BS4X" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True   llamafactory-cli train examples/train_sfl/helpsteer3_sfl_binary_english_qwen3_8b_bs4x.yaml

# --- BT English Qwen3-14B bs4x ---
pkill -9 -f launcher.py; pkill -9 -f torchrun
WANDB_RUN_GROUP="HELPSTEER3_QWEN3_14B_BT_ENGLISH_LORA_R16_BS4X" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True   llamafactory-cli train examples/train_lora/helpsteer3_bt_rm_english_qwen3_14b_bs4x.yaml

# --- SfL binary English Qwen3-14B bs4x ---
pkill -9 -f launcher.py; pkill -9 -f torchrun
WANDB_RUN_GROUP="HELPSTEER3_QWEN3_14B_SFL_BINARY_ENGLISH_LORA_R16_BS4X" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True   llamafactory-cli train examples/train_sfl/helpsteer3_sfl_binary_english_qwen3_14b_bs4x.yaml
