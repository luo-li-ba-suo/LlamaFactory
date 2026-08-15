cd /workspace/projects/LlamaFactory

# ============================================================
# 7B upsample_correctness ×3
# ============================================================

# --- BT RM ×3 ---
# pkill -9 -f launcher.py; pkill -9 -f torchrun
# WANDB_RUN_GROUP="Qwen_2_5_7B_BT" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True   llamafactory-cli train examples/train_lora/helpsteer2_bt_rm.yaml
# pkill -9 -f launcher.py; pkill -9 -f torchrun
# WANDB_RUN_GROUP="Qwen_2_5_7B_BT" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True   llamafactory-cli train examples/train_lora/helpsteer2_bt_rm.yaml
# pkill -9 -f launcher.py; pkill -9 -f torchrun
# WANDB_RUN_GROUP="Qwen_2_5_7B_BT" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True   llamafactory-cli train examples/train_lora/helpsteer2_bt_rm.yaml

# --- BT RM upsample_correctness ×3 ---
# pkill -9 -f launcher.py; pkill -9 -f torchrun
# WANDB_RUN_GROUP="Qwen_2_5_7B_BT_upsample_correctness" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True   llamafactory-cli train examples/train_lora/helpsteer2_bt_rm_upsample_correctness.yaml
# pkill -9 -f launcher.py; pkill -9 -f torchrun
# WANDB_RUN_GROUP="Qwen_2_5_7B_BT_upsample_correctness" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True   llamafactory-cli train examples/train_lora/helpsteer2_bt_rm_upsample_correctness.yaml
# pkill -9 -f launcher.py; pkill -9 -f torchrun
# WANDB_RUN_GROUP="Qwen_2_5_7B_BT_upsample_correctness" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True   llamafactory-cli train examples/train_lora/helpsteer2_bt_rm_upsample_correctness.yaml

# # --- SfL 1-5 T2 upsample_correctness ×3 ---
# pkill -9 -f launcher.py; pkill -9 -f torchrun
# WANDB_RUN_GROUP="Qwen_2_5_7B_SfL_1_5_T2_upsample_correctness" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True   llamafactory-cli train examples/train_sfl/helpsteer2_sfl_1_5_T2_upsample_correctness.yaml
# pkill -9 -f launcher.py; pkill -9 -f torchrun
# WANDB_RUN_GROUP="Qwen_2_5_7B_SfL_1_5_T2_upsample_correctness" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True   llamafactory-cli train examples/train_sfl/helpsteer2_sfl_1_5_T2_upsample_correctness.yaml
# pkill -9 -f launcher.py; pkill -9 -f torchrun
# WANDB_RUN_GROUP="Qwen_2_5_7B_SfL_1_5_T2_upsample_correctness" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True   llamafactory-cli train examples/train_sfl/helpsteer2_sfl_1_5_T2_upsample_correctness.yaml

# # --- SfL binary upsample_correctness ×3 ---
# pkill -9 -f launcher.py; pkill -9 -f torchrun
# WANDB_RUN_GROUP="Qwen_2_5_7B_SfL_upsample_correctness" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True   llamafactory-cli train examples/train_sfl/helpsteer2_sfl_upsample_correctness.yaml
# pkill -9 -f launcher.py; pkill -9 -f torchrun
# WANDB_RUN_GROUP="Qwen_2_5_7B_SfL_upsample_correctness" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True   llamafactory-cli train examples/train_sfl/helpsteer2_sfl_upsample_correctness.yaml
# pkill -9 -f launcher.py; pkill -9 -f torchrun
# WANDB_RUN_GROUP="Qwen_2_5_7B_SfL_upsample_correctness" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True   llamafactory-cli train examples/train_sfl/helpsteer2_sfl_upsample_correctness.yaml


# # ============================================================
# # 14B ×3
# # ============================================================

# # --- 14B round 1 ---
# # pkill -9 -f launcher.py; pkill -9 -f torchrun
# # WANDB_RUN_GROUP="Qwen_2_5_14B_BT" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True   llamafactory-cli train examples/train_lora/helpsteer2_bt_rm_14b.yaml
# pkill -9 -f launcher.py; pkill -9 -f torchrun
# WANDB_RUN_GROUP="Qwen_2_5_14B_SfL" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True   llamafactory-cli train examples/train_sfl/helpsteer2_sfl_14b.yaml
# pkill -9 -f launcher.py; pkill -9 -f torchrun
# WANDB_RUN_GROUP="Qwen_2_5_14B_SfL_1_5" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True   llamafactory-cli train examples/train_sfl/helpsteer2_sfl_1_5_14b.yaml

# # --- 14B round 2 ---
# # pkill -9 -f launcher.py; pkill -9 -f torchrun
# # WANDB_RUN_GROUP="Qwen_2_5_14B_BT" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True   llamafactory-cli train examples/train_lora/helpsteer2_bt_rm_14b.yaml
# pkill -9 -f launcher.py; pkill -9 -f torchrun
# WANDB_RUN_GROUP="Qwen_2_5_14B_SfL" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True   llamafactory-cli train examples/train_sfl/helpsteer2_sfl_14b.yaml
# pkill -9 -f launcher.py; pkill -9 -f torchrun
# WANDB_RUN_GROUP="Qwen_2_5_14B_SfL_1_5" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True   llamafactory-cli train examples/train_sfl/helpsteer2_sfl_1_5_14b.yaml

# # --- 14B round 3 ---
# # pkill -9 -f launcher.py; pkill -9 -f torchrun
# # WANDB_RUN_GROUP="Qwen_2_5_14B_BT" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True   llamafactory-cli train examples/train_lora/helpsteer2_bt_rm_14b.yaml
# pkill -9 -f launcher.py; pkill -9 -f torchrun
# WANDB_RUN_GROUP="Qwen_2_5_14B_SfL" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True   llamafactory-cli train examples/train_sfl/helpsteer2_sfl_14b.yaml
# pkill -9 -f launcher.py; pkill -9 -f torchrun
# WANDB_RUN_GROUP="Qwen_2_5_14B_SfL_1_5" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True   llamafactory-cli train examples/train_sfl/helpsteer2_sfl_1_5_14b.yaml


# ============================================================
# 7B ablation: strict (coh≤ AND comp≤) ×3
# ============================================================

# --- BT RM strict ×3 ---
pkill -9 -f launcher.py; pkill -9 -f torchrun
WANDB_RUN_GROUP="Qwen_2_5_7B_BT_strict" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True   llamafactory-cli train examples/train_lora/helpsteer2_bt_rm_upsample_correctness_strict.yaml
pkill -9 -f launcher.py; pkill -9 -f torchrun
WANDB_RUN_GROUP="Qwen_2_5_7B_BT_strict" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True   llamafactory-cli train examples/train_lora/helpsteer2_bt_rm_upsample_correctness_strict.yaml
pkill -9 -f launcher.py; pkill -9 -f torchrun
WANDB_RUN_GROUP="Qwen_2_5_7B_BT_strict" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True   llamafactory-cli train examples/train_lora/helpsteer2_bt_rm_upsample_correctness_strict.yaml

# --- SfL binary strict ×3 ---
pkill -9 -f launcher.py; pkill -9 -f torchrun
WANDB_RUN_GROUP="Qwen_2_5_7B_SfL_strict" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True   llamafactory-cli train examples/train_sfl/helpsteer2_sfl_upsample_correctness_strict.yaml
pkill -9 -f launcher.py; pkill -9 -f torchrun
WANDB_RUN_GROUP="Qwen_2_5_7B_SfL_strict" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True   llamafactory-cli train examples/train_sfl/helpsteer2_sfl_upsample_correctness_strict.yaml
pkill -9 -f launcher.py; pkill -9 -f torchrun
WANDB_RUN_GROUP="Qwen_2_5_7B_SfL_strict" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True   llamafactory-cli train examples/train_sfl/helpsteer2_sfl_upsample_correctness_strict.yaml

# --- SfL 1-5 T2 strict ×3 ---
pkill -9 -f launcher.py; pkill -9 -f torchrun
WANDB_RUN_GROUP="Qwen_2_5_7B_SfL_1_5_strict" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True   llamafactory-cli train examples/train_sfl/helpsteer2_sfl_1_5_T2_upsample_correctness_strict.yaml
pkill -9 -f launcher.py; pkill -9 -f torchrun
WANDB_RUN_GROUP="Qwen_2_5_7B_SfL_1_5_strict" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True   llamafactory-cli train examples/train_sfl/helpsteer2_sfl_1_5_T2_upsample_correctness_strict.yaml
pkill -9 -f launcher.py; pkill -9 -f torchrun
WANDB_RUN_GROUP="Qwen_2_5_7B_SfL_1_5_strict" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True   llamafactory-cli train examples/train_sfl/helpsteer2_sfl_1_5_T2_upsample_correctness_strict.yaml


# ============================================================
# 7B ablation: anti (coh< OR comp<) ×3
# ============================================================

# --- BT RM anti ×3 ---
pkill -9 -f launcher.py; pkill -9 -f torchrun
WANDB_RUN_GROUP="Qwen_2_5_7B_BT_anti" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True   llamafactory-cli train examples/train_lora/helpsteer2_bt_rm_upsample_correctness_anti.yaml
pkill -9 -f launcher.py; pkill -9 -f torchrun
WANDB_RUN_GROUP="Qwen_2_5_7B_BT_anti" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True   llamafactory-cli train examples/train_lora/helpsteer2_bt_rm_upsample_correctness_anti.yaml
pkill -9 -f launcher.py; pkill -9 -f torchrun
WANDB_RUN_GROUP="Qwen_2_5_7B_BT_anti" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True   llamafactory-cli train examples/train_lora/helpsteer2_bt_rm_upsample_correctness_anti.yaml

# --- SfL binary anti ×3 ---
pkill -9 -f launcher.py; pkill -9 -f torchrun
WANDB_RUN_GROUP="Qwen_2_5_7B_SfL_anti" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True   llamafactory-cli train examples/train_sfl/helpsteer2_sfl_upsample_correctness_anti.yaml
pkill -9 -f launcher.py; pkill -9 -f torchrun
WANDB_RUN_GROUP="Qwen_2_5_7B_SfL_anti" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True   llamafactory-cli train examples/train_sfl/helpsteer2_sfl_upsample_correctness_anti.yaml
pkill -9 -f launcher.py; pkill -9 -f torchrun
WANDB_RUN_GROUP="Qwen_2_5_7B_SfL_anti" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True   llamafactory-cli train examples/train_sfl/helpsteer2_sfl_upsample_correctness_anti.yaml

# --- SfL 1-5 T2 anti ×3 ---
pkill -9 -f launcher.py; pkill -9 -f torchrun
WANDB_RUN_GROUP="Qwen_2_5_7B_SfL_1_5_anti" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True   llamafactory-cli train examples/train_sfl/helpsteer2_sfl_1_5_T2_upsample_correctness_anti.yaml
pkill -9 -f launcher.py; pkill -9 -f torchrun
WANDB_RUN_GROUP="Qwen_2_5_7B_SfL_1_5_anti" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True   llamafactory-cli train examples/train_sfl/helpsteer2_sfl_1_5_T2_upsample_correctness_anti.yaml
pkill -9 -f launcher.py; pkill -9 -f torchrun
WANDB_RUN_GROUP="Qwen_2_5_7B_SfL_1_5_anti" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True   llamafactory-cli train examples/train_sfl/helpsteer2_sfl_1_5_T2_upsample_correctness_anti.yaml
