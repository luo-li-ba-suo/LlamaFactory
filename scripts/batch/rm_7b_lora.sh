cd /workspace/projects/LlamaFactory && pip install -e . -q
pkill -9 -f launcher.py; pkill -9 -f torchrun
WANDB_RUN_GROUP="Qwen_2_5_7B_BT" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True   llamafactory-cli train examples/train_lora/helpsteer2_bt_rm.yaml
pkill -9 -f launcher.py; pkill -9 -f torchrun
WANDB_RUN_GROUP="Qwen_2_5_7B_SfL" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True   llamafactory-cli train examples/train_sfl/helpsteer2_sfl.yaml
pkill -9 -f launcher.py; pkill -9 -f torchrun
WANDB_RUN_GROUP="Qwen_2_5_7B_SfL_1_5" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True   llamafactory-cli train examples/train_sfl/helpsteer2_sfl_1_5.yaml

