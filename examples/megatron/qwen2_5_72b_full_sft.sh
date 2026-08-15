#!/bin/bash
#
# Qwen2.5-72B 双节点全量 SFT 训练 (Megatron-Core)
# TP=8（节点内张量并行）, PP=2（节点间流水线并行）
#
# 用法:
#   节点 0:  bash examples/megatron/qwen2_5_72b_full_sft.sh 0 <master_ip>
#   节点 1:  bash examples/megatron/qwen2_5_72b_full_sft.sh 1 <master_ip>
#

set -e

# ============================================================
# 分布式配置
# ============================================================
NODE_RANK=${1:?Usage: $0 <node_rank> <master_addr> [master_port]}
MASTER_ADDR=${2:?Usage: $0 <node_rank> <master_addr> [master_port]}
MASTER_PORT=${3:-29500}

export NNODES=2
export NODE_RANK
export MASTER_ADDR
export MASTER_PORT

# ============================================================
# 训练配置
# ============================================================
CONFIG_FILE="examples/megatron/qwen2_5_72b_full_sft.yaml"

echo "============================================================"
echo "Qwen2.5-72B Megatron-Core 全量 SFT (TP=8, PP=2)"
echo "============================================================"
echo "  节点 Rank:     $NODE_RANK"
echo "  Master 地址:  $MASTER_ADDR:$MASTER_PORT"
echo "  总节点数:      $NNODES (16 GPUs)"
echo "  TP=8 (节点内), PP=2 (节点间), DP=1"
echo "  配置文件:      $CONFIG_FILE"
echo "============================================================"

# ============================================================
# 环境变量
# ============================================================
export USE_MCA=1
export FORCE_TORCHRUN=1
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

# NCCL 跨节点通信
export NCCL_TIMEOUT=1800
export NCCL_IB_TIMEOUT=23

# InfiniBand（如果有的话）
# export NCCL_IB_DISABLE=0
# export NCCL_NET_GDR_LEVEL=5

llamafactory-cli train "$CONFIG_FILE"
