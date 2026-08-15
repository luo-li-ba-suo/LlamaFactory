#!/bin/bash
#
# Qwen2.5-72B 双节点全量 SFT 训练脚本
#
# 用法:
#   在节点 0 (master) 上:  bash examples/train_full/qwen2_5_72b_full_sft.sh 0 <master_ip>
#   在节点 1 (worker) 上:  bash examples/train_full/qwen2_5_72b_full_sft.sh 1 <master_ip>
#
# 弹性模式（可选，需要 RDZV_ID）:
#   RDZV_ID=qwen25_72b_sft bash examples/train_full/qwen2_5_72b_full_sft.sh 0 <master_ip>
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

# 每节点 GPU 数量（留空则自动检测，也可手动指定）
# export NPROC_PER_NODE=8

# ============================================================
# 训练配置
# ============================================================
CONFIG_FILE="examples/train_full/qwen2_5_72b_full_sft.yaml"

echo "============================================================"
echo "Qwen2.5-72B 双节点全量 SFT 训练"
echo "============================================================"
echo "  节点 Rank:     $NODE_RANK"
echo "  Master 地址:  $MASTER_ADDR:$MASTER_PORT"
echo "  总节点数:      $NNODES"
echo "  配置文件:      $CONFIG_FILE"
echo "============================================================"

# ============================================================
# 环境优化
# ============================================================
export FORCE_TORCHRUN=1
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export TORCH_NCCL_AVOID_RECORD_STREAMS=1

# NCCL 网络超时（大模型加载慢，需要更长的超时时间）
export NCCL_TIMEOUT=1800
export NCCL_IB_TIMEOUT=23

# 如果使用 InfiniBand，取消下行注释
# export NCCL_IB_DISABLE=0
# export NCCL_NET_GDR_LEVEL=5

# ============================================================
# 启动训练
# ============================================================
if [ -n "$RDZV_ID" ]; then
    # 弹性启动模式（支持容错和动态扩缩节点）
    echo "  模式: 弹性启动 (RDZV_ID=$RDZV_ID)"
    export MIN_NNODES=1
    export MAX_NNODES=2
    export MAX_RESTARTS=3
fi

llamafactory-cli train "$CONFIG_FILE"
