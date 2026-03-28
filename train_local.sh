#!/bin/bash

set -x

export HF_HOME="/home/ss-oss1/checkpoints/jianxin/cache"
export HF_DATASETS_CACHE="/home/ss-oss1/checkpoints/jianxin/cache/hf_datasets"

# Weights & Biases: local run dir (override with WANDB_DIR). API key: export WANDB_API_KEY or `wandb login`.
export WANDB_DIR="${WANDB_DIR:-${HF_HOME}/wandb}"
mkdir -p "${WANDB_DIR}"

export TOKENIZERS_PARALLELISM=false
if [ -z "$CUDA_VISIBLE_DEVICES" ]; then
  NPROC_PER_NODE=$(nvidia-smi -L | wc -l)
else
  # 可见 GPU 数量
  NPROC_PER_NODE=$(echo $CUDA_VISIBLE_DEVICES | tr ',' '\n' | wc -l)
fi
echo "Using NPROC_PER_NODE=$NPROC_PER_NODE GPUs"
NNODES=${NNODES:=1}
NPROC_PER_NODE=${NPROC_PER_NODE:=$NPROC_PER_NODE}
NODE_RANK=${NODE_RANK:=0}
MASTER_ADDR=${MASTER_ADDR:=0.0.0.0}
MASTER_PORT=${MASTER_PORT:=62500}


torchrun --nnodes=$NNODES --nproc-per-node $NPROC_PER_NODE --node-rank $NODE_RANK \
  --master-addr=$MASTER_ADDR --master-port=$MASTER_PORT $@ 2>&1 | tee log.txt
