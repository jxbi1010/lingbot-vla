#!/bin/bash

set -x

BASE_CACHE="${BASE_CACHE:-/checkpoints/jianxin/cache}"
mkdir -p "${BASE_CACHE}/datasets" "${BASE_CACHE}/torch_inductor" "${BASE_CACHE}/triton" "${BASE_CACHE}/wandb"

export HF_HOME="${BASE_CACHE}"
export HF_DATASETS_CACHE="${BASE_CACHE}/datasets"

export USER=jianxin
export LOGNAME=jianxin

export WANDB_API_KEY=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
export WANDB_DIR="${BASE_CACHE}/wandb"

export TORCH_INDUCTOR_CACHE_DIR="${BASE_CACHE}/torch_inductor"
export TRITON_CACHE_DIR="${BASE_CACHE}/triton"

export TOKENIZERS_PARALLELISM=false

if [ -z "$CUDA_VISIBLE_DEVICES" ]; then
  NPROC_PER_NODE=$(nvidia-smi -L | wc -l)
else
  # 可见 GPU 数量
  NPROC_PER_NODE=$(echo $CUDA_VISIBLE_DEVICES | tr ',' '\n' | wc -l)
fi
echo "Using NPROC_PER_NODE=$NPROC_PER_NODE GPUs"

# --- Distributed Setup ---
# Arena/Kubeflow PyTorchJob injects: WORLD_SIZE, RANK, MASTER_ADDR, MASTER_PORT
if [ -n "${WORLD_SIZE:-}" ]; then
  NNODES=${WORLD_SIZE}
  NODE_RANK=${RANK}
else
  # Fallback for local or PET-style envs
  NNODES=${PET_NNODES:-${NNODES:-1}}
  NODE_RANK=${PET_NODE_RANK:-${NODE_RANK:-0}}
fi

# Ensure MASTER_ADDR is not localhost if running multi-node
if [ "$NNODES" -gt 1 ] && [ "$MASTER_ADDR" = "0.0.0.0" ]; then
    echo "Error: MASTER_ADDR is 0.0.0.0 but NNODES > 1. Check Arena env."
    # Usually Arena handles this, but safety first.
fi

echo "Distributed Config: NNODES=$NNODES, NODE_RANK=$NODE_RANK, MASTER_ADDR=$MASTER_ADDR"

NPROC_PER_NODE=${NPROC_PER_NODE:=$NPROC_PER_NODE}
MASTER_ADDR=${MASTER_ADDR:=0.0.0.0}
MASTER_PORT=${MASTER_PORT:=62500}


torchrun --nnodes=$NNODES --nproc-per-node $NPROC_PER_NODE --node-rank $NODE_RANK \
  --master-addr=$MASTER_ADDR --master-port=$MASTER_PORT $@ 2>&1 | tee log.txt
