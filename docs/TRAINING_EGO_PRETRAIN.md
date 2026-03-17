# Lingbot VLA Ego Pretrain Training Guide

This guide explains how to run ego pretraining with `ego_pretrain_10k.yaml` and how to avoid the "no space on disk" / dataset loading issues.

## Root Cause of "No Space" Error

When running **multi-GPU training** with `torchrun`, **all ranks build the LeRobot dataset at the same time**. Each rank:

1. Loads parquet files from `train_path`
2. Generates the train split
3. Writes to cache (`$XDG_CACHE_HOME/huggingface/lerobot/` or similar)

Concurrent writes from multiple processes can cause:

- Race conditions and temp file explosion
- "No space left on device" (often `/tmp` or cache dir)
- "Another rank downloading dataset again" after splits are generated

## Recommended: Two-Step Procedure

### Step 1: Pre-warm the dataset cache (single process)

Run this **before** distributed training to populate the cache in a single process:

```bash
cd /home/jianxin/jx_ws/lingbot-vla

# Use your config
python scripts/warm_dataset_cache.py configs/vla/ego_pretrain_10k.yaml

# Or with explicit paths
python scripts/warm_dataset_cache.py configs/vla/ego_pretrain_10k.yaml \
  --train_path /home/ss-oss1/data/user/jiankai/Data/lerobot_test_data/lerobot_data_10k \
  --norm_stats_file assets/norm_stats/ego_pretrain_10k_norm.json
```

This will load the dataset, generate splits, and iterate over samples to warm the cache. Wait until it prints "Done. Cache is ready."

### Step 2: Run training

**Option A: Single-GPU (most reliable, avoids multi-rank dataset issues)**

```bash
cd /home/jianxin/jx_ws/lingbot-vla

CUDA_VISIBLE_DEVICES=0 bash train.sh \
  tasks/vla/train_lingbotvla.py \
  configs/vla/ego_pretrain_10k.yaml
```

`CUDA_VISIBLE_DEVICES=0` forces `train.sh` to use 1 GPU → 1 process → no concurrent dataset loading.

**Option B: Multi-GPU (after cache is warmed)**

```bash
cd /home/jianxin/jx_ws/lingbot-vla

# Use 4 GPUs
CUDA_VISIBLE_DEVICES=0,1,2,3 bash train.sh \
  tasks/vla/train_lingbotvla.py \
  configs/vla/ego_pretrain_10k.yaml
```

The training script now wraps dataset building in `main_process_first`, so rank 0 populates the cache first and other ranks wait before loading.

## Config Summary for ego_pretrain_10k.yaml

| Key | Value |
|-----|-------|
| `data.train_path` | Path to LeRobot dataset (parquet format) |
| `data.norm_stats_file` | `assets/norm_stats/ego_pretrain_10k_norm.json` |
| `data.data_name` | `robotwin_example` (uses RobotwinDataset) |
| `data.num_workers` | 0 (recommended to avoid DataLoader multiprocessing conflicts) |
| `train.output_dir` | Output directory for checkpoints |

## Ensure norm stats exist

If `assets/norm_stats/ego_pretrain_10k_norm.json` is missing, compute it first:

```bash
CUDA_VISIBLE_DEVICES=0 bash train.sh \
  scripts/compute_norm_ego.py \
  configs/norm/ego_pretrain.yaml \
  --data.train_path /path/to/lerobot_data_10k \
  --data.norm_path assets/norm_stats/ego_pretrain_10k_norm.json
```

## Cache location

LeRobot caches under `$XDG_CACHE_HOME/huggingface/lerobot/` (default: `~/.cache/huggingface/lerobot/`). If disk space is limited there:

```bash
export XDG_CACHE_HOME=/path/to/large/disk/cache
```

Then run warm_cache and training with this env set.
