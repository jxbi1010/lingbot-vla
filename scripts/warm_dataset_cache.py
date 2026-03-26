#!/usr/bin/env python3
"""
Pre-warm the HuggingFace/LeRobot dataset cache before distributed training.

When running multi-GPU training with torchrun, all ranks build the dataset
simultaneously, which can cause:
- Race conditions and temp file explosion
- "Not enough disk space" (often /tmp or cache dir)

Run this script once before distributed training to populate the cache in a
single process. Then all ranks will load from cache quickly.

Usage:
  # From config (same config as training):
  python scripts/warm_dataset_cache.py configs/vla/ego_finetune.yaml

  # With overrides:
  python scripts/warm_dataset_cache.py configs/vla/ego_finetune.yaml \\
    --data.train_path /path/to/dataset \\
    --data.norm_stats_file assets/norm_stats/your_norm.json

  # Control how many samples to iterate (default: 100, use 0 to skip iteration):
  python scripts/warm_dataset_cache.py configs/vla/ego_finetune.yaml --num_samples 50

Environment (set before training; warm script uses same paths):
  export HF_DATASETS_CACHE=/path/to/large/disk/hf_datasets_cache
  export HF_HOME=/path/to/large/disk/hf_home
  export XDG_CACHE_HOME=/path/to/large/disk/cache
"""

import argparse
import logging
import os
import sys
from pathlib import Path

import yaml

# Add project root for imports
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from lingbotvla.data.vla_data.base_dataset import resolve_vla_subset_fields
from lingbotvla.utils.arguments import normalize_lerobot_roots

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def load_config(config_path: str) -> dict:
    """Load YAML config, resolving _base_ if present."""
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")

    with open(config_path, encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    # Resolve _base_ inheritance
    if "_base_" in data:
        base_ref = data.pop("_base_")
        config_dir = config_path.parent
        if not Path(base_ref).is_absolute():
            base_ref = config_dir / base_ref
        with open(base_ref, encoding="utf-8") as f:
            base_data = yaml.safe_load(f) or {}
        if "_base_" in base_data:
            base_data.pop("_base_")
        for key in base_data:
            if key not in data:
                data[key] = base_data[key]
            elif isinstance(data[key], dict) and isinstance(base_data[key], dict):
                data[key] = {**base_data[key], **data[key]}

    return data


def make_data_config(config: dict, overrides: dict):
    """Build a minimal data config object for dataset construction."""

    class DataConfig:
        pass

    data = config.get("data", {})
    data.update(overrides)

    cfg = DataConfig()
    raw_tp = data.get("train_path")
    if raw_tp is None:
        cfg.train_path = None
    elif isinstance(raw_tp, list):
        cfg.train_path = normalize_lerobot_roots([str(x) for x in raw_tp])
    else:
        cfg.train_path = normalize_lerobot_roots([str(raw_tp)])
    cfg.norm_stats_file = data.get("norm_stats_file", "assets/norm_stats/database_lerobot_00_norm.json")
    cfg.img_size = data.get("img_size", 224)
    cfg.norm_type = data.get("norm_type", "bounds_99_woclip")
    cfg.chunk_subset = data.get("chunk_subset")
    cfg.episode_subset = data.get("episode_subset")
    cfg.train_chunk_subset = data.get("train_chunk_subset")
    cfg.val_chunk_subset = data.get("val_chunk_subset")
    cfg.train_episode_subset = data.get("train_episode_subset")
    cfg.val_episode_subset = data.get("val_episode_subset")

    eff_ep, eff_chunk = resolve_vla_subset_fields(cfg, for_validation=False)
    cfg.episode_subset = eff_ep
    cfg.chunk_subset = eff_chunk

    return cfg


def main():
    parser = argparse.ArgumentParser(
        description="Pre-warm dataset cache for distributed training",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "config",
        help="Path to YAML config (e.g. configs/vla/ego_finetune.yaml)",
    )
    parser.add_argument(
        "--data.train_path",
        dest="train_path",
        default=None,
        help="Override train_path from config",
    )
    parser.add_argument(
        "--data.norm_stats_file",
        dest="norm_stats_file",
        default=None,
        help="Override norm_stats_file from config",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=100,
        help="Number of samples to iterate for cache warm (0 = skip iteration, only build dataset)",
    )
    args = parser.parse_args()

    # Load config
    config = load_config(args.config)
    logger.info(f"Loaded config from {args.config}")

    overrides = {}
    if args.train_path:
        overrides["train_path"] = args.train_path
    if args.norm_stats_file:
        overrides["norm_stats_file"] = args.norm_stats_file

    data_config = make_data_config(config, overrides)
    if not data_config.train_path:
        raise ValueError("train_path is required (from config or --data.train_path)")

    train_roots = data_config.train_path

    data_name = config.get("data", {}).get("data_name", "robotwin_example")
    model_config = config.get("model", {})
    model_path = model_config.get("model_path") or model_config.get("config_path")
    tokenizer_path = model_config.get("tokenizer_path") or model_path

    if not tokenizer_path:
        raise ValueError("model.tokenizer_path (or model_path) required in config")

    train_cfg = config.get("train", {})

    # Create minimal model config for dataset (prepare_state, prepare_action need max_state_dim, max_action_dim)
    class MinimalModelConfig:
        max_state_dim = train_cfg.get("max_state_dim", 75)
        max_action_dim = train_cfg.get("max_action_dim", 75)

    model_config_obj = MinimalModelConfig()

    # Try loading full model config if path exists (for tokenizer_max_length etc.); fallback to minimal
    if model_path and Path(model_path).exists():
        try:
            from transformers import AutoConfig

            logger.info("Loading model config...")
            model_config_obj = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
            if not hasattr(model_config_obj, "max_state_dim"):
                model_config_obj.max_state_dim = train_cfg.get("max_state_dim", 75)
            if not hasattr(model_config_obj, "max_action_dim"):
                model_config_obj.max_action_dim = train_cfg.get("max_action_dim", 75)
        except Exception as e:
            logger.warning("Could not load model config, using minimal config: %s", e)

    # Load processor (tokenizer + image processor)
    logger.info("Loading processor...")
    from transformers import AutoProcessor

    processor = AutoProcessor.from_pretrained(tokenizer_path, padding_side="right", trust_remote_code=True)
    image_processor = (
        processor.image_processor
        if hasattr(processor, "image_processor") and "qwen" in (tokenizer_path or "").lower()
        else None
    )

    # Build dataset (this populates the HF/LeRobot cache)
    logger.info("Building dataset (this populates the cache)...")
    from lingbotvla.data.vla_data import RobotwinDataset, AlohaAgilexDataset, liberoDataset
    from torch.utils.data import ConcatDataset

    if "libero" in data_name.lower():
        if len(train_roots) != 1:
            raise ValueError("libero expects exactly one train_path root")
        train_dataset = liberoDataset(
            repo_id=train_roots[0],
            config=model_config_obj,
            tokenizer=processor.tokenizer,
            data_config=data_config,
            image_processor=image_processor,
            use_depth_align=False,
        )
    elif "robotwin" in data_name.lower():
        if len(train_roots) == 1:
            train_dataset = RobotwinDataset(
                repo_id=train_roots[0],
                config=model_config_obj,
                tokenizer=processor.tokenizer,
                data_config=data_config,
                image_processor=image_processor,
                use_depth_align=False,
            )
        else:
            datasets = [
                RobotwinDataset(
                    repo_id=path,
                    config=model_config_obj,
                    tokenizer=processor.tokenizer,
                    data_config=data_config,
                    image_processor=image_processor,
                    use_depth_align=False,
                )
                for path in train_roots
            ]
            train_dataset = ConcatDataset(datasets)
            logger.info(f"Loaded {len(train_roots)} task datasets: {train_roots}")
    elif "aloha_agilex" in data_name.lower():
        if len(train_roots) == 1:
            train_dataset = AlohaAgilexDataset(
                repo_id=train_roots[0],
                config=model_config_obj,
                tokenizer=processor.tokenizer,
                data_config=data_config,
                image_processor=image_processor,
                use_depth_align=False,
            )
        else:
            datasets = [
                AlohaAgilexDataset(
                    repo_id=path,
                    config=model_config_obj,
                    tokenizer=processor.tokenizer,
                    data_config=data_config,
                    image_processor=image_processor,
                    use_depth_align=False,
                )
                for path in train_roots
            ]
            train_dataset = ConcatDataset(datasets)
            logger.info(f"Loaded {len(train_roots)} task datasets: {train_roots}")
    else:
        raise ValueError(f"Unsupported data_name: {data_name}")

    n = len(train_dataset)
    logger.info(f"Dataset built: {n} samples")

    # Optionally iterate to force any lazy loading
    if args.num_samples > 0:
        num_iter = min(args.num_samples, n)
        logger.info(f"Iterating over {num_iter} samples to warm cache...")
        import random

        indices = list(range(n))
        random.shuffle(indices)
        for i in range(num_iter):
            idx = indices[i]
            try:
                _ = train_dataset[idx]
            except Exception as e:
                logger.warning(f"Sample {idx} failed: {e}")

    logger.info("Done. Cache is ready.")
    logger.info(
        "Cache location: %s",
        os.environ.get("HF_DATASETS_CACHE", os.path.expanduser("~/.cache/huggingface/datasets")),
    )


if __name__ == "__main__":
    main()
