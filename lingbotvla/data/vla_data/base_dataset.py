# Copyright 2026 Robbyant Team and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import os
from pathlib import Path
from typing import Callable, Dict, List, Literal, Optional, Union

import numpy as np
import torch
from datasets import load_dataset
from datasets.distributed import split_dataset_by_node
from torch.utils.data import Dataset, IterableDataset
from torchvision.transforms.v2 import Resize
from transformers import AutoTokenizer, AutoImageProcessor

from lerobot.datasets.dataset_metadata import LeRobotDatasetMetadata
from lerobot.datasets.lerobot_dataset import LeRobotDataset
import json
import yaml
from PIL import Image
from .transform import Normalizer, extract_semantic_motion, load_norm_stats_from_file, prepare_action, prepare_images, prepare_language, prepare_state

from ...utils import logging


def _default_pi0_config_cls():
    """Lazy import: `lerobot.policies` __init__ pulls heavy deps (e.g. Groot → transformers)."""
    from lerobot.policies.pi0.configuration_pi0 import PI0Config

    return PI0Config


def _lerobot_parquet_paths(data_dir: Path) -> List[Path]:
    """Paths HuggingFace loads: same as lerobot ``load_nested_dataset`` (``data/chunk-*/*.parquet``)."""
    paths = sorted(data_dir.glob("*/*.parquet"))
    if paths:
        return paths
    return sorted(data_dir.rglob("*.parquet"))


def _parquet_load_failure_hint(dataset_root: Path, max_report: int = 12) -> str:
    """Diagnose ArrowInvalid: tiny files, then pyarrow.read_metadata on each file LeRobot actually loads."""
    data_dir = dataset_root / "data"
    if not data_dir.is_dir():
        return f"No directory {data_dir} (expected LeRobot layout with data/chunk-*/*.parquet)."

    paths = _lerobot_parquet_paths(data_dir)
    if not paths:
        return (
            f"No parquet under {data_dir} (tried data/*/*.parquet then rglob). "
            "LeRobot v3 expects e.g. data/chunk-000/file-000.parquet."
        )

    bad_small: List[str] = []
    for p in paths:
        try:
            sz = p.stat().st_size
        except OSError:
            continue
        if sz < 64:
            bad_small.append(f"{p} ({sz} B)")
        if len(bad_small) >= max_report:
            break
    if bad_small:
        return (
            "Empty/tiny parquet files: "
            + "; ".join(bad_small)
            + (f" ... (max {max_report})" if len(bad_small) >= max_report else "")
        )

    try:
        import pyarrow.parquet as pq
    except ImportError:
        return (
            f"Checked {len(paths)} parquet path(s) under {data_dir}; none < 64 B. "
            "Install pyarrow to enable per-file metadata validation in this hint."
        )

    corrupt: List[str] = []
    for p in paths:
        try:
            pq.read_metadata(str(p))
        except Exception as ex:
            corrupt.append(f"{p}\n    -> {type(ex).__name__}: {ex}")
            if len(corrupt) >= max_report:
                break
    if corrupt:
        return (
            f"pyarrow cannot read Parquet metadata ({len(corrupt)} file(s), same order as HF loads):\n"
            + "\n".join(corrupt)
        )

    return (
        f"All {len(paths)} parquet file(s) under {data_dir} passed pyarrow.read_metadata. "
        "Failure may be from HuggingFace `datasets` mmap/stream handling or version skew. "
        "Try: pip install -U 'datasets>=3' 'pyarrow>=15' (or match lerobot pins), "
        "unset/clear HF_DATASETS_CACHE, then retry."
    )


def _open_lerobot_dataset(ctor, *, dataset_path: Path, **lerobot_kwargs):
    """Construct LeRobotDataset with actionable errors on corrupt parquet (ArrowInvalid)."""
    try:
        return ctor(**lerobot_kwargs)
    except Exception as e:
        err = str(e).lower()
        if (
            "arrow" in err
            or "parquet" in err
            or "schema" in err
            or "length 0" in err
            or "record batch" in err
        ):
            hint = _parquet_load_failure_hint(dataset_path)
            raise RuntimeError(
                f"Failed to load LeRobot parquet for root={dataset_path.resolve()}. "
                f"Underlying error: {type(e).__name__}: {e}\n{hint}"
            ) from e
        raise


def _dataset_root_from_config(data_config, repo_id: Union[str, Path]) -> Path:
    """Resolve on-disk LeRobot root. If ``train_path`` is a list (multi-dataset YAML) or a comma-separated string, use ``repo_id`` for this dataset instance."""
    raw = getattr(data_config, "train_path", None)
    if isinstance(raw, list):
        return Path(repo_id)
    if raw is not None:
        s = str(raw).strip()
        if s and "," in s:
            return Path(repo_id)
        if s:
            return Path(s)
    return Path(repo_id)


def _lerobot_tolerance_s(verify_timestamps_sync: bool) -> float:
    """LeRobot v3 has no verify_timestamps_sync; use tolerance_s for delta-timestamp checks."""
    return 1e-4 if verify_timestamps_sync else float("inf")


def _task_label_from_index(meta: LeRobotDatasetMetadata, task_index: Union[int, torch.Tensor]) -> str:
    """Resolve task string from task_index (v3 stores tasks as a pandas DataFrame indexed by name)."""
    idx = int(task_index.item() if isinstance(task_index, torch.Tensor) else task_index)
    tasks = meta.tasks
    if hasattr(tasks, "iloc"):
        return str(tasks.iloc[idx].name)
    return str(tasks[idx])


def _get_episodes_subset(data_config, dataset_meta) -> Optional[List[int]]:
    """Return episode indices to load. episode_subset takes precedence over chunk_subset.
    episode_subset can be:
    - List of ints: [0, 1, 2, 3] for explicit indices
    - List [start, end] with two ints: inclusive episode range (same as range(start, end + 1))
    chunk_subset can be:
    - int: single chunk (e.g. 0 for chunk-000)
    - List [start, end]: chunk index range inclusive (e.g. [0, 99] for chunk-000 to chunk-099)
    - List [n]: single chunk (e.g. [0] for chunk-000)
    """
    episode_subset = getattr(data_config, "episode_subset", None)
    if episode_subset is not None:
        if not isinstance(episode_subset, (list, tuple)) or len(episode_subset) == 0:
            raise TypeError(
                "episode_subset must be a non-empty list: explicit indices [0, 1, 2] or inclusive range [start, end] "
                f"(got {type(episode_subset).__name__}: {episode_subset!r})"
            )
        if len(episode_subset) == 2 and all(isinstance(x, int) for x in episode_subset):
            # [0, 100] -> range 0 to 100 inclusive
            start, end = episode_subset[0], episode_subset[1]
            return list(range(start, end + 1))
        return list(episode_subset)
    chunk_subset = getattr(data_config, "chunk_subset", None)
    if chunk_subset is None:
        return None
    chunks_size = dataset_meta.chunks_size
    total_episodes = dataset_meta.total_episodes
    # Normalize: int or list [0] / [0, 99]
    if isinstance(chunk_subset, (list, tuple)):
        if len(chunk_subset) == 2:
            # [0, 99] -> chunks 0 through 99 inclusive (chunk-000 to chunk-099)
            start_chunk, end_chunk = int(chunk_subset[0]), int(chunk_subset[1])
        elif len(chunk_subset) == 1:
            # [0] -> single chunk
            start_chunk = end_chunk = int(chunk_subset[0])
        else:
            return None
    elif isinstance(chunk_subset, int):
        start_chunk = end_chunk = chunk_subset
    else:
        raise TypeError(
            "chunk_subset must be int (single chunk index), [n] for one chunk, or [start, end] inclusive "
            f"(got {type(chunk_subset).__name__}: {chunk_subset!r})"
        )
    start = start_chunk * chunks_size
    end = min((end_chunk + 1) * chunks_size, total_episodes)
    return list(range(start, end))


def resolve_vla_subset_fields(data_config, *, for_validation: bool):
    """Return ``(episode_subset, chunk_subset)`` for train vs val.

    Split-specific ``train_episode_subset`` / ``val_episode_subset`` and
    ``train_chunk_subset`` / ``val_chunk_subset`` override the shared
    ``episode_subset`` / ``chunk_subset`` when set.
    """
    if for_validation:
        ep = getattr(data_config, "val_episode_subset", None)
        if ep is None:
            ep = getattr(data_config, "episode_subset", None)
        ch = getattr(data_config, "val_chunk_subset", None)
        if ch is None:
            ch = getattr(data_config, "chunk_subset", None)
    else:
        ep = getattr(data_config, "train_episode_subset", None)
        if ep is None:
            ep = getattr(data_config, "episode_subset", None)
        ch = getattr(data_config, "train_chunk_subset", None)
        if ch is None:
            ch = getattr(data_config, "chunk_subset", None)
    return ep, ch


class VlaDataset(Dataset):
    def __init__(
        self,
        repo_id="path2dataset",
        config=None,
        tokenizer=AutoTokenizer,
        data_config=None,
        image_processor=None,
        use_depth_align=False,
        action_name="action",
        verify_timestamps_sync: bool = True,
    ):
        if config is None:
            config = _default_pi0_config_cls()
        self.image_processor = image_processor
        # [i / 30 for i in range(50)] represents action chunks in 50 steps at 30 FPS.
        # The timestamps are set to 0 for the images and state, as we only use current obs.
        self.config = config
        self.tokenizer = tokenizer
        self.dataset_meta = LeRobotDatasetMetadata(repo_id)
        delta_timestamps = {
            action_name: [t / self.dataset_meta.fps for t in range(50)],
        }
        self.dataset = LeRobotDataset(
            repo_id=repo_id,
            delta_timestamps=delta_timestamps,
            tolerance_s=_lerobot_tolerance_s(verify_timestamps_sync),
        )
        self.action_name = action_name

    def __len__(self):
        return len(self.dataset)

    def getdata(self, idx):
        item = self.dataset[idx]
        task = _task_label_from_index(self.dataset_meta, item["task_index"])
        assert task == item["task"]
        return item

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        if idx < 0 or idx >= len(self):
            raise IndexError(f"Index {idx} out of bounds.")
        max_retries = 200
        attempts = 0
        cur = idx
        last_err = None
        while attempts < max_retries:
            try:
                return self.getdata(cur)
            except Exception as e:
                last_err = e
                attempts += 1
                cur = np.random.randint(0, len(self))
                if cur >= len(self):
                    cur = 0
                continue

        raise RuntimeError(
            f"Failed to fetch a valid item starting from idx={idx} after {attempts} attempts. "
            f"Last error: {repr(last_err)}"
        )

class liberoDataset(Dataset):
    def __init__(
        self,
        repo_id="libero",
        config=None,
        tokenizer=AutoTokenizer,
        data_config=None,
        image_processor=None,
        use_depth_align=False,
        verify_timestamps_sync: bool = True,
    ):
        if config is None:
            config = _default_pi0_config_cls()
        image_transforms = Resize((data_config.img_size, data_config.img_size))
        self.image_processor = image_processor
        # [i / 30 for i in range(50)] represents action chunks in 50 steps at 30 FPS.
        # The timestamps are set to 0 for the images and state, as we only use current obs.
        self.config = config
        self.tokenizer = tokenizer
        self.norm_stats_file = data_config.norm_stats_file
        self.dataset_meta = LeRobotDatasetMetadata(repo_id)
        delta_timestamps = {
            "actions": [t / self.dataset_meta.fps for t in range(50)],
        }
        episodes = _get_episodes_subset(data_config, self.dataset_meta)
        self.dataset = LeRobotDataset(
            repo_id=repo_id,
            episodes=episodes,
            image_transforms=image_transforms,
            delta_timestamps=delta_timestamps,
            tolerance_s=_lerobot_tolerance_s(verify_timestamps_sync),
        )
        norm_stats = load_norm_stats_from_file(self.norm_stats_file)
        self.normalizer = Normalizer(
            norm_stats=norm_stats,
            from_file=True,
            data_type='libero',
            norm_type={
                "image": "identity",
                "wrist_image": "identity",
                "state": data_config.norm_type,
                "actions": data_config.norm_type,
            },
        )
        self.use_depth_align = use_depth_align

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        task = _task_label_from_index(self.dataset_meta, item["task_index"])
        assert task == item["task"]

        normalized_item = self.normalizer.normalize(item)
        base_image = (normalized_item["image"] * 255).to(torch.uint8)
        wrist_image = (normalized_item["wrist_image"] * 255).to(
            torch.uint8
        )
        batch_dict =  {
            "image": {"base_0_rgb": base_image, "left_wrist_0_rgb": wrist_image},
            "state": normalized_item["state"].to(torch.float32),
            "action": normalized_item["actions"].to(torch.float32),
            "action_is_pad": normalized_item["actions_is_pad"],
            "prompt": extract_semantic_motion(item["task"]),
        }
        state = prepare_state(self.config, batch_dict) # bs,8 -> bs,32
        lang_tokens, lang_masks = prepare_language(self.config, self.tokenizer, batch_dict) # bs, seq_len
        actions = prepare_action(self.config, batch_dict) # bs,50,7 -> bs,50,32 , 7
        images, img_masks, pil_images = prepare_images(self.config, self.image_processor, batch_dict,  use_depth_align=self.use_depth_align)

        batch_dict = {
            'images': images,
            'img_masks': img_masks,
            'state': state,
            'lang_tokens': lang_tokens,
            'lang_masks': lang_masks,
            'actions': actions,
            'action_is_pad': batch_dict['action_is_pad'],
        }

        if self.use_depth_align: batch_dict['pil_images'] = pil_images

        return batch_dict

class RobotwinDataset(Dataset):
    def __init__(
        self,
        repo_id="robotwin",
        config=None,
        tokenizer=AutoTokenizer,
        data_config=None,
        image_processor=None,
        use_depth_align=False,
        verify_timestamps_sync: bool = True,
    ):
        if config is None:
            config = _default_pi0_config_cls()
        image_transforms = Resize((data_config.img_size, data_config.img_size))
        self.image_processor = image_processor
        # [i / 30 for i in range(50)] represents action chunks in 50 steps at 30 FPS.
        # The timestamps are set to 0 for the images and state, as we only use current obs.
        self.config = config
        self.tokenizer = tokenizer
        self.norm_stats_file = data_config.norm_stats_file
        dataset_path = _dataset_root_from_config(data_config, repo_id)
        # Use dataset folder name as repo_id (not "local") to avoid 401 when fallback download is attempted
        local_repo_id = dataset_path.name
        self.dataset_meta = LeRobotDatasetMetadata(repo_id=local_repo_id, root=str(dataset_path))
        delta_timestamps = {
            "action": [t / self.dataset_meta.fps for t in range(50)],
        }
        episodes = _get_episodes_subset(data_config, self.dataset_meta)
        # root must be the full path to the dataset folder (containing meta/, data/)
        self.dataset = _open_lerobot_dataset(
            LeRobotDataset,
            dataset_path=dataset_path,
            repo_id=local_repo_id,
            root=str(dataset_path),
            episodes=episodes,
            image_transforms=image_transforms,
            delta_timestamps=delta_timestamps,
            tolerance_s=_lerobot_tolerance_s(verify_timestamps_sync),
        )

        norm_stats = load_norm_stats_from_file(self.norm_stats_file)
        self.normalizer = Normalizer(
            norm_stats=norm_stats,
            from_file=True,
            data_type='auto',
            norm_type={
                "observation.images.cam_high": "identity",
                "observation.images.cam_left_wrist": "identity",
                "observation.images.cam_right_wrist": "identity",
                "observation.state": data_config.norm_type,
                "action": data_config.norm_type,
            },
        )
        self.use_depth_align = use_depth_align

    def __len__(self):
        return len(self.dataset)

    def getdata(self, idx):
        # 1. Fetch raw data
        item = self.dataset[idx]
        
        # 2. Critical: Clone tensors to break the link to the memory-mapped Arrow page
        item = {k: v.clone() if torch.is_tensor(v) else v for k, v in item.items()}
        
        task = _task_label_from_index(self.dataset_meta, item["task_index"])
        assert task == item["task"]
        
        normalized_item = self.normalizer.normalize(item)
        
        # Process images
        base_image = (normalized_item["observation.images.cam_high"] * 255).to(torch.uint8)
        left_wrist_image = (
            (normalized_item["observation.images.cam_left_wrist"] * 255).to(torch.uint8)
            if "observation.images.cam_left_wrist" in normalized_item
            else torch.zeros_like(base_image)
        )
        right_wrist_image = (
            (normalized_item["observation.images.cam_right_wrist"] * 255).to(torch.uint8)
            if "observation.images.cam_right_wrist" in normalized_item
            else torch.zeros_like(base_image)
        )

        task_label = _task_label_from_index(self.dataset_meta, item["task_index"])
        prompt = [extract_semantic_motion(task_label)]

        # Intermediate dict for your processing functions
        prep_dict = {
            "image": {"base_0_rgb": base_image, "left_wrist_0_rgb": left_wrist_image, "right_wrist_0_rgb": right_wrist_image},
            "state": normalized_item["observation.state"].to(torch.float32),
            "action": normalized_item["action"].to(torch.float32),
            "action_is_pad": normalized_item["action_is_pad"],
            "prompt": prompt,
        }
        
        # Generate final training components
        state = prepare_state(self.config, prep_dict)
        lang_tokens, lang_masks = prepare_language(self.config, self.tokenizer, prep_dict)
        actions = prepare_action(self.config, prep_dict)
        images, img_masks, pil_images = prepare_images(self.config, self.image_processor, prep_dict, use_depth_align=self.use_depth_align)

        final_output = {
            'images': images,
            'img_masks': img_masks,
            'state': state,
            'lang_tokens': lang_tokens,
            'lang_masks': lang_masks,
            'actions': actions,
            'action_is_pad': prep_dict['action_is_pad'],
        }
        
        if self.use_depth_align: 
            final_output['pil_images'] = pil_images

        # 3. Explicitly cleanup all temporary references
        del item, normalized_item, prep_dict
        
        return final_output

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        if idx < 0 or idx >= len(self):
            raise IndexError(f"Index {idx} out of bounds.")
        max_retries = 200
        attempts = 0
        cur = idx
        last_err = None
        while attempts < max_retries:
            try:
                return self.getdata(cur)
            except Exception as e:
                last_err = e
                attempts += 1
                cur = np.random.randint(0, len(self))
                if cur >= len(self):
                    cur = 0
                continue

        raise RuntimeError(
            f"Failed to fetch a valid item starting from idx={idx} after {attempts} attempts. "
            f"Last error: {repr(last_err)}"
        )



class AlohaAgilexDataset(Dataset):
    def __init__(
        self,
        repo_id="aloha_agilex",
        config=None,
        tokenizer=AutoTokenizer,
        data_config=None,
        image_processor=None,
        use_depth_align=False,
        verify_timestamps_sync: bool = True,
    ):
        if config is None:
            config = _default_pi0_config_cls()
        image_transforms = Resize((data_config.img_size, data_config.img_size))
        self.image_processor = image_processor
        # [i / 30 for i in range(50)] represents action chunks in 50 steps at 30 FPS.
        # The timestamps are set to 0 for the images and state, as we only use current obs.
        self.config = config
        self.tokenizer = tokenizer
        self.norm_stats_file = data_config.norm_stats_file
        dataset_path = _dataset_root_from_config(data_config, repo_id)
        # Use dataset folder name as repo_id (not "local") to avoid 401 when fallback download is attempted
        local_repo_id = dataset_path.name
        self.dataset_meta = LeRobotDatasetMetadata(repo_id=local_repo_id, root=str(dataset_path))
        delta_timestamps = {
            "action": [t / self.dataset_meta.fps for t in range(50)],
        }
        episodes = _get_episodes_subset(data_config, self.dataset_meta)
        # root must be the full path to the dataset folder (containing meta/, data/)
        self.dataset = _open_lerobot_dataset(
            LeRobotDataset,
            dataset_path=dataset_path,
            repo_id=local_repo_id,
            root=str(dataset_path),
            episodes=episodes,
            image_transforms=image_transforms,
            delta_timestamps=delta_timestamps,
            tolerance_s=_lerobot_tolerance_s(verify_timestamps_sync),
        )

        # required_columns = [
        # "observation.state", 
        # "action", 
        # "episode_index", 
        # "frame_index", 
        # "task_index", 
        # "timestamp", 
        # "index"
        # ]

        # if "action_is_pad" in self.dataset.hf_dataset.column_names:
        #     required_columns.append("action_is_pad")
        # self.dataset.hf_dataset = self.dataset.hf_dataset.select_columns(required_columns)

        norm_stats = load_norm_stats_from_file(self.norm_stats_file)
        self.normalizer = Normalizer(
            norm_stats=norm_stats,
            from_file=True,
            data_type='auto',
            norm_type={
                "observation.images.cam_front": "identity",
                "observation.images.cam_left": "identity",
                "observation.images.cam_right": "identity",
                "observation.state": data_config.norm_type,
                "action": data_config.norm_type,
            },
        )
        self.use_depth_align = use_depth_align

    def __len__(self):
        return len(self.dataset)

    def getdata(self, idx):
        item = self.dataset[idx]
        task = _task_label_from_index(self.dataset_meta, item["task_index"])
        assert task == item["task"]
        
        normalized_item = self.normalizer.normalize(item)
        # Fallback: use black image when wrist cameras are missing (e.g. video decode failed, num_workers>0)
        base_image = (normalized_item["observation.images.cam_front"] * 255).to(torch.uint8)
        left_wrist_image = (
            (normalized_item["observation.images.cam_left"] * 255).to(torch.uint8)
            if "observation.images.cam_left" in normalized_item
            else torch.zeros_like(base_image)
        )
        right_wrist_image = (
            (normalized_item["observation.images.cam_right"] * 255).to(torch.uint8)
            if "observation.images.cam_right" in normalized_item
            else torch.zeros_like(base_image)
        )

        task_label = _task_label_from_index(self.dataset_meta, item["task_index"])
        prompt = [extract_semantic_motion(task_label)]

        batch_dict =  {
            "image": {"base_0_rgb": base_image, "left_wrist_0_rgb": left_wrist_image, "right_wrist_0_rgb": right_wrist_image},
            "state": normalized_item["observation.state"].to(torch.float32),
            "action": normalized_item["action"].to(torch.float32),
            "action_is_pad": normalized_item["action_is_pad"],
            "prompt": prompt,
        }
        state = prepare_state(self.config, batch_dict) # bs,8 -> bs,32
        lang_tokens, lang_masks = prepare_language(self.config, self.tokenizer, batch_dict) # bs, seq_len
        actions = prepare_action(self.config, batch_dict) # bs,50,7 -> bs,50,32 , 7
        images, img_masks, pil_images = prepare_images(self.config, self.image_processor, batch_dict, use_depth_align=self.use_depth_align)

        batch_dict = {
            'images': images,
            'img_masks': img_masks,
            'state': state,
            'lang_tokens': lang_tokens,
            'lang_masks': lang_masks,
            'actions': actions,
            'action_is_pad': batch_dict['action_is_pad'],
        }
        if self.use_depth_align: batch_dict['pil_images'] = pil_images

        return batch_dict

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        if idx < 0 or idx >= len(self):
            raise IndexError(f"Index {idx} out of bounds.")
        max_retries = 200
        attempts = 0
        cur = idx
        last_err = None
        while attempts < max_retries:
            try:
                return self.getdata(cur)
            except Exception as e:
                last_err = e
                attempts += 1
                cur = np.random.randint(0, len(self))
                if cur >= len(self):
                    cur = 0
                continue

        raise RuntimeError(
            f"Failed to fetch a valid item starting from idx={idx} after {attempts} attempts. "
            f"Last error: {repr(last_err)}"
        )


class CustomizedRobotwinDataset(Dataset):
    def __init__(
        self,
        repo_id="robotwin",
        config=None,
        tokenizer=AutoTokenizer,
        data_config=None,
        image_processor=None,
        use_depth_align=False,
        verify_timestamps_sync: bool = True,
    ):
        if config is None:
            config = _default_pi0_config_cls()
        image_transforms = Resize((data_config.img_size, data_config.img_size))
        self.image_processor = image_processor
        # [i / 30 for i in range(50)] represents action chunks in 50 steps at 30 FPS.
        # The timestamps are set to 0 for the images and state, as we only use current obs.
        self.config = config
        self.tokenizer = tokenizer
        self.norm_stats_file = data_config.norm_stats_file
        self.dataset_meta = LeRobotDatasetMetadata(repo_id)
        delta_timestamps = {
            "action": [t / self.dataset_meta.fps for t in range(50)],
        }
        episodes = _get_episodes_subset(data_config, self.dataset_meta)
        self.dataset = LeRobotDataset(
            repo_id=repo_id,
            episodes=episodes,
            image_transforms=image_transforms,
            delta_timestamps=delta_timestamps,
            tolerance_s=_lerobot_tolerance_s(verify_timestamps_sync),
        )
        norm_stats = load_norm_stats_from_file(self.norm_stats_file)
        self.normalizer = Normalizer(
            norm_stats=norm_stats,
            from_file=True,
            data_type='auto',
            norm_type={
                "observation.images.cam_high": "identity",
                "observation.images.cam_left_wrist": "identity",
                "observation.images.cam_right_wrist": "identity",
                "observation.state": data_config.norm_type,
                "action": data_config.norm_type,
            },
        )
        self.use_depth_align = use_depth_align

    def __len__(self):
        return len(self.dataset)

    def getdata(self, idx):
        item = self.dataset[idx]
        task = _task_label_from_index(self.dataset_meta, item["task_index"])
        assert task == item["task"]

        normalized_item = self.normalizer.normalize(item)
        # Fallback: use black image when wrist cameras are missing (e.g. video decode failed, num_workers>0)
        base_image = (normalized_item["observation.images.cam_high"] * 255).to(torch.uint8)
        left_wrist_image = (
            (normalized_item["observation.images.cam_left_wrist"] * 255).to(torch.uint8)
            if "observation.images.cam_left_wrist" in normalized_item
            else torch.zeros_like(base_image)
        )
        right_wrist_image = (
            (normalized_item["observation.images.cam_right_wrist"] * 255).to(torch.uint8)
            if "observation.images.cam_right_wrist" in normalized_item
            else torch.zeros_like(base_image)
        )
        batch_dict =  {
            "image": {"base_0_rgb": base_image, "left_wrist_0_rgb": left_wrist_image, "right_wrist_0_rgb": right_wrist_image},
            "state": normalized_item["observation.state"].to(torch.float32),
            "action": normalized_item["action"].to(torch.float32),
            "action_is_pad": normalized_item["action_is_pad"],
            "prompt": [item["task"]],
        }
        state = prepare_state(self.config, batch_dict) # bs,8 -> bs,32
        lang_tokens, lang_masks = prepare_language(self.config, self.tokenizer, batch_dict) # bs, seq_len
        actions = prepare_action(self.config, batch_dict) # bs,50,7 -> bs,50,32 , 7
        images, img_masks, pil_images = prepare_images(self.config, self.image_processor, batch_dict, use_depth_align=self.use_depth_align)

        batch_dict = {
            'images': images,
            'img_masks': img_masks,
            'state': state,
            'lang_tokens': lang_tokens,
            'lang_masks': lang_masks,
            'actions': actions,
            'action_is_pad': batch_dict['action_is_pad'],
        }
        if self.use_depth_align: batch_dict['pil_images'] = pil_images

        return batch_dict

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        if idx < 0 or idx >= len(self):
            raise IndexError(f"Index {idx} out of bounds.")
        max_retries = 200
        attempts = 0
        cur = idx
        last_err = None
        while attempts < max_retries:
            try:
                return self.getdata(cur)
            except Exception as e:
                last_err = e
                attempts += 1
                cur = np.random.randint(0, len(self))
                if cur >= len(self):
                    cur = 0
                continue

        raise RuntimeError(
            f"Failed to fetch a valid item starting from idx={idx} after {attempts} attempts. "
            f"Last error: {repr(last_err)}"
        )