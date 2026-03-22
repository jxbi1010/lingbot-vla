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
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Callable, Dict, List, Literal, Optional
import numpy as np
import torch
from datasets import load_dataset
from datasets.distributed import split_dataset_by_node
from torch.utils.data import Dataset, IterableDataset
from lerobot.common.policies.pi0.configuration_pi0 import PI0Config
from torchvision.transforms.v2 import Resize
from transformers import AutoTokenizer, AutoImageProcessor
import lerobot.common.datasets.lerobot_dataset as _lerobot_ds_mod
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
import json
import yaml
from PIL import Image
from .transform import Normalizer, extract_semantic_motion, load_norm_stats_from_file, prepare_action, prepare_images, prepare_language, prepare_state

from ...utils import logging

_LEROBOT_TS_PATCH_LOCK = threading.Lock()


@contextmanager
def _optional_skip_lerobot_timestamps_sync(verify_timestamps_sync: bool):
    """When False, temporarily no-op lerobot's check_timestamps_sync during LeRobotDataset construction."""
    if verify_timestamps_sync:
        yield
        return
    with _LEROBOT_TS_PATCH_LOCK:
        orig = _lerobot_ds_mod.check_timestamps_sync
        _lerobot_ds_mod.check_timestamps_sync = lambda *a, **k: None
        try:
            yield
        finally:
            _lerobot_ds_mod.check_timestamps_sync = orig


def _get_episodes_subset(data_config, dataset_meta) -> Optional[List[int]]:
    """Return episode indices to load. episode_subset takes precedence over chunk_subset.
    episode_subset can be:
    - List of ints: [0, 1, 2, 3] for explicit indices
    - String "0-100" for range 0 to 100 inclusive
    - List [0, 100] for range 0 to 100 inclusive (2-element list = range)
    chunk_subset can be:
    - int: single chunk (e.g. 0 for chunk-000)
    - List [start, end]: chunk range inclusive (e.g. [0, 99] for chunk-000 to chunk-099)
    """
    episode_subset = getattr(data_config, "episode_subset", None)
    if episode_subset is None:
        pass
    elif isinstance(episode_subset, str):
        # "0-100" -> range 0 to 100 inclusive
        parts = episode_subset.strip().split("-")
        if len(parts) == 2:
            start, end = int(parts[0].strip()), int(parts[1].strip())
            return list(range(start, end + 1))
    elif isinstance(episode_subset, (list, tuple)) and len(episode_subset) > 0:
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
    # Normalize: allow int (legacy) or list [0] / [0, 99]
    if isinstance(chunk_subset, (list, tuple)):
        if len(chunk_subset) == 2:
            # [0, 99] -> chunks 0 through 99 inclusive (chunk-000 to chunk-099)
            start_chunk, end_chunk = int(chunk_subset[0]), int(chunk_subset[1])
        elif len(chunk_subset) == 1:
            # [0] -> single chunk
            start_chunk = end_chunk = int(chunk_subset[0])
        else:
            return None
    else:
        # Single int (legacy)
        start_chunk = end_chunk = int(chunk_subset)
    start = start_chunk * chunks_size
    end = min((end_chunk + 1) * chunks_size, total_episodes)
    return list(range(start, end))


class VlaDataset(Dataset):
    def __init__(
        self,
        repo_id="path2dataset",
        config=PI0Config,
        tokenizer=AutoTokenizer,
        data_config=None,
        image_processor=None,
        use_depth_align=False,
        action_name="action",
        verify_timestamps_sync: bool = True,
    ):
        self.image_processor = image_processor
        # [i / 30 for i in range(50)] represents action chunks in 50 steps at 30 FPS.
        # The timestamps are set to 0 for the images and state, as we only use current obs.
        self.config = config
        self.tokenizer = tokenizer
        self.dataset_meta = LeRobotDatasetMetadata(repo_id)
        delta_timestamps = {
            action_name: [t / self.dataset_meta.fps for t in range(50)],
        }
        with _optional_skip_lerobot_timestamps_sync(verify_timestamps_sync):
            self.dataset = LeRobotDataset(
                repo_id=repo_id,
                delta_timestamps=delta_timestamps,
            )
        self.action_name = action_name

    def __len__(self):
        return len(self.dataset)

    def getdata(self, idx):
        item = self.dataset[idx]
        task = self.dataset_meta.tasks[int(item['task_index'])]
        assert task == item['task']
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
        config=PI0Config,
        tokenizer=AutoTokenizer,
        data_config=None,
        image_processor=None,
        use_depth_align=False,
        verify_timestamps_sync: bool = True,
    ):
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
        with _optional_skip_lerobot_timestamps_sync(verify_timestamps_sync):
            self.dataset = LeRobotDataset(
                repo_id=repo_id,
                episodes=episodes,
                image_transforms=image_transforms,
                delta_timestamps=delta_timestamps,
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
        task = self.dataset_meta.tasks[int(item['task_index'])]
        assert task == item['task']

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
        config=PI0Config,
        tokenizer=AutoTokenizer,
        data_config=None,
        image_processor=None,
        use_depth_align=False,
        verify_timestamps_sync: bool = True,
    ):
        image_transforms = Resize((data_config.img_size, data_config.img_size))
        self.image_processor = image_processor
        # [i / 30 for i in range(50)] represents action chunks in 50 steps at 30 FPS.
        # The timestamps are set to 0 for the images and state, as we only use current obs.
        self.config = config
        self.tokenizer = tokenizer
        self.norm_stats_file = data_config.norm_stats_file
        train_path = getattr(data_config, "train_path", None) or repo_id
        dataset_path = Path(train_path)
        # Use dataset folder name as repo_id (not "local") to avoid 401 when fallback download is attempted
        local_repo_id = dataset_path.name
        self.dataset_meta = LeRobotDatasetMetadata(repo_id=local_repo_id, root=str(dataset_path))
        delta_timestamps = {
            "action": [t / self.dataset_meta.fps for t in range(50)],
        }
        episodes = _get_episodes_subset(data_config, self.dataset_meta)
        # root must be the full path to the dataset folder (containing meta/, data/)
        with _optional_skip_lerobot_timestamps_sync(verify_timestamps_sync):
            self.dataset = LeRobotDataset(
                repo_id=local_repo_id,
                root=str(dataset_path),
                episodes=episodes,
                image_transforms=image_transforms,
                delta_timestamps=delta_timestamps,
            )

        required_columns = [
        "observation.state", 
        "action", 
        "episode_index", 
        "frame_index", 
        "task_index", 
        "timestamp", 
        "index"
        ]

        if "action_is_pad" in self.dataset.hf_dataset.column_names:
            required_columns.append("action_is_pad")
        self.dataset.hf_dataset = self.dataset.hf_dataset.select_columns(required_columns)

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
        task = self.dataset_meta.tasks[int(item['task_index'])]
        assert task == item['task']
        
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

        task_index = int(item['task_index'])    
        task_label = self.dataset_meta.tasks[task_index]
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



class AlohaAgilexDataset(Dataset):
    def __init__(
        self,
        repo_id="aloha_agilex",
        config=PI0Config,
        tokenizer=AutoTokenizer,
        data_config=None,
        image_processor=None,
        use_depth_align=False,
        verify_timestamps_sync: bool = True,
    ):
        image_transforms = Resize((data_config.img_size, data_config.img_size))
        self.image_processor = image_processor
        # [i / 30 for i in range(50)] represents action chunks in 50 steps at 30 FPS.
        # The timestamps are set to 0 for the images and state, as we only use current obs.
        self.config = config
        self.tokenizer = tokenizer
        self.norm_stats_file = data_config.norm_stats_file
        train_path = getattr(data_config, "train_path", None) or repo_id
        dataset_path = Path(train_path)
        # Use dataset folder name as repo_id (not "local") to avoid 401 when fallback download is attempted
        local_repo_id = dataset_path.name
        self.dataset_meta = LeRobotDatasetMetadata(repo_id=local_repo_id, root=str(dataset_path))
        delta_timestamps = {
            "action": [t / self.dataset_meta.fps for t in range(50)],
        }
        episodes = _get_episodes_subset(data_config, self.dataset_meta)
        # root must be the full path to the dataset folder (containing meta/, data/)
        with _optional_skip_lerobot_timestamps_sync(verify_timestamps_sync):
            self.dataset = LeRobotDataset(
                repo_id=local_repo_id,
                root=str(dataset_path),
                episodes=episodes,
                image_transforms=image_transforms,
                delta_timestamps=delta_timestamps,
            )

        required_columns = [
        "observation.state", 
        "action", 
        "episode_index", 
        "frame_index", 
        "task_index", 
        "timestamp", 
        "index"
        ]

        if "action_is_pad" in self.dataset.hf_dataset.column_names:
            required_columns.append("action_is_pad")
        self.dataset.hf_dataset = self.dataset.hf_dataset.select_columns(required_columns)

        norm_stats = load_norm_stats_from_file(self.norm_stats_file)
        self.normalizer = Normalizer(
            norm_stats=norm_stats,
            from_file=True,
            data_type='auto',
            norm_type={
                "observation.images.cam_f": "identity",
                "observation.images.cam_l": "identity",
                "observation.images.cam_r": "identity",
                "observation.state": data_config.norm_type,
                "action": data_config.norm_type,
            },
        )
        self.use_depth_align = use_depth_align

    def __len__(self):
        return len(self.dataset)

    def getdata(self, idx):
        item = self.dataset[idx]
        task = self.dataset_meta.tasks[int(item['task_index'])]
        assert task == item['task']
        
        normalized_item = self.normalizer.normalize(item)
        # Fallback: use black image when wrist cameras are missing (e.g. video decode failed, num_workers>0)
        base_image = (normalized_item["observation.images.cam_f"] * 255).to(torch.uint8)
        left_wrist_image = (
            (normalized_item["observation.images.cam_l"] * 255).to(torch.uint8)
            if "observation.images.cam_l" in normalized_item
            else torch.zeros_like(base_image)
        )
        right_wrist_image = (
            (normalized_item["observation.images.cam_r"] * 255).to(torch.uint8)
            if "observation.images.cam_r" in normalized_item
            else torch.zeros_like(base_image)
        )

        task_index = int(item['task_index'])    
        task_label = self.dataset_meta.tasks[task_index]
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
        config=PI0Config,
        tokenizer=AutoTokenizer,
        data_config=None,
        image_processor=None,
        use_depth_align=False,
        verify_timestamps_sync: bool = True,
    ):
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
        with _optional_skip_lerobot_timestamps_sync(verify_timestamps_sync):
            self.dataset = LeRobotDataset(
                repo_id=repo_id,
                episodes=episodes,
                image_transforms=image_transforms,
                delta_timestamps=delta_timestamps,
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
        task = self.dataset_meta.tasks[int(item['task_index'])]
        assert task == item['task']

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