# Copyright 2025 Bytedance Ltd. and/or its affiliates
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
from typing import Callable, Dict, List, Literal, Optional
import numpy as np
import torch
from datasets import load_dataset
from datasets.distributed import split_dataset_by_node
from torch.utils.data import Dataset, IterableDataset
from lerobot.common.policies.pi0.configuration_pi0 import PI0Config
from torchvision.transforms.v2 import Resize
from transformers import AutoTokenizer, AutoImageProcessor
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
import json
from ..distributed.parallel_state import get_parallel_state
from ..utils import logging
from .vla_data import *
from .vla_data.transform import Normalizer, prepare_action, prepare_images, prepare_language, prepare_state
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from pathlib import Path

logger = logging.get_logger(__name__)

try:
    import datasets.features.features as features

    _OLD_GENERATE_FROM_DICT = features.generate_from_dict

    def _new_generate_from_dict(obj):
        if isinstance(obj, dict) and obj.get("_type") == "List":
            obj["_type"] = "Sequence"
        return _OLD_GENERATE_FROM_DICT(obj)

    features.generate_from_dict = _new_generate_from_dict
except (ImportError, AttributeError):
    # If datasets or the function doesn't exist, do nothing.
    pass

class DummyDataset(Dataset):
    def __init__(self, size: int, seq_length: int):
        """
        Args:
            size (int): Nums of datasets
            seq_length (int, optional): seq_length
        """
        self.size = size
        self.seq_length = seq_length
        self.vocab_size = 32768

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, index: int) -> List[Dict[str, "torch.Tensor"]]:
        input_ids = torch.randint(low=0, high=self.vocab_size, size=(self.seq_length,))
        attention_mask = torch.ones((self.seq_length,), dtype=torch.long)
        labels = input_ids.clone()
        return [{"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}]

class MappingDataset(Dataset):
    """
    Mapping dataset.
    Args:
        data (Dataset): Dataset
        transform (Optional[Callable]): transform function
    """

    def __init__(self, data: "Dataset", transform: Optional[Callable] = None):
        self._data = data
        self._transform = transform

    def __len__(self) -> int:
        return len(self._data)

    def __getitem__(self, index: int) -> List[Dict[str, "torch.Tensor"]]:
        if self._transform is not None:
            return self._transform(self._data[index])
        else:
            return self._data[index]


class IterativeDataset(IterableDataset):
    """
    Iterative dataset.
    Args:
        data (Dataset): Dataset
        transform (Optional[Callable]): transform function
    """

    def __init__(self, data: "Dataset", transform: Optional[Callable] = None):
        self._data = data
        self._transform = transform

    def __iter__(self):
        for sample in self._data:
            if self._transform is not None:
                result = self._transform(sample)
                if result is not None:
                    if isinstance(result, (list, tuple)):
                        for r in result:
                            yield r
                    else:
                        yield result
            else:
                yield sample

    def load_state_dict(self, state_dict):
        self._data.load_state_dict(state_dict["dataset"])

    def state_dict(self):
        return {"dataset": self._data.state_dict()}

    def set_epoch(self, epoch: int):
        self._data.set_epoch(epoch)


def build_dummy_dataset(size: int, max_seq_len: int) -> "Dataset":
    return DummyDataset(size=size, seq_length=max_seq_len)


def build_mapping_dataset(
    data_path: str,
    transform: Optional[Callable] = None,
    namespace: Literal["train", "test"] = "train",
) -> "Dataset":
    """
    Build mapping dataset.
    Args:
        data_path (str): data path
        transform (Optional[Callable]): transform function
        namespace (Literal["train", "test"]): dataset namespace
    Returns:
        Dataset: mapping dataset
    """
    # data_files = []
    # data_paths = data_path.split(",")
    # for data_path in data_paths:
    #     if os.path.isdir(data_path):
    #         data_files.extend([os.path.join(data_path, fn) for fn in os.listdir(data_path)])
    #     elif os.path.isfile(data_path):
    #         data_files.append(data_path)
    #     else:
    #         raise FileNotFoundError(f"Dataset {data_path} not exists.")
    # file_extenstion = os.path.splitext(data_files[0])[-1][1:]
    # if file_extenstion not in ["parquet", "jsonl", "json", "csv", "arrow"]:
    #     raise ValueError(f"{file_extenstion} files are not supported.")

    # file_extenstion = "json" if file_extenstion == "jsonl" else file_extenstion
    # dataset = load_dataset(file_extenstion, data_files=data_files, split=namespace)

    # return MappingDataset(data=dataset, transform=transform)

    path_obj = Path(data_path)
    dataset = LeRobotDataset(
        repo_id=path_obj.name, 
        root=path_obj.parent,
        local_files_only=True
    )
    return MappingDataset(data=dataset, transform=transform)


def build_iterative_dataset(
    data_path: str,
    transform: Optional[Callable] = None,
    namespace: Literal["train", "test"] = "train",
    seed: int = 42,
) -> "IterableDataset":
    """ "
    Build iterative dataset.
    Args:
        data_path (str): data path
        transform (Optional[Callable]): transform function
        namespace (Literal["train", "test"]): dataset namespace
        seed (int): random seed
    Returns:
        IterableDataset: iterative dataset
    """

    data_files = []
    data_paths = data_path.split(",")
    for data_path in data_paths:
        if os.path.isdir(data_path):
            data_files.extend([os.path.join(data_path, fn) for fn in os.listdir(data_path)])
        elif os.path.isfile(data_path):
            data_files.append(data_path)
        else:
            raise FileNotFoundError(f"Dataset {data_path} not exists.")

    parallel_state = get_parallel_state()
    file_extenstion = os.path.splitext(data_files[0])[-1][1:]
    if file_extenstion not in ["parquet", "jsonl", "json", "csv", "arrow"]:
        raise ValueError(f"{file_extenstion} files are not supported.")

    file_extenstion = "json" if file_extenstion == "jsonl" else file_extenstion
    dataset = load_dataset(file_extenstion, data_files=data_files, split=namespace, streaming=True)
    dataset = dataset.shuffle(seed=seed, buffer_size=10_000)
    dataset = split_dataset_by_node(dataset, parallel_state.dp_rank, parallel_state.dp_size)

    return IterativeDataset(dataset, transform)


def _collect_lerobot_parquet_files(data_path: str) -> List[str]:
    """Collect parquet files from LeRobot dataset structure (data/chunk-*/episode_*.parquet)."""
    path = Path(data_path)
    if (path / "data").exists():
        data_dir = path / "data"
    else:
        data_dir = path
    parquet_files = sorted(data_dir.rglob("*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"No parquet files found under {data_path}")
    return [str(f) for f in parquet_files]


def build_iterative_lerobot_dataset(
    data_path: str,
    transform: Optional[Callable] = None,
    namespace: Literal["train", "test"] = "train",
    seed: int = 42,
    shuffle_buffer_size: int = 2000,
) -> "IterableDataset":
    """
    Build iterative dataset for LeRobot format (images in parquet).
    Handles nested structure: data/chunk-000/episode_000000.parquet.

    Larger shuffle_buffer_size reduces network round-trips when streaming from
    remote storage (OSS/S3) at the cost of memory.
    """
    data_files = _collect_lerobot_parquet_files(data_path)
    parallel_state = get_parallel_state()
    dataset = load_dataset("parquet", data_files=data_files, split=namespace, streaming=True)
    dataset = dataset.shuffle(seed=seed, buffer_size=shuffle_buffer_size)
    dataset = split_dataset_by_node(dataset, parallel_state.dp_rank, parallel_state.dp_size)
    return IterativeDataset(dataset, transform)