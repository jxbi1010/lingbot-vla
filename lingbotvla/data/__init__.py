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

from importlib import import_module
from typing import Any

from .chat_template import build_chat_template
from .data_collator import (
    CollatePipeline,
    DataCollatorWithPacking,
    DataCollatorWithPadding,
    DataCollatorWithPositionIDs,
    MakeMicroBatchCollator,
    TextSequenceShardCollator,
    UnpackDataCollator,
)
from .data_loader import build_dataloader
from .data_transform import (
    VLADataCollatorWithPacking,
)

# Avoid importing `.dataset` at package import: it chains into vla_data/lerobot and
# can pull `lerobot.policies` (heavy). Use lazy exports (PEP 562).
_DATASET_EXPORTS = (
    "build_iterative_dataset",
    "build_iterative_lerobot_dataset",
    "build_mapping_dataset",
    "liberoDataset",
    "RobotwinDataset",
)

__all__ = [
    "build_chat_template",
    "CollatePipeline",
    "DataCollatorWithPacking",
    "DataCollatorWithPadding",
    "DataCollatorWithPositionIDs",
    "MakeMicroBatchCollator",
    "TextSequenceShardCollator",
    "UnpackDataCollator",
    "build_dataloader",
    "VLADataCollatorWithPacking",
    *_DATASET_EXPORTS,
]


def __getattr__(name: str) -> Any:
    if name in _DATASET_EXPORTS:
        mod = import_module(".dataset", __name__)
        return getattr(mod, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(
        list(globals())
        + list(_DATASET_EXPORTS)
    )
