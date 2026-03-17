"""
Compute normalization statistics for LeRobot-format data in jx_ws/lerobot_data/.

Mimics compute_norm_robotwin_5.py with default paths for lerobot_data.
"""

import json
from pathlib import Path
from dataclasses import asdict, dataclass, field

import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader

from lingbotvla.utils import helper
from lingbotvla.utils.arguments import DataArguments, ModelArguments, TrainingArguments, parse_args
import lingbotvla.utils.normalize as normalize
from lingbotvla.data.vla_data.base_dataset import VlaDataset

logger = helper.create_logger(__name__)


@dataclass
class MyDataArguments(DataArguments):
    norm_path: str = field(
        default= None,
        metadata={"help": "Path to save norm stats."},
    )
    chunk_size: int = field(
        default=50,
        metadata={"help": "Chunk size of action."},
    )


@dataclass
class Arguments:
    model: "ModelArguments" = field(default_factory=ModelArguments)
    data: "MyDataArguments" = field(default_factory=MyDataArguments)
    train: "TrainingArguments" = field(default_factory=TrainingArguments)


def compute_norm(dataset, task_id, batch_size, stats_dict, state_norm_keys, action_norm_keys, delta_norm):
    data_loader = DataLoader(
        dataset, batch_size=batch_size, num_workers=16, shuffle=False, drop_last=True
    )
    success = True
    try:
        for batch in tqdm(data_loader, desc=f"Computing stats of {task_id}"):
            for key in state_norm_keys:
                values = batch[key]
                if hasattr(values, "numpy"):
                    values = values.numpy()
                values = np.asarray(values)
                stats_dict[key].update(values.reshape(-1, values.shape[-1]))
            for key in action_norm_keys:
                values = (
                    batch[key][:, 0]
                    if not delta_norm[key]
                    else batch[key].reshape(batch[key].shape[0], -1)
                )
                if hasattr(values, "numpy"):
                    values = values.numpy()
                values = np.asarray(values)
                stats_dict[key].update(values.reshape(-1, values.shape[-1]))
    except Exception:
        success = False
    return success


def main():
    args = parse_args(Arguments)
    logger.info(f"Process rank: {args.train.global_rank}, world size: {args.train.world_size}")

    # Compute norm is CPU-only; run only on rank 0 to avoid redundant work and shutdown races
    if args.train.global_rank != 0:
        return

    logger.info_rank0(json.dumps(asdict(args), indent=2))
    logger.info_rank0("Prepare data")

    train_path = args.data.train_path
    assert args.data.datasets_type == "vla", "datasets_type must be 'vla'"

    dataset = VlaDataset(repo_id=train_path, action_name="action")

    state_norm_keys = ["observation.state"]
    action_norm_keys = ["action"]
    delta_norm = {"action": False}
    stats_dict = {
        key: normalize.RunningStats()
        for key in action_norm_keys + state_norm_keys
    }
    chunk_size = args.data.chunk_size

    success = False
    try:
        success = compute_norm(
            dataset,
            train_path,
            args.train.global_batch_size,
            stats_dict,
            state_norm_keys,
            action_norm_keys,
            delta_norm,
        )
    except Exception as e:
        print(f"Failed: {train_path} {e}")

    if success:
        norm_stats = {}
        for key, running_stats in stats_dict.items():
            if key in delta_norm and delta_norm[key]:
                norm_stats[key] = running_stats.get_statistics(chunk_size=chunk_size)
            else:
                norm_stats[key] = running_stats.get_statistics()

        output_path = Path(args.data.norm_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        count = list(stats_dict.values())[0]._count
        normalize.save(output_path, norm_stats, count)
        print(f"Wrote norm stats to: {output_path} (count={count})")


if __name__ == "__main__":
    main()
