"""
Standalone script to compute normalization statistics for LeRobot-format data.
No lingbotvla dependency - uses only numpy, torch, lerobot, and tqdm.

Usage:
  # From config (e.g. configs/norm/ego_pretrain.yaml):
  python compute_norm_ego_standalone.py --config configs/norm/ego_pretrain.yaml

  # CLI args (override config when both provided):
  python compute_norm_ego_standalone.py \\
    --train_path /path/to/lerobot_dataset \\
    --norm_path assets/norm_stats/ego_pretrain_norm.json \\
    --batch_size 512 \\
    --chunk_size 50
"""

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import yaml
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

# LeRobot dataset (v3 layout)
from lerobot.datasets.dataset_metadata import LeRobotDatasetMetadata
from lerobot.datasets.lerobot_dataset import LeRobotDataset

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _normalize_dataset_roots(raw) -> list:
    """Accept YAML string, comma-separated string, or list of paths."""
    if raw is None:
        return []
    if isinstance(raw, list):
        out = []
        for p in raw:
            s = str(p).strip()
            if not s:
                continue
            if "," in s:
                out.extend(x.strip() for x in s.split(",") if x.strip())
            else:
                out.append(s)
        return out
    s = str(raw).strip()
    if not s:
        return []
    if "," in s:
        return [x.strip() for x in s.split(",") if x.strip()]
    return [s]


# ---------------------------------------------------------------------------
# Inlined RunningStats (from lingbotvla.utils.normalize) - no pydantic/numpydantic
# ---------------------------------------------------------------------------


class RunningStats:
    """Compute running statistics of a batch of vectors."""

    def __init__(self):
        self._count = 0
        self._mean = None
        self._mean_of_squares = None
        self._min = None
        self._max = None
        self._histograms = None
        self._bin_edges = None
        self._num_quantile_bins = 5000

    def update(self, batch: np.ndarray) -> None:
        if batch.ndim == 1:
            batch = batch.reshape(-1, 1)

        num_elements, vector_length = batch.shape

        if self._count == 0:
            self._mean = np.mean(batch, axis=0).astype(np.float64)
            self._mean_of_squares = np.mean(batch**2, axis=0).astype(np.float64)
            self._min = np.min(batch, axis=0)
            self._max = np.max(batch, axis=0)
            self._histograms = [np.zeros(self._num_quantile_bins) for _ in range(vector_length)]
            self._bin_edges = [
                np.linspace(self._min[i] - 1e-10, self._max[i] + 1e-10, self._num_quantile_bins + 1)
                for i in range(vector_length)
            ]
        else:
            if vector_length != self._mean.size:
                raise ValueError("Vector length mismatch.")
            new_max = np.max(batch, axis=0)
            new_min = np.min(batch, axis=0)
            max_changed = np.any(new_max > self._max)
            min_changed = np.any(new_min < self._min)
            self._max = np.maximum(self._max, new_max)
            self._min = np.minimum(self._min, new_min)
            if max_changed or min_changed:
                self._adjust_histograms()

        self._count += num_elements
        batch_mean = np.mean(batch, axis=0)
        batch_mean_of_squares = np.mean(batch**2, axis=0)
        self._mean += (batch_mean - self._mean) * (num_elements / self._count)
        self._mean_of_squares += (batch_mean_of_squares - self._mean_of_squares) * (
            num_elements / self._count
        )
        self._update_histograms(batch)

    def get_statistics(self, chunk_size=None):
        if self._count < 2:
            raise ValueError("Cannot compute statistics for less than 2 vectors.")

        variance = self._mean_of_squares - self._mean**2
        stddev = np.sqrt(np.maximum(0, variance))
        q01, q99 = self._compute_quantiles([0.01, 0.99])
        q02, q98 = self._compute_quantiles([0.02, 0.98])

        mean = self._mean.copy()
        if chunk_size is not None:
            mean = mean.reshape(chunk_size, -1)
            stddev = stddev.reshape(chunk_size, -1)
            q01 = q01.reshape(chunk_size, -1)
            q99 = q99.reshape(chunk_size, -1)
            q02 = q02.reshape(chunk_size, -1)
            q98 = q98.reshape(chunk_size, -1)

        return {
            "mean": mean.tolist(),
            "std": stddev.tolist(),
            "q01": q01.tolist(),
            "q99": q99.tolist(),
            "q02": q02.tolist(),
            "q98": q98.tolist(),
        }

    def _adjust_histograms(self):
        for i in range(len(self._histograms)):
            old_edges = self._bin_edges[i]
            new_edges = np.linspace(self._min[i], self._max[i], self._num_quantile_bins + 1)
            new_hist, _ = np.histogram(old_edges[:-1], bins=new_edges, weights=self._histograms[i])
            self._histograms[i] = new_hist
            self._bin_edges[i] = new_edges

    def _update_histograms(self, batch: np.ndarray) -> None:
        for i in range(batch.shape[1]):
            hist, _ = np.histogram(batch[:, i], bins=self._bin_edges[i])
            self._histograms[i] += hist

    def _compute_quantiles(self, quantiles):
        results = []
        for q in quantiles:
            target_count = q * self._count
            q_values = []
            for hist, edges in zip(self._histograms, self._bin_edges, strict=True):
                cumsum = np.cumsum(hist)
                idx = np.searchsorted(cumsum, target_count)
                q_values.append(edges[idx])
            results.append(np.array(q_values))
        return results


# ---------------------------------------------------------------------------
# Dataset wrapper (replaces VlaDataset)
# ---------------------------------------------------------------------------


def make_lerobot_dataset(repo_id: str, action_name: str = "action") -> torch.utils.data.Dataset:
    """Create a LeRobot dataset for norm computation. Returns raw items with observation.state and action."""
    path = Path(repo_id)
    if path.is_dir():
        rid, root = path.name, str(path)
    else:
        rid, root = repo_id, None
    meta = LeRobotDatasetMetadata(rid, root=root) if root else LeRobotDatasetMetadata(rid)
    delta_timestamps = {action_name: [t / meta.fps for t in range(50)]}
    kwargs = {"repo_id": rid, "delta_timestamps": delta_timestamps}
    if root:
        kwargs["root"] = root
    return LeRobotDataset(**kwargs)


# ---------------------------------------------------------------------------
# Main logic
# ---------------------------------------------------------------------------


def compute_norm_stats(
    train_path: str,
    norm_path: str,
    batch_size: int = 512,
    chunk_size: int = 50,
    state_key: str = "observation.state",
    action_key: str = "action",
    num_workers: int = 16,
) -> bool:
    """
    Compute normalization statistics for LeRobot-format data and save to JSON.

    Returns True on success, False on failure.
    """
    dataset = make_lerobot_dataset(train_path, action_name=action_key)

    state_norm_keys = [state_key]
    action_norm_keys = [action_key]
    delta_norm = {action_key: False}

    stats_dict = {
        key: RunningStats()
        for key in action_norm_keys + state_norm_keys
    }

    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
        drop_last=True,
    )

    success = True
    try:
        for batch in tqdm(data_loader, desc=f"Computing stats of {train_path}"):
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

    except Exception as e:
        logger.exception("Failed during norm computation")
        return False

    norm_stats = {}
    for key, running_stats in stats_dict.items():
        if key in delta_norm and delta_norm[key]:
            norm_stats[key] = running_stats.get_statistics(chunk_size=chunk_size)
        else:
            norm_stats[key] = running_stats.get_statistics()

    count = list(stats_dict.values())[0]._count
    output_path = Path(norm_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    payload = {"norm_stats": norm_stats, "count": count}
    output_path.write_text(json.dumps(payload, indent=2))

    logger.info(f"Wrote norm stats to: {output_path} (count={count})")
    return True


def load_config(path: str) -> dict:
    """Load YAML config. Expects data.train_path, data.norm_path, train.global_batch_size, data.chunk_size."""
    with open(path) as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description="Compute norm stats for LeRobot data")
    parser.add_argument(
        "config",
        nargs="?",
        default=None,
        help="Path to YAML config (e.g. configs/norm/ego_pretrain.yaml)",
    )
    parser.add_argument("--train_path", help="Path to LeRobot dataset (repo_id)")
    parser.add_argument("--norm_path", help="Path to save norm stats JSON")
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--chunk_size", type=int, default=50)
    parser.add_argument("--state_key", default="observation.state")
    parser.add_argument("--action_key", default="action")
    parser.add_argument("--num_workers", type=int, default=16)
    args = parser.parse_args()


    # Load from config if provided (positional: script.py configs/norm/ego_pretrain.yaml)
    config = {}
    if args.config:
        config = load_config(args.config)
        logger.info(f"Loaded config from {args.config}")

    # Resolve values: CLI overrides config
    raw_tp = args.train_path or (config.get("data", {}).get("train_path"))
    roots = _normalize_dataset_roots(raw_tp)
    norm_path = args.norm_path or (config.get("data", {}).get("norm_path"))

    if len(roots) != 1 or not norm_path:
        parser.error("Exactly one train_path root and norm_path required (use --config or CLI)")

    train_path = roots[0]

    success = compute_norm_stats(
        train_path=train_path,
        norm_path=norm_path,
        batch_size=args.batch_size,
        chunk_size=args.chunk_size,
        state_key=args.state_key,
        action_key=args.action_key,
        num_workers=args.num_workers,
    )

    if not success:
        raise SystemExit(1)


if __name__ == "__main__":
    main()


# python lingbot-vla/scripts/compute_norm_from_episodes_stats.py \
#   --episodes_stats /home/ss-oss1/data/user/jiankai/Data/lerobot_test_data/lerobot_data_10k/meta/episodes_stats.jsonl \
#   --output assets/norm_stats/ego10k_pretrain_norm.json