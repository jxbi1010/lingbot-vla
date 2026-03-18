"""
Compute normalization statistics from LeRobot chunk stats (episodes_state_action JSON files).

Uses a sampling approach: loads each chunk file and samples frames to compute global
mean, std, and quantiles (q01, q02, q98, q99). Output format matches ego_pretrain_10k_norm.json.

Usage:
  python compute_norm_lerobot_chunks.py \\
    --stats_dir /home/ss-oss1/data/dataset/egocentric/training_egocentric/retarget_lerobot/database_lerobot_00/stats \\
    --output lingbot-vla/norm_stats/database_lerobot_00_norm.json \\
    [--max_frames_per_chunk 50000] \\
    [--sample_ratio 1.0]
"""

import argparse
import json
import logging
from pathlib import Path

import numpy as np
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# RunningStats (from compute_norm_ego_standalone / lingbotvla.utils.normalize)
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
# Chunk loading and sampling
# ---------------------------------------------------------------------------


def sample_frames_from_chunk(
    chunk_path: Path,
    max_frames_per_chunk: int | None,
    sample_ratio: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Load a chunk and return sampled state and action arrays.

    Returns:
        state_arr: (N, dim) float64
        action_arr: (N, dim) float64
    """
    with open(chunk_path) as f:
        data = json.load(f)

    episodes = data["episodes"]
    all_states = []
    all_actions = []

    for ep in episodes:
        states = np.array(ep["observation.state"], dtype=np.float64)
        actions = np.array(ep["action"], dtype=np.float64)
        n = len(states)
        if n == 0:
            continue

        # Apply sample_ratio: randomly keep frames with probability sample_ratio
        if sample_ratio < 1.0:
            mask = rng.random(n) < sample_ratio
            states = states[mask]
            actions = actions[mask]

        all_states.append(states)
        all_actions.append(actions)

    if not all_states:
        return np.empty((0, 0)), np.empty((0, 0))

    state_arr = np.vstack(all_states)
    action_arr = np.vstack(all_actions)

    # Apply max_frames_per_chunk sampling if needed
    n_total = len(state_arr)
    if max_frames_per_chunk is not None and n_total > max_frames_per_chunk:
        indices = rng.choice(n_total, size=max_frames_per_chunk, replace=False)
        state_arr = state_arr[indices]
        action_arr = action_arr[indices]

    return state_arr, action_arr


def main():
    parser = argparse.ArgumentParser(
        description="Compute norm stats from LeRobot chunk stats (episodes_state_action JSON)"
    )
    parser.add_argument(
        "--stats_dir",
        required=True,
        help="Path to stats directory containing chunk-*_episodes_state_action.json",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output path for norm stats JSON (e.g. lingbot-vla/norm_stats/database_lerobot_00_norm.json)",
    )
    parser.add_argument(
        "--max_frames_per_chunk",
        type=int,
        default=None,
        help="Max frames to sample per chunk (default: use all). Reduces memory for large chunks.",
    )
    parser.add_argument(
        "--sample_ratio",
        type=float,
        default=1.0,
        help="Fraction of frames to keep per episode (0-1). Default 1.0 = use all.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for sampling",
    )
    args = parser.parse_args()

    stats_dir = Path(args.stats_dir)
    if not stats_dir.exists():
        raise FileNotFoundError(f"Stats dir not found: {stats_dir}")

    chunk_files = sorted(
        stats_dir.glob("chunk-*_episodes_state_action.json"),
        key=lambda p: p.name,
    )
    if not chunk_files:
        raise FileNotFoundError(f"No chunk-*_episodes_state_action.json in {stats_dir}")

    logger.info(f"Found {len(chunk_files)} chunk files in {stats_dir}")

    stats_dict = {
        "observation.state": RunningStats(),
        "action": RunningStats(),
    }
    rng = np.random.default_rng(args.seed)
    total_frames = 0

    for chunk_path in tqdm(chunk_files, desc="Processing chunks"):
        state_arr, action_arr = sample_frames_from_chunk(
            chunk_path,
            max_frames_per_chunk=args.max_frames_per_chunk,
            sample_ratio=args.sample_ratio,
            rng=rng,
        )
        if len(state_arr) == 0:
            logger.warning(f"Chunk {chunk_path.name} has no data, skipping")
            continue

        stats_dict["observation.state"].update(state_arr)
        stats_dict["action"].update(action_arr)
        total_frames += len(state_arr)

    norm_stats = {
        key: rs.get_statistics()
        for key, rs in stats_dict.items()
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    count = stats_dict["observation.state"]._count
    payload = {"norm_stats": norm_stats, "count": count}
    output_path.write_text(json.dumps(payload, indent=2))

    logger.info(f"Wrote norm stats to {output_path} (count={count}, total_frames_sampled={total_frames})")


if __name__ == "__main__":
    main()
