"""
Compute normalization stats from episodes_stats.jsonl metadata (no raw data needed).

Merges per-episode min, max, mean, std, count into global stats. Approximates
quantiles (q01, q02, q98, q99) using mean ± z*std (normal distribution assumption).

Usage:
  python compute_norm_from_episodes_stats.py \\
    --episodes_stats datasets/ego_data/meta/episodes_stats.jsonl \\
    --output assets/norm_stats/ego_pretrain_norm.json \\
    --keys observation.state action
"""

import argparse
import json
import logging
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Z-scores for approximate quantiles (standard normal)
# 1st percentile: z ≈ -2.326, 99th: z ≈ 2.326
# 2nd percentile: z ≈ -2.054, 98th: z ≈ 2.054
Z_Q01 = -2.326347874
Z_Q99 = 2.326347874
Z_Q02 = -2.053748911
Z_Q98 = 2.053748911

NORM_KEYS = ["observation.state", "action"]


def _get_count(ep_stats: dict) -> int:
    """Extract scalar count from episode stats (may be list or scalar)."""
    c = ep_stats.get("count")
    if c is None:
        return 0
    return int(c[0]) if isinstance(c, (list, tuple)) else int(c)


def merge_stats(episodes_stats: list[dict], keys: list[str]) -> dict:
    """
    Merge per-episode stats into global stats for each key.

    For each key:
    - mean: weighted average by count
    - std: from merged mean_of_squares (Var = E[X^2] - E[X]^2)
    - min/max: element-wise min/max across episodes
    - q01, q02, q98, q99: approximated as mean ± z*std, clipped to [min, max]
    """
    merged = {}
    for key in keys:
        if key not in episodes_stats[0]["stats"]:
            logger.warning(f"Key {key} not in episodes_stats, skipping")
            continue

        total_count = 0
        sum_mean_weighted = None
        sum_mean_of_squares_weighted = None
        global_min = None
        global_max = None

        for ep in episodes_stats:
            s = ep["stats"][key]
            count = _get_count(s)
            if count == 0:
                continue

            mean = np.array(s["mean"], dtype=np.float64)
            std = np.array(s["std"], dtype=np.float64)
            mn = np.array(s["min"], dtype=np.float64)
            mx = np.array(s["max"], dtype=np.float64)

            # mean_of_squares = E[X^2] = Var(X) + E[X]^2 = std^2 + mean^2
            mean_of_squares = std**2 + mean**2

            if sum_mean_weighted is None:
                sum_mean_weighted = mean * count
                sum_mean_of_squares_weighted = mean_of_squares * count
                global_min = mn.copy()
                global_max = mx.copy()
            else:
                sum_mean_weighted += mean * count
                sum_mean_of_squares_weighted += mean_of_squares * count
                global_min = np.minimum(global_min, mn)
                global_max = np.maximum(global_max, mx)

            total_count += count

        if total_count == 0:
            logger.warning(f"No data for key {key}")
            continue

        global_mean = sum_mean_weighted / total_count
        global_mean_of_squares = sum_mean_of_squares_weighted / total_count
        variance = np.maximum(0, global_mean_of_squares - global_mean**2)
        global_std = np.sqrt(variance)

        # Avoid division by zero
        eps = 1e-8
        global_std = np.maximum(global_std, eps)

        # Approximate quantiles: mean ± z*std, clipped to [min, max]
        q01 = np.clip(global_mean + Z_Q01 * global_std, global_min, global_max)
        q99 = np.clip(global_mean + Z_Q99 * global_std, global_min, global_max)
        q02 = np.clip(global_mean + Z_Q02 * global_std, global_min, global_max)
        q98 = np.clip(global_mean + Z_Q98 * global_std, global_min, global_max)

        merged[key] = {
            "mean": global_mean.tolist(),
            "std": global_std.tolist(),
            "q01": q01.tolist(),
            "q99": q99.tolist(),
            "q02": q02.tolist(),
            "q98": q98.tolist(),
        }

    return merged


def main():
    parser = argparse.ArgumentParser(
        description="Compute norm stats from episodes_stats.jsonl (no raw data)"
    )
    parser.add_argument(
        "--episodes_stats",
        required=True,
        help="Path to episodes_stats.jsonl",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Path to output norm JSON (e.g. assets/norm_stats/ego_pretrain_norm.json)",
    )
    parser.add_argument(
        "--keys",
        nargs="+",
        default=NORM_KEYS,
        help=f"Keys to merge (default: {NORM_KEYS})",
    )
    args = parser.parse_args()

    episodes_stats_path = Path(args.episodes_stats)
    if not episodes_stats_path.exists():
        raise FileNotFoundError(f"episodes_stats not found: {episodes_stats_path}")

    episodes_stats = []
    with open(episodes_stats_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            episodes_stats.append(json.loads(line))

    if not episodes_stats:
        raise ValueError("episodes_stats.jsonl is empty")

    logger.info(f"Loaded {len(episodes_stats)} episodes from {episodes_stats_path}")

    norm_stats = merge_stats(episodes_stats, args.keys)

    # Sanity: log that state and action are merged separately (they can be similar in teleop data)
    if "observation.state" in norm_stats and "action" in norm_stats:
        s_mean = norm_stats["observation.state"]["mean"][0]
        a_mean = norm_stats["action"]["mean"][0]
        logger.info(f"observation.state mean[0]={s_mean:.6f}, action mean[0]={a_mean:.6f} (merged separately)")

    # Total sample count (from first key present in stats)
    count_key = next(
        (k for k in args.keys if episodes_stats and k in episodes_stats[0]["stats"]),
        args.keys[0],
    )
    total_count = sum(
        _get_count(ep["stats"][count_key])
        for ep in episodes_stats
        if count_key in ep["stats"]
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    payload = {"norm_stats": norm_stats, "count": total_count}
    output_path.write_text(json.dumps(payload, indent=2))

    logger.info(f"Wrote norm stats to {output_path} (count={total_count})")


if __name__ == "__main__":
    main()
