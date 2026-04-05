#!/usr/bin/env python3
"""
Merge several LeRobot ``meta/stats.json`` files into one lingbot-vla norm JSON
(same schema as ``lerobot_stats_to_norm_json.py``: ``norm_stats`` with
``action`` + ``observation.state``, each with mean/std/q01–q99/q02–q98).

Aggregation (per feature dimension):
  - **mean**: count-weighted average across datasets.
  - **std**: pooled std from per-dataset mean/variance and counts
    (same formula as combining parallel samples).
  - **min / max**: element-wise min of mins, max of maxs.
  - **q01 / q99 / q02 / q98**: if present in stats, element-wise min of q01s and max of q99s
    (conservative envelope over datasets); if missing, derived from merged min/max like the
    single-file converter.

Cobot-style bimanual data typically uses **16** action dims and **16** state dims; pass
``--action_dim 16`` (and optionally ``--state_dim 16``).

Examples::

  # Explicit dataset roots (each must contain meta/stats.json)
  python scripts/merge_lerobot_stats_to_norm_json.py \\
    --dataset_paths /data/cobot/disk-1_A_v30 /data/cobot/disk-2_B_v30 \\
    --output assets/norm_stats/cobot_merge_norm.json \\
    --action_dim 16

  # Same bundle as ego_finetune_v3_cobot.yaml (task JSON + hours + option)
  python scripts/merge_lerobot_stats_to_norm_json.py \\
    --cobot_task_options_json configs/tasks/cobot_magic_dataset_options.json \\
    --cobot_lerobot_root /path/to/cobot_magic_raw_lerobot \\
    --cobot_hours 100 --cobot_option A \\
    --output assets/norm_stats/cobot_100hr_A_norm.json \\
    --action_dim 16
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
from pathlib import Path
from typing import Any

_SCRIPTS_DIR = Path(__file__).resolve().parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

# Reuse helpers from single-file converter (same repo, scripts/).
from lerobot_stats_to_norm_json import (  # noqa: E402
    _as_float_list,
    _global_count,
    _take_dim,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def _load_stats_json(path: Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _merge_percentiles_from_blocks(
    blocks: list[dict[str, Any]],
    dim: int | None,
    prefix: str,
) -> dict[str, list[float]]:
    """Merge multiple raw LeRobot stat blocks for one feature (e.g. 'action')."""
    if not blocks:
        raise ValueError(f"{prefix}: no blocks to merge")

    weighted: list[tuple[int, dict[str, Any]]] = []
    for b in blocks:
        n = _global_count(b)
        if n <= 0:
            logger.warning("%s: skipping block with count=%s", prefix, n)
            continue
        weighted.append((n, b))
    if not weighted:
        raise ValueError(f"{prefix}: all blocks had zero count")

    w_sum = sum(w for w, _ in weighted)
    # --- mean (weighted) ---
    first_mean = _as_float_list(weighted[0][1]["mean"], f"{prefix}.mean")
    dlen = len(_take_dim(first_mean, dim, prefix))
    mean_acc = [0.0] * dlen
    for w, b in weighted:
        m = _take_dim(_as_float_list(b["mean"], f"{prefix}.mean"), dim, prefix)
        if len(m) != dlen:
            raise ValueError(f"{prefix}: inconsistent dim {len(m)} vs {dlen}")
        for i in range(dlen):
            mean_acc[i] += (w / w_sum) * m[i]
    mean = mean_acc

    # --- std (pooled from per-group mean and var) ---
    std_out = [0.0] * dlen
    for i in range(dlen):
        pooled_second_moment = 0.0
        for w, b in weighted:
            m = _take_dim(_as_float_list(b["mean"], f"{prefix}.mean"), dim, prefix)[i]
            s = _take_dim(_as_float_list(b["std"], f"{prefix}.std"), dim, prefix)[i]
            pooled_second_moment += (w / w_sum) * (m * m + s * s)
        mu = mean[i]
        var = max(0.0, pooled_second_moment - mu * mu)
        std_out[i] = math.sqrt(var)

    # --- min / max ---
    min_v = None
    max_v = None
    for w, b in weighted:
        lo = _take_dim(_as_float_list(b["min"], f"{prefix}.min"), dim, prefix + ".min")
        hi = _take_dim(_as_float_list(b["max"], f"{prefix}.max"), dim, prefix + ".max")
        if len(lo) != dlen or len(hi) != dlen:
            raise ValueError(f"{prefix}: min/max dim mismatch")
        if min_v is None:
            min_v = list(lo)
            max_v = list(hi)
        else:
            for j in range(dlen):
                min_v[j] = min(min_v[j], lo[j])
                max_v[j] = max(max_v[j], hi[j])
    assert min_v is not None and max_v is not None

    # --- quantiles: conservative merge; missing q01/q99 in a file -> use its min/max ---
    def _merge_q(qkey: str) -> list[float]:
        has_any = any(qkey in b and b[qkey] is not None for _, b in weighted)
        if not has_any:
            return list(min_v if qkey in ("q01", "q02") else max_v)

        if qkey in ("q01", "q02"):
            out = [float("inf")] * dlen
            for w, b in weighted:
                if qkey in b and b[qkey] is not None:
                    v = _take_dim(_as_float_list(b[qkey], f"{prefix}.{qkey}"), dim, prefix)
                    for j in range(dlen):
                        out[j] = min(out[j], v[j])
                else:
                    lo = _take_dim(_as_float_list(b["min"], f"{prefix}.min"), dim, prefix)
                    for j in range(dlen):
                        out[j] = min(out[j], lo[j])
            return out
        # q99, q98: max
        out = [float("-inf")] * dlen
        for w, b in weighted:
            if qkey in b and b[qkey] is not None:
                v = _take_dim(_as_float_list(b[qkey], f"{prefix}.{qkey}"), dim, prefix)
                for j in range(dlen):
                    out[j] = max(out[j], v[j])
            else:
                hi = _take_dim(_as_float_list(b["max"], f"{prefix}.max"), dim, prefix)
                for j in range(dlen):
                    out[j] = max(out[j], hi[j])
        return out

    q01 = _merge_q("q01")
    q99 = _merge_q("q99")
    q02 = _merge_q("q02")
    q98 = _merge_q("q98")

    return {
        "mean": mean,
        "std": std_out,
        "q01": q01,
        "q99": q99,
        "q02": q02,
        "q98": q98,
    }


def _merge_stats_json_dicts(
    stats_dicts: list[dict[str, Any]],
    action_dim: int | None,
    state_dim: int | None,
) -> dict[str, Any]:
    actions = [s["action"] for s in stats_dicts]
    states = [s["observation.state"] for s in stats_dicts]
    merged_action = _merge_percentiles_from_blocks(actions, action_dim, "action")
    merged_state = _merge_percentiles_from_blocks(states, state_dim, "observation.state")

    count_total = 0
    for s in stats_dicts:
        count_total += _global_count(s["action"])
    # If observation.state count differs, sum state counts — usually same as action frame count per dataset
    c2 = sum(_global_count(s["observation.state"]) for s in stats_dicts)
    if c2 != count_total:
        logger.warning("Summed action count (%s) != summed observation.state count (%s); using max", count_total, c2)
    count = max(count_total, c2)

    return {
        "action": merged_action,
        "observation.state": merged_state,
        "_merge_count_hint": count,
    }


def _resolve_cobot_bundle_paths(
    task_json: Path,
    lerobot_root: Path,
    hours: int,
    option: str,
) -> list[Path]:
    opt = option.strip().upper()
    if opt not in ("A", "B", "C"):
        raise ValueError("cobot_option must be A, B, or C")
    key = f"{int(hours)}hr_datasets_{opt}"
    with open(task_json, encoding="utf-8") as f:
        data = json.load(f)
    if key not in data:
        raise KeyError(f"{key!r} not in {task_json}; keys sample: {sorted(data.keys())[:12]}")
    names = [d["name"] for d in data[key].get("datasets", [])]
    return [lerobot_root / n for n in names]


def main() -> int:
    p = argparse.ArgumentParser(
        description="Merge multiple LeRobot meta/stats.json → one ego-style norm JSON"
    )
    p.add_argument(
        "--dataset_paths",
        nargs="*",
        default=[],
        help="LeRobot dataset roots (each …/meta/stats.json must exist)",
    )
    p.add_argument(
        "--stats_json",
        nargs="*",
        default=[],
        help="Explicit paths to stats.json files (alternative to --dataset_paths)",
    )
    p.add_argument(
        "--cobot_task_options_json",
        type=str,
        default=None,
        help="Path to cobot_magic_dataset_options.json",
    )
    p.add_argument(
        "--cobot_lerobot_root",
        type=str,
        default=None,
        help="Parent of dataset folders (used with cobot_task_options_json)",
    )
    p.add_argument(
        "--cobot_hours",
        type=int,
        default=None,
        help="100, 200, 500, 1000, or 2000",
    )
    p.add_argument(
        "--cobot_option",
        type=str,
        default="A",
        help="A, B, or C",
    )
    p.add_argument("--output", type=str, required=True, help="Output norm JSON path")
    p.add_argument(
        "--action_dim",
        type=int,
        default=16,
        help="Truncate action (and default state) to this many dims (cobot: 16)",
    )
    p.add_argument(
        "--state_dim",
        type=int,
        default=None,
        help="Truncate observation.state (default: same as --action_dim)",
    )
    args = p.parse_args()

    repo = Path(__file__).resolve().parent.parent

    paths: list[Path] = []
    for d in args.dataset_paths:
        paths.append(Path(d).expanduser().resolve())
    for s in args.stats_json:
        paths.append(Path(s).expanduser().resolve())

    if args.cobot_task_options_json:
        if not args.cobot_lerobot_root or args.cobot_hours is None:
            p.error("cobot_task_options_json requires cobot_lerobot_root and cobot_hours")
        tj = Path(args.cobot_task_options_json).expanduser()
        if not tj.is_absolute():
            tj = (repo / tj).resolve()
        root = Path(args.cobot_lerobot_root).expanduser().resolve()
        paths = _resolve_cobot_bundle_paths(tj, root, args.cobot_hours, args.cobot_option)

    if not paths:
        p.error("Provide --dataset_paths and/or --stats_json, or --cobot_* options")

    stats_paths: list[Path] = []
    for base in paths:
        sp = base if base.name == "stats.json" else base / "meta" / "stats.json"
        if not sp.is_file():
            logger.error("Missing stats: %s", sp)
            return 1
        stats_paths.append(sp)

    stats_dicts = [_load_stats_json(sp) for sp in stats_paths]
    state_dim = args.state_dim if args.state_dim is not None else args.action_dim

    merged_raw = _merge_stats_json_dicts(stats_dicts, args.action_dim, state_dim)
    count_hint = int(merged_raw.pop("_merge_count_hint", 0))

    payload = {
        "norm_stats": {
            "action": merged_raw["action"],
            "observation.state": merged_raw["observation.state"],
        },
        "count": count_hint,
    }

    out = Path(args.output).expanduser()
    if not out.is_absolute():
        out = (repo / out).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
        f.write("\n")

    ns = payload["norm_stats"]
    logger.info(
        "Wrote %s (merged %d datasets, action_dim=%d state_dim=%d count=%s)",
        out,
        len(stats_paths),
        len(ns["action"]["mean"]),
        len(ns["observation.state"]["mean"]),
        payload["count"],
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
