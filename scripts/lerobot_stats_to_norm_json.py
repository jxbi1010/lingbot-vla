#!/usr/bin/env python3
"""
Build lingbot-vla norm JSON (ego-style: observation.state + action, mean/std/q01–q99/q02–q98)
from a LeRobot dataset ``meta/stats.json``.

LeRobot often only stores min, max, mean, std, count (no empirical percentiles). In that case
we set q01=min, q99=max, q02=min, q98=max so ``bounds_99_woclip`` / ``bounds_98`` still work;
tighten stats later with ``compute_norm_*.py`` on raw frames if you need true percentiles.

Usage (use conda env ``lbot`` if that is where lerobot deps live)::

  conda activate lbot
  python scripts/lerobot_stats_to_norm_json.py \\
    --dataset_path /path/to/disk-1_Stacking_cups_v30 \\
    --output assets/norm_stats/disk-1_Stacking_cups_v30_norm.json \\
    --action_dim 16
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def _as_float_list(x: Any, name: str) -> list[float]:
    if x is None:
        raise ValueError(f"Missing {name}")
    if isinstance(x, (int, float)):
        return [float(x)]
    if isinstance(x, list):
        return [float(v) for v in x]
    raise TypeError(f"{name}: expected list or number, got {type(x)}")


def _take_dim(vec: list[float], dim: int | None, label: str) -> list[float]:
    if dim is None or len(vec) <= dim:
        return vec
    if len(vec) > dim:
        logger.warning("%s: truncating length %d -> %d", label, len(vec), dim)
        return vec[:dim]
    return vec


def _merge_percentiles_or_minmax(
    block: dict[str, Any],
    dim: int | None,
    prefix: str,
) -> dict[str, list[float]]:
    mean = _take_dim(_as_float_list(block.get("mean"), f"{prefix}.mean"), dim, prefix)
    std = _take_dim(_as_float_list(block.get("std"), f"{prefix}.std"), dim, prefix)
    min_v = _as_float_list(block.get("min"), f"{prefix}.min")
    max_v = _as_float_list(block.get("max"), f"{prefix}.max")
    min_v = _take_dim(min_v, dim, prefix + ".min")
    max_v = _take_dim(max_v, dim, prefix + ".max")

    if len(mean) != len(std) or len(mean) != len(min_v) or len(mean) != len(max_v):
        raise ValueError(
            f"{prefix}: length mismatch mean={len(mean)} std={len(std)} min={len(min_v)} max={len(max_v)}"
        )

    has_q01 = "q01" in block and block["q01"] is not None
    has_q99 = "q99" in block and block["q99"] is not None
    if has_q01 and has_q99:
        q01 = _take_dim(_as_float_list(block["q01"], f"{prefix}.q01"), dim, prefix + ".q01")
        q99 = _take_dim(_as_float_list(block["q99"], f"{prefix}.q99"), dim, prefix + ".q99")
    else:
        logger.warning(
            "%s: no q01/q99 in stats.json — using min as q01 and max as q99 (same for q02/q98 if missing).",
            prefix,
        )
        q01 = list(min_v)
        q99 = list(max_v)

    has_q02 = "q02" in block and block["q02"] is not None
    has_q98 = "q98" in block and block["q98"] is not None
    if has_q02 and has_q98:
        q02 = _take_dim(_as_float_list(block["q02"], f"{prefix}.q02"), dim, prefix + ".q02")
        q98 = _take_dim(_as_float_list(block["q98"], f"{prefix}.q98"), dim, prefix + ".q98")
    else:
        q02 = list(q01)
        q98 = list(q99)

    return {
        "mean": mean,
        "std": std,
        "q01": q01,
        "q99": q99,
        "q02": q02,
        "q98": q98,
    }


def _global_count(block: dict[str, Any]) -> int:
    c = block.get("count")
    if c is None:
        return 0
    if isinstance(c, list) and len(c) > 0:
        return int(c[0])
    return int(c)


def load_lerobot_stats(path: Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def stats_to_norm_payload(
    stats: dict[str, Any],
    action_dim: int | None,
    state_dim: int | None,
) -> tuple[dict[str, Any], int]:
    if "action" not in stats:
        raise KeyError("stats.json must contain 'action'")
    if "observation.state" not in stats:
        raise KeyError("stats.json must contain 'observation.state'")

    action_block = stats["action"]
    state_block = stats["observation.state"]

    norm_stats = {
        "action": _merge_percentiles_or_minmax(action_block, action_dim, "action"),
        "observation.state": _merge_percentiles_or_minmax(state_block, state_dim, "observation.state"),
    }

    ca = _global_count(action_block)
    cs = _global_count(state_block)
    count = max(ca, cs)
    if ca and cs and ca != cs:
        logger.warning("action count (%s) != observation.state count (%s); using max", ca, cs)

    return {"norm_stats": norm_stats, "count": count}


def main() -> int:
    p = argparse.ArgumentParser(description="LeRobot meta/stats.json → ego-style norm JSON")
    p.add_argument(
        "--dataset_path",
        type=str,
        default=None,
        help="LeRobot dataset root (must contain meta/stats.json)",
    )
    p.add_argument(
        "--stats_json",
        type=str,
        default=None,
        help="Direct path to stats.json (overrides --dataset_path)",
    )
    p.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output path, e.g. assets/norm_stats/disk-1_Stacking_cups_v30_norm.json",
    )
    p.add_argument(
        "--action_dim",
        type=int,
        default=None,
        help="Truncate action (and default state) to this many dims (default: full length in file)",
    )
    p.add_argument(
        "--state_dim",
        type=int,
        default=None,
        help="Truncate observation.state separately (default: same as --action_dim or full)",
    )
    args = p.parse_args()

    if args.stats_json:
        stats_path = Path(args.stats_json).expanduser().resolve()
    elif args.dataset_path:
        stats_path = Path(args.dataset_path).expanduser().resolve() / "meta" / "stats.json"
    else:
        logger.error("Provide --dataset_path or --stats_json")
        return 1

    if not stats_path.is_file():
        logger.error("Not found: %s", stats_path)
        return 1

    stats = load_lerobot_stats(stats_path)
    state_dim = args.state_dim if args.state_dim is not None else args.action_dim
    payload = stats_to_norm_payload(stats, args.action_dim, state_dim)

    out = Path(args.output).expanduser()
    if not out.is_absolute():
        repo = Path(__file__).resolve().parent.parent
        out = (repo / out).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
        f.write("\n")

    logger.info("Wrote %s (action dim=%d, state dim=%d, count=%s)", out, len(payload["norm_stats"]["action"]["mean"]), len(payload["norm_stats"]["observation.state"]["mean"]), payload["count"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
