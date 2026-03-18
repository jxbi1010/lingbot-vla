#!/usr/bin/env python3
"""
Check if parquet and video files in a LeRobot dataset match the expected structure
from meta/info.json. Reports any missing files.
"""
import json
import sys
from pathlib import Path

from tqdm import tqdm


def check_lerobot_dataset(root: Path) -> bool:
    """Check parquet and video files. Returns True if all files exist."""
    root = Path(root)
    meta_dir = root / "meta"
    info_path = meta_dir / "info.json"

    if not info_path.is_file():
        print(f"ERROR: meta/info.json not found at {info_path}")
        return False

    with open(info_path) as f:
        info = json.load(f)

    data_path_fmt = info["data_path"]
    video_path_fmt = info.get("video_path")
    total_episodes = info["total_episodes"]
    chunks_size = info.get("chunks_size", 1000)

    # Get video keys from features (dtype == "video")
    video_keys = [
        key for key, ft in info.get("features", {}).items()
        if ft.get("dtype") == "video"
    ]

    missing_parquet = []
    missing_videos = []

    for ep_idx in tqdm(range(total_episodes), desc="Checking files", unit="ep"):
        ep_chunk = ep_idx // chunks_size

        # Check parquet
        parquet_rel = data_path_fmt.format(episode_chunk=ep_chunk, episode_index=ep_idx)
        parquet_path = root / parquet_rel
        if not parquet_path.is_file():
            missing_parquet.append(parquet_rel)

        # Check videos
        if video_path_fmt and video_keys:
            for vid_key in video_keys:
                video_rel = video_path_fmt.format(
                    episode_chunk=ep_chunk,
                    video_key=vid_key,
                    episode_index=ep_idx,
                )
                video_path = root / video_rel
                if not video_path.is_file():
                    missing_videos.append((ep_idx, vid_key, video_rel))

    # Report
    print(f"Dataset: {root}")
    print(f"Total episodes: {total_episodes}")
    print(f"Chunks size: {chunks_size}")
    print(f"Video keys: {video_keys}")
    print()

    all_ok = True

    if missing_parquet:
        all_ok = False
        print(f"MISSING PARQUET ({len(missing_parquet)} files):")
        for p in missing_parquet[:20]:
            print(f"  - {p}")
        if len(missing_parquet) > 20:
            print(f"  ... and {len(missing_parquet) - 20} more")
        print()

    if missing_videos:
        all_ok = False
        print(f"MISSING VIDEOS ({len(missing_videos)} files):")
        # Show first 20, grouped by type
        shown = 0
        for ep_idx, vid_key, rel in missing_videos:
            if shown >= 20:
                print(f"  ... and {len(missing_videos) - 20} more")
                break
            print(f"  - ep={ep_idx} {vid_key}: {rel}")
            shown += 1
        print()

    if all_ok:
        total_parquet = total_episodes
        total_videos = total_episodes * len(video_keys) if video_keys else 0
        print(f"OK: All {total_parquet} parquet files and {total_videos} video files present.")
    else:
        print(f"SUMMARY: {len(missing_parquet)} missing parquet, {len(missing_videos)} missing videos")

    return all_ok


if __name__ == "__main__":
    dataset_path = sys.argv[1] if len(sys.argv) > 1 else "/home/ss-oss1/data/dataset/egocentric/training_egocentric/retarget_lerobot/database_lerobot_00"
    ok = check_lerobot_dataset(dataset_path)
    sys.exit(0 if ok else 1)
