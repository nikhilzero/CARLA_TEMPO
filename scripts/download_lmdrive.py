"""
download_lmdrive.py — Selectively download LMDrive dataset from HuggingFace.

The LMDrive dataset (opendilab/lmdrive) is structured as:
  weather-{W}_data_town{TT}_{size}/
    routes_{N}/
      rgb_front/, rgb_left/, rgb_right/, rgb_rear/
      lidar/, measurements/, affordances/, ...

This script downloads only the towns and weather conditions you specify,
avoiding downloading the full multi-TB dataset.

Usage:
    python scripts/download_lmdrive.py --output-dir /scratch/nd967/CARLA_TEMPO/InterFuser/dataset

Edit TOWNS and WEATHERS below to control scope.
"""

import os
import sys
import argparse
from pathlib import Path

# =============================================================================
# CONFIGURE YOUR DOWNLOAD SCOPE HERE
# =============================================================================

# Towns to download (integers). Full dataset has Towns 01-07, 10.
# For a master's thesis, 4 train + 1 test is defensible.
# Start with train towns; add test town separately.
TOWNS = [1, 2, 3, 4, 5]   # Town01-04 = train, Town05 = held-out test

# Weather IDs to download. Full LMDrive has 14 daytime + 7 night conditions.
# Subset of 6 covers the main lighting/precipitation conditions.
WEATHERS = [1, 3, 6, 8, 14, 18]

# Route size type. "tiny" = short routes (faster download, good for thesis).
# Options: "tiny", "short", "long"
ROUTE_SIZE = "tiny"

# HuggingFace repo
HF_REPO = "opendilab/lmdrive"

# =============================================================================


def build_pattern_list(towns, weathers, size):
    """Build the list of folder patterns to download from the HF repo."""
    patterns = []
    for w in weathers:
        for t in towns:
            town_str = f"town{t:02d}"
            folder = f"weather-{w}_data_{town_str}_{size}"
            patterns.append(folder)
    return patterns


def check_disk_space(path, required_gb=50):
    """Warn if free space is below required_gb."""
    import shutil
    total, used, free = shutil.disk_usage(path)
    free_gb = free / (1024 ** 3)
    print(f"Free disk space at {path}: {free_gb:.1f} GB")
    if free_gb < required_gb:
        print(f"WARNING: Less than {required_gb} GB free. Download may fail.")
        print("Reduce TOWNS or WEATHERS scope, or free up space first.")
        sys.exit(1)
    return free_gb


def download_subset(output_dir, towns, weathers, size, repo):
    try:
        from huggingface_hub import snapshot_download, list_repo_files
    except ImportError:
        print("ERROR: huggingface_hub not installed.")
        print("Run: pip install huggingface_hub")
        sys.exit(1)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    patterns = build_pattern_list(towns, weathers, size)

    print(f"Downloading from HuggingFace repo: {repo}")
    print(f"Towns: {towns}")
    print(f"Weathers: {weathers}")
    print(f"Route size: {size}")
    print(f"Folder patterns ({len(patterns)} total):")
    for p in patterns:
        print(f"  {p}")
    print()

    # Estimate rough size: each tiny route ~500MB-2GB depending on content
    estimated_gb = len(patterns) * 1.5
    print(f"Rough size estimate: ~{estimated_gb:.0f} GB (varies by route count per folder)")
    print()

    check_disk_space("/scratch/nd967", required_gb=max(estimated_gb * 1.2, 20))

    # Download each folder separately so we can track progress
    # and resume partial downloads if the job gets interrupted.
    failed = []
    for i, pattern in enumerate(patterns):
        print(f"[{i+1}/{len(patterns)}] Downloading: {pattern}")
        try:
            snapshot_download(
                repo_id=repo,
                repo_type="dataset",
                local_dir=str(output_dir),
                allow_patterns=[f"{pattern}/*"],
                ignore_patterns=["*.git*"],
            )
            print(f"  -> Done: {pattern}")
        except Exception as e:
            print(f"  -> FAILED: {pattern} — {e}")
            failed.append(pattern)

    print()
    print("=== Download Summary ===")
    print(f"Requested: {len(patterns)} folders")
    print(f"Failed:    {len(failed)} folders")
    if failed:
        print("Failed folders:")
        for f in failed:
            print(f"  {f}")

    # Rebuild dataset_index.txt from what was actually downloaded
    rebuild_index(output_dir)


def rebuild_index(dataset_dir):
    """
    Regenerate dataset_index.txt by scanning the downloaded folders.
    Each line: <relative_path> <frame_count>
    This is what CarlaMVDetDataset reads to find routes.
    """
    dataset_dir = Path(dataset_dir)
    index_path = dataset_dir / "dataset_index.txt"

    lines = []
    route_dirs = sorted(dataset_dir.rglob("measurements"))
    for measurements_dir in route_dirs:
        route_dir = measurements_dir.parent
        # Count frames by counting measurement json files
        frame_count = len(list(measurements_dir.glob("*.json")))
        if frame_count == 0:
            # Some datasets use .npy for measurements
            frame_count = len(list(measurements_dir.glob("*.npy")))
        if frame_count > 0:
            rel_path = route_dir.relative_to(dataset_dir)
            lines.append(f"{rel_path} {frame_count}")

    with open(index_path, "w") as f:
        f.write("\n".join(lines) + "\n")

    print(f"Rebuilt dataset_index.txt: {len(lines)} routes found")
    print(f"  -> {index_path}")

    # Print summary
    total_frames = sum(int(l.split()[-1]) for l in lines)
    print(f"  -> Total frames: {total_frames:,}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--towns", type=int, nargs="+", default=TOWNS)
    parser.add_argument("--weathers", type=int, nargs="+", default=WEATHERS)
    parser.add_argument("--size", default=ROUTE_SIZE, choices=["tiny", "short", "long"])
    parser.add_argument("--repo", default=HF_REPO)
    args = parser.parse_args()

    download_subset(args.output_dir, args.towns, args.weathers, args.size, args.repo)


if __name__ == "__main__":
    main()
