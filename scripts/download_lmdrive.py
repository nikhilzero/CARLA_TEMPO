"""
download_lmdrive.py — Download and prepare LMDrive dataset from HuggingFace.

Repo:  OpenDILabCommunity/LMDrive
Structure in repo:  data/Town{NN}/routes_town{NN}_{size}_w{W}_{timestamp}.tar.gz

Each tar.gz extracts to a route directory containing:
  rgb_full/   — 800×2400 jpg (4 views stacked: front / left / right / rear)
  lidar/      — .npy point clouds
  measurements/ — .json per frame
  affordances/, actors_data/, birdview/, 3d_bbs/, ...

This script:
  1. Lists available tar.gz files for target towns/weathers/size
  2. Downloads up to MAX_ROUTES_PER_COMBO per (town, weather)
  3. Extracts each tar.gz
  4. Splits rgb_full (800×2400) → rgb_front / rgb_left / rgb_right / rgb_rear
  5. Rebuilds dataset_index.txt

Usage:
    python scripts/download_lmdrive.py --output-dir /scratch/nd967/CARLA_TEMPO/InterFuser/dataset

Edit TOWNS, WEATHERS, ROUTE_SIZE, MAX_ROUTES_PER_COMBO below to control scope.
"""

import os
import sys
import tarfile
import argparse
import shutil
from pathlib import Path

# =============================================================================
# CONFIGURE DOWNLOAD SCOPE HERE
# =============================================================================

TOWNS = [1, 2, 3, 4, 5]          # Town01-04 = train, Town05 = held-out test
WEATHERS = [1, 3, 6, 8, 14, 18]  # 6 conditions: daytime, rain, fog, overcast
ROUTE_SIZE = "tiny"               # "tiny" | "short" | "long"

# Max routes (tar.gz files) to download per (town, weather) combination.
# 10 routes × 30 combos = 300 routes.  Each ~51 MB compressed, ~200 MB extracted.
# Total estimate: ~300 × 200 MB = ~60 GB extracted.
MAX_ROUTES_PER_COMBO = 10

HF_REPO = "OpenDILabCommunity/LMDrive"

# rgb_full is 800×2400: 4 views stacked vertically, each 800×600
RGB_CROP = {
    "rgb_front": (0,    0,   800,  600),
    "rgb_left":  (0,  600,   800, 1200),
    "rgb_right": (0, 1200,   800, 1800),
    "rgb_rear":  (0, 1800,   800, 2400),
}

# =============================================================================


def check_disk_space(path, required_gb=30):
    total, used, free = shutil.disk_usage(path)
    free_gb = free / (1024 ** 3)
    print(f"Free disk space at {path}: {free_gb:.1f} GB")
    if free_gb < required_gb:
        print(f"ERROR: Less than {required_gb} GB free. Aborting.")
        sys.exit(1)
    return free_gb


def list_target_files(towns, weathers, size):
    """Return dict: {(town, weather): [hf_file_path, ...]} for all target combos."""
    try:
        from huggingface_hub import list_repo_tree
    except ImportError:
        print("ERROR: huggingface_hub not installed. Run: pip install huggingface_hub")
        sys.exit(1)

    result = {}
    for t in towns:
        town_str = f"Town{t:02d}"
        print(f"  Listing {town_str}...", flush=True)
        items = list(list_repo_tree(HF_REPO, repo_type="dataset",
                                    path_in_repo=f"data/{town_str}", recursive=False))
        all_files = [getattr(i, "path", str(i)) for i in items]
        for w in weathers:
            pattern = f"_{size}_w{w}_"
            matches = sorted([f for f in all_files if pattern in f])
            result[(t, w)] = matches
            print(f"    Town{t:02d} w{w}: {len(matches)} available, "
                  f"will download {min(len(matches), MAX_ROUTES_PER_COMBO)}")
    return result


def download_file(hf_path, output_dir):
    """Download a single file from HF repo to output_dir, return local path."""
    from huggingface_hub import hf_hub_download
    local = hf_hub_download(
        repo_id=HF_REPO,
        repo_type="dataset",
        filename=hf_path,
        local_dir=str(output_dir),
    )
    return Path(local)


def extract_and_convert(tar_path, output_dir):
    """
    Extract tar.gz into output_dir, then split rgb_full → separate view dirs.
    Returns the route directory path, or None on failure.
    """
    from PIL import Image
    import numpy as np

    output_dir = Path(output_dir)

    # Extract
    try:
        with tarfile.open(tar_path, "r:gz") as tf:
            tf.extractall(output_dir)
    except Exception as e:
        print(f"    ERROR extracting {tar_path.name}: {e}")
        return None

    # Find the extracted route dir (should be a single top-level dir)
    # tar may contain e.g. routes_town01_tiny_w18_08_26_10_06_21/
    route_name = tar_path.stem  # strip .tar.gz → routes_town01_tiny_w18_...
    if route_name.endswith(".tar"):
        route_name = route_name[:-4]  # handle .tar.gz double extension
    route_dir = output_dir / route_name

    if not route_dir.exists():
        # Try to find any newly extracted dir
        candidates = [d for d in output_dir.iterdir()
                      if d.is_dir() and "routes_" in d.name and not d.name.startswith("weather-")]
        if not candidates:
            print(f"    WARNING: Could not find extracted dir for {tar_path.name}")
            return None
        route_dir = candidates[-1]

    # Rename to weather-{W}_data_town{NN}_{timestamp} so carla_dataset.py regex matches:
    #   pattern = re.compile('weather-(\d+).*town(\d\d)')
    import re as _re
    src_m = _re.match(r"routes_town(\d+)_tiny_w(\d+)_(.+)", route_dir.name)
    if src_m:
        town_n, weather_n, ts = src_m.group(1), src_m.group(2), src_m.group(3)
        canonical = output_dir / f"weather-{weather_n}_data_town{int(town_n):02d}_{ts}"
        if not canonical.exists():
            route_dir.rename(canonical)
        route_dir = canonical

    # Split rgb_full → separate view dirs
    rgb_full_dir = route_dir / "rgb_full"
    if not rgb_full_dir.exists():
        # Already split or no rgb_full — skip conversion
        return route_dir

    # Create output view dirs
    for view in RGB_CROP:
        (route_dir / view).mkdir(exist_ok=True)

    frame_files = sorted(rgb_full_dir.glob("*.jpg"))
    if not frame_files:
        frame_files = sorted(rgb_full_dir.glob("*.png"))

    for frame_path in frame_files:
        try:
            img = Image.open(frame_path)
            for view, box in RGB_CROP.items():
                cropped = img.crop(box)
                cropped.save(route_dir / view / frame_path.name, quality=95)
        except Exception as e:
            print(f"    WARNING: Failed to split frame {frame_path.name}: {e}")
            continue

    # Remove rgb_full dir to save space
    shutil.rmtree(rgb_full_dir, ignore_errors=True)

    return route_dir


def rebuild_index(dataset_dir):
    """Scan dataset_dir for route dirs with measurements/, write dataset_index.txt."""
    dataset_dir = Path(dataset_dir)
    index_path = dataset_dir / "dataset_index.txt"

    lines = []
    # Look for any dir that has measurements/ and rgb_front/
    for measurements_dir in sorted(dataset_dir.rglob("measurements")):
        route_dir = measurements_dir.parent
        if not (route_dir / "rgb_front").exists():
            continue  # not yet converted or incomplete
        frame_count = len(list(measurements_dir.glob("*.json")))
        if frame_count == 0:
            frame_count = len(list(measurements_dir.glob("*.npy")))
        if frame_count > 0:
            rel_path = route_dir.relative_to(dataset_dir)
            lines.append(f"{rel_path} {frame_count}")

    with open(index_path, "w") as f:
        f.write("\n".join(lines) + "\n")

    total_frames = sum(int(l.split()[-1]) for l in lines)
    print(f"\nRebuilt dataset_index.txt:")
    print(f"  Routes: {len(lines)}")
    print(f"  Total frames: {total_frames:,}")
    print(f"  Path: {index_path}")
    return len(lines), total_frames


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--towns", type=int, nargs="+", default=TOWNS)
    parser.add_argument("--weathers", type=int, nargs="+", default=WEATHERS)
    parser.add_argument("--size", default=ROUTE_SIZE, choices=["tiny", "short", "long"])
    parser.add_argument("--max-routes", type=int, default=MAX_ROUTES_PER_COMBO,
                        help="Max routes to download per (town, weather) combo")
    parser.add_argument("--index-only", action="store_true",
                        help="Skip download, just rebuild dataset_index.txt")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.index_only:
        rebuild_index(output_dir)
        return

    print("=" * 60)
    print("LMDrive Dataset Download")
    print(f"  Repo:      {HF_REPO}")
    print(f"  Towns:     {args.towns}")
    print(f"  Weathers:  {args.weathers}")
    print(f"  Size:      {args.size}")
    print(f"  Max routes per (town,weather): {args.max_routes}")
    print(f"  Output:    {output_dir}")
    print("=" * 60)

    check_disk_space(str(output_dir.parent), required_gb=30)

    print("\nListing available files from HuggingFace...")
    target_files = list_target_files(args.towns, args.weathers, args.size)

    total_to_download = sum(min(len(v), args.max_routes) for v in target_files.values())
    print(f"\nTotal files to download: {total_to_download}")
    estimated_gb = total_to_download * 0.05  # ~51 MB per tar.gz
    print(f"Estimated compressed size: ~{estimated_gb:.1f} GB")
    print(f"Estimated extracted size:  ~{estimated_gb * 4:.1f} GB (4× compressed)")

    # Temp dir for tar.gz files (will be cleaned after extraction)
    tar_tmp_dir = output_dir / "_tarballs"
    tar_tmp_dir.mkdir(exist_ok=True)

    downloaded = 0
    failed = []

    for (t, w), files in sorted(target_files.items()):
        subset = files[:args.max_routes]
        for hf_path in subset:
            fname = Path(hf_path).name
            route_name = fname.replace(".tar.gz", "")
            route_out = output_dir / route_name

            if route_out.exists() and (route_out / "rgb_front").exists():
                print(f"  [SKIP — already extracted] {fname}")
                downloaded += 1
                continue

            downloaded += 1
            print(f"  [{downloaded}/{total_to_download}] {fname}", flush=True)

            try:
                local_tar = download_file(hf_path, tar_tmp_dir)
                route_dir = extract_and_convert(local_tar, output_dir)
                if route_dir:
                    print(f"    -> OK: {route_dir.name}")
                    local_tar.unlink(missing_ok=True)  # free space after extraction
                else:
                    failed.append(hf_path)
            except Exception as e:
                print(f"    -> FAILED: {e}")
                failed.append(hf_path)

    # Clean up temp dir
    shutil.rmtree(tar_tmp_dir, ignore_errors=True)

    print("\n" + "=" * 60)
    print(f"Download complete. Failed: {len(failed)}/{total_to_download}")
    if failed:
        print("Failed files:")
        for f in failed:
            print(f"  {f}")

    n_routes, n_frames = rebuild_index(output_dir)
    print(f"\nDone. {n_routes} routes, {n_frames:,} frames available for training.")
    print("=" * 60)


if __name__ == "__main__":
    main()
