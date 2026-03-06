"""
temporal/eval.py — Unified evaluation script for InterFuserTemporal.

Computes waypoint L1 (matching baseline eval_l1_error) and waypoint L2,
so temporal and baseline checkpoints can be compared on equal terms.

Usage:
    # Evaluate temporal model
    python -m temporal.eval \\
        --data-dir /scratch/nd967/CARLA_TEMPO/InterFuser/dataset \\
        --checkpoint /scratch/nd967/CARLA_TEMPO/runs/temporal/.../model_best.pth.tar \\
        --temporal-frames 4 --frame-stride 1 --temporal-depth 2 \\
        --val-towns 1 --val-weathers 18

    # Output: JSON file next to checkpoint + stdout table

Metrics reported:
    waypoint_l1         : unweighted mean absolute error (same units as baseline eval_l1_error)
    waypoint_l2         : unweighted RMSE (Euclidean distance per step)
    weighted_waypoint_l1: distance-weighted L1 (exact baseline loss, for audit)
    loss_traffic        : MVTL1Loss (detection)
    loss_waypoints      : WaypointL1Loss (training loss)
    loss_junction       : CrossEntropyLoss
    loss_traffic_light  : CrossEntropyLoss
    loss_stop_sign      : CrossEntropyLoss
    loss_total          : weighted sum (same weights as training)
"""

import argparse
import json
import os
import sys
import math

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# ---- Ensure InterFuser package is importable ----------------------------------------
_INTERFUSER_PKG = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "InterFuser", "interfuser")
)
if _INTERFUSER_PKG not in sys.path:
    sys.path.insert(0, _INTERFUSER_PKG)

from timm.data import create_carla_dataset, create_carla_loader
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

from temporal.models.interfuser_temporal import build_interfuser_temporal
from temporal.data.temporal_dataset import (
    TemporalWindowDataset,
    collate_temporal,
    episode_lengths_from_carla_dataset,
)
from temporal.utils.losses import build_loss_fns, WaypointL1Loss


# -------------------------------------------------------------------------------------
# Argument parsing
# -------------------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Evaluate InterFuserTemporal checkpoint")

    # Data
    p.add_argument("--data-dir", required=True)
    p.add_argument("--val-towns", type=int, nargs="+", default=[1])
    p.add_argument("--val-weathers", type=int, nargs="+", default=[18])

    # Model
    p.add_argument("--checkpoint", required=True, help="Path to model_best.pth.tar")
    p.add_argument("--temporal-frames", type=int, default=4)
    p.add_argument("--frame-stride", type=int, default=1)
    p.add_argument("--temporal-depth", type=int, default=2)

    # Eval
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--output-json", default=None,
                   help="Where to write JSON results (default: next to checkpoint)")

    return p.parse_args()


# -------------------------------------------------------------------------------------
# Dataset builder (same as train.py)
# -------------------------------------------------------------------------------------

def build_val_dataset(data_dir, towns, weathers):
    base_ds = create_carla_dataset(
        "carla",
        root=data_dir,
        towns=towns,
        weathers=weathers,
        with_lidar=True,
        multi_view=True,
        augment_prob=0.0,
    )
    if len(base_ds) == 0:
        raise RuntimeError(
            f"Val dataset empty for towns={towns}, weathers={weathers} at {data_dir}"
        )
    create_carla_loader(
        base_ds,
        input_size=[3, 224, 224],
        batch_size=1,
        multi_view_input_size=[3, 128, 128],
        is_training=False,
        scale=[1.0, 1.0],
        color_jitter=0.0,
        mean=IMAGENET_DEFAULT_MEAN,
        std=IMAGENET_DEFAULT_STD,
        num_workers=1,
        persistent_workers=False,
    )
    return base_ds


# -------------------------------------------------------------------------------------
# Waypoint metrics (raw L1 and L2, without step-weighting)
# -------------------------------------------------------------------------------------

def waypoint_l1_l2(pred_wp, target_wp):
    """
    Compute mean absolute error and RMSE for waypoints.

    Args:
        pred_wp:   (B, N_steps, 2) tensor
        target_wp: (B, N_steps, 2) tensor
    Returns:
        (mean_l1, mean_l2): scalars (Python floats)
    """
    invalid = target_wp.ge(1000)          # mark padding entries
    pred_wp  = pred_wp.clone()
    target_wp = target_wp.clone()
    pred_wp[invalid]   = 0.0
    target_wp[invalid] = 0.0

    # L1: mean absolute error per coordinate, then average
    l1_per_coord = (pred_wp - target_wp).abs()               # (B, N, 2)
    l1_per_coord[invalid] = 0.0

    valid_count = (~invalid).sum().clamp(min=1)
    mean_l1 = l1_per_coord.sum() / valid_count

    # L2: Euclidean distance per step, then average over valid steps
    diff = pred_wp - target_wp                                # (B, N, 2)
    dist = diff.pow(2).sum(dim=-1).sqrt()                     # (B, N)
    invalid_steps = invalid.any(dim=-1)                       # (B, N)
    dist[invalid_steps] = 0.0
    valid_steps = (~invalid_steps).sum().clamp(min=1)
    mean_l2 = dist.sum() / valid_steps

    return mean_l1.item(), mean_l2.item()


# -------------------------------------------------------------------------------------
# Main evaluation loop
# -------------------------------------------------------------------------------------

@torch.no_grad()
def evaluate(model, loader, loss_fns, device):
    model.eval()

    # Accumulators
    total = dict(
        waypoint_l1=0.0,
        waypoint_l2=0.0,
        weighted_waypoint_l1=0.0,
        loss_traffic=0.0,
        loss_waypoints=0.0,
        loss_junction=0.0,
        loss_traffic_light=0.0,
        loss_stop_sign=0.0,
        loss_total=0.0,
    )
    steps = 0

    for inputs, target in loader:
        # Move to device
        inputs = [
            {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in f.items()}
            for f in inputs
        ]
        target = [x.to(device) if isinstance(x, torch.Tensor) else x for x in target]

        output = model(inputs)
        # output: (traffic, waypoints, is_junction, traffic_light_state, stop_sign, traffic_feature)

        # --- Raw waypoint metrics ---
        pred_wp   = output[1]        # (B, N_steps, 2)
        target_wp = target[1]        # (B, N_steps, 2)
        l1, l2 = waypoint_l1_l2(pred_wp, target_wp)
        total["waypoint_l1"] += l1
        total["waypoint_l2"] += l2

        # --- Training losses (same as train.py) ---
        loss_traffic, _ = loss_fns["traffic"](output[0], target[4])
        loss_wp         = loss_fns["waypoints"](output[1], target[1])
        loss_junc       = loss_fns["cls"](output[2], target[2])
        loss_tl         = loss_fns["cls"](output[3], target[3])
        loss_stop       = loss_fns["stop_cls"](output[4], target[6])

        loss_total = (
            loss_traffic * 0.5
            + loss_wp    * 0.2
            + loss_junc  * 0.05
            + loss_tl    * 0.1
            + loss_stop  * 0.01
        )

        def _item(x):
            return x.item() if isinstance(x, torch.Tensor) else float(x)

        total["weighted_waypoint_l1"] += _item(loss_wp)
        total["loss_traffic"]         += _item(loss_traffic)
        total["loss_waypoints"]       += _item(loss_wp)
        total["loss_junction"]        += _item(loss_junc)
        total["loss_traffic_light"]   += _item(loss_tl)
        total["loss_stop_sign"]       += _item(loss_stop)
        total["loss_total"]           += _item(loss_total)

        steps += 1

    # Average across steps
    results = {k: v / max(steps, 1) for k, v in total.items()}
    results["steps"] = steps
    return results


# -------------------------------------------------------------------------------------
# Main
# -------------------------------------------------------------------------------------

def main():
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Checkpoint: {args.checkpoint}")

    # ---- Model ----
    model = build_interfuser_temporal(
        num_frames=args.temporal_frames,
        temporal_encoder_depth=args.temporal_depth,
        pretrained_path=None,     # load full temporal checkpoint below
    )

    ckpt = torch.load(args.checkpoint, map_location="cpu")
    # Checkpoint may be a full training checkpoint or just state_dict
    state_dict = ckpt.get("model", ckpt.get("state_dict", ckpt))
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"  [WARN] Missing keys: {len(missing)} — {missing[:3]}")
    if unexpected:
        print(f"  [WARN] Unexpected keys: {len(unexpected)} — {unexpected[:3]}")

    model = model.to(device)
    model.eval()
    print(f"Model loaded. Params: {sum(p.numel() for p in model.parameters()):,}")

    # ---- Dataset ----
    print(f"Loading val dataset: towns={args.val_towns}, weathers={args.val_weathers}")
    val_base = build_val_dataset(args.data_dir, args.val_towns, args.val_weathers)
    ep_lengths = episode_lengths_from_carla_dataset(val_base)
    print(f"  Routes: {len(ep_lengths)}, total frames: {sum(ep_lengths)}")

    val_ds = TemporalWindowDataset(
        val_base,
        num_frames=args.temporal_frames,
        frame_stride=args.frame_stride,
        episode_lengths=ep_lengths,
    )
    print(f"  Temporal windows: {len(val_ds)}")

    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        collate_fn=collate_temporal,
        pin_memory=(device.type == "cuda"),
        drop_last=False,
    )

    # ---- Loss functions ----
    loss_fns = build_loss_fns()

    # ---- Evaluate ----
    print("\nRunning evaluation...")
    results = evaluate(model, val_loader, loss_fns, device)

    # ---- Report ----
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    print(f"  Checkpoint:           {args.checkpoint}")
    print(f"  T={args.temporal_frames}, stride={args.frame_stride}, depth={args.temporal_depth}")
    print(f"  Steps evaluated:      {results['steps']}")
    print()
    print("  -- Waypoint Metrics (comparable to baseline) --")
    print(f"  waypoint_l1:          {results['waypoint_l1']:.6f}  (raw mean |pred - target|, m)")
    print(f"  waypoint_l2:          {results['waypoint_l2']:.6f}  (Euclidean dist per step, m)")
    print(f"  weighted_waypoint_l1: {results['weighted_waypoint_l1']:.6f}  (distance-weighted, = WaypointL1Loss)")
    print()
    print("  -- Individual Head Losses --")
    print(f"  loss_traffic:         {results['loss_traffic']:.6f}")
    print(f"  loss_waypoints:       {results['loss_waypoints']:.6f}")
    print(f"  loss_junction:        {results['loss_junction']:.6f}")
    print(f"  loss_traffic_light:   {results['loss_traffic_light']:.6f}")
    print(f"  loss_stop_sign:       {results['loss_stop_sign']:.6f}")
    print()
    print(f"  loss_total (5-head):  {results['loss_total']:.6f}")
    print("=" * 60)

    # ---- Save JSON ----
    if args.output_json is None:
        args.output_json = os.path.join(
            os.path.dirname(args.checkpoint), "eval_results.json"
        )
    results["checkpoint"] = args.checkpoint
    results["temporal_frames"] = args.temporal_frames
    results["frame_stride"] = args.frame_stride
    results["temporal_depth"] = args.temporal_depth
    results["val_towns"] = args.val_towns
    results["val_weathers"] = args.val_weathers

    with open(args.output_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {args.output_json}")


if __name__ == "__main__":
    main()
