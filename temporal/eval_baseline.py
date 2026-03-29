"""
temporal/eval_baseline.py — Unified evaluation for the baseline InterFuser checkpoint.

Runs the same waypoint L1 / L2 metrics as temporal/eval.py so both models
can be compared on identical numbers.

Uses T=1 TemporalWindowDataset (single frame) + the same collate_temporal,
then calls the baseline model's forward_features + encoder + decoder manually
(same code path that InterFuserTemporal uses internally).

Usage:
    python -m temporal.eval_baseline \\
        --data-dir /scratch/nd967/CARLA_TEMPO/InterFuser/dataset \\
        --checkpoint /scratch/nd967/interfuser_project/InterFuser/interfuser/output/\\
20260210-110341-interfuser_baseline-224-real_data_test/model_best.pth.tar \\
        --val-towns 1 --val-weathers 18
"""

import argparse
import json
import os
import sys

import torch
from torch.utils.data import DataLoader

# ---- Ensure InterFuser package is importable ----------------------------------------
_INTERFUSER_PKG = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "InterFuser", "interfuser")
)
if _INTERFUSER_PKG not in sys.path:
    sys.path.insert(0, _INTERFUSER_PKG)

import timm
from timm.data import create_carla_dataset, create_carla_loader
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

from temporal.data.temporal_dataset import (
    TemporalWindowDataset,
    collate_temporal,
    episode_lengths_from_carla_dataset,
)
from temporal.utils.losses import build_loss_fns


# -------------------------------------------------------------------------------------
# Args
# -------------------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Evaluate baseline InterFuser checkpoint")
    p.add_argument("--data-dir", required=True)
    p.add_argument("--val-towns", type=int, nargs="+", default=[1])
    p.add_argument("--val-weathers", type=int, nargs="+", default=[18])
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--output-json", default=None)
    return p.parse_args()


# -------------------------------------------------------------------------------------
# Dataset
# -------------------------------------------------------------------------------------

def build_val_dataset(data_dir, towns, weathers):
    base_ds = create_carla_dataset(
        "carla", root=data_dir, towns=towns, weathers=weathers,
        with_lidar=True, multi_view=True, augment_prob=0.0,
    )
    if len(base_ds) == 0:
        raise RuntimeError(f"Val dataset empty for towns={towns}, weathers={weathers}")
    create_carla_loader(
        base_ds,
        input_size=[3, 224, 224], batch_size=1,
        multi_view_input_size=[3, 128, 128], is_training=False,
        scale=[1.0, 1.0], color_jitter=0.0,
        mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD,
        num_workers=1, persistent_workers=False,
    )
    return base_ds


# -------------------------------------------------------------------------------------
# Single-frame forward (same path as InterFuserTemporal, T=1)
# -------------------------------------------------------------------------------------

def baseline_forward(base, x):
    """
    Run the baseline model on a single frame dict.
    Mirrors InterFuserTemporal.forward() for T=1, without temporal fusion.

    Args:
        base: interfuser_baseline model
        x:    data dict with tensors on device
    Returns:
        (traffic, waypoints, is_junction, traffic_light_state, stop_sign, traffic_feature)
    """
    front_image = x["rgb"]
    measurements = x["measurements"]
    target_point = x["target_point"]
    bs = front_image.shape[0]

    # Feature extraction
    memory = base.forward_features(
        x["rgb"], x["rgb_left"], x["rgb_right"], x["rgb_center"],
        x["lidar"], x["measurements"],
    )

    # Spatial encoder (same as temporal model, step 5)
    memory = base.encoder(memory, mask=base.attn_mask)

    # Decoder (same as temporal model, step 6)
    if base.end2end:
        tgt = base.query_pos_embed.repeat(bs, 1, 1)
    else:
        tgt = base.position_encoding(
            torch.ones((bs, 1, 20, 20), device=front_image.device)
        )
        tgt = tgt.flatten(2)
        tgt = torch.cat([tgt, base.query_pos_embed.repeat(bs, 1, 1)], 2)
    tgt = tgt.permute(2, 0, 1)

    hs = base.decoder(base.query_embed.repeat(1, bs, 1), memory, query_pos=tgt)[0]
    hs = hs.permute(1, 0, 2)   # (B, N_queries, D)

    if base.end2end:
        return base.waypoints_generator(hs, target_point)

    if base.waypoints_pred_head != "heatmap":
        traffic_feature         = hs[:, :400]
        is_junction_feature     = hs[:, 400]
        traffic_light_feature   = hs[:, 400]
        stop_sign_feature       = hs[:, 400]
        waypoints_feature       = hs[:, 401:411]
    else:
        traffic_feature         = hs[:, :400]
        is_junction_feature     = hs[:, 400]
        traffic_light_feature   = hs[:, 400]
        stop_sign_feature       = hs[:, 400]
        waypoints_feature       = hs[:, 401:405]

    if base.waypoints_pred_head == "heatmap":
        waypoints = base.waypoints_generator(waypoints_feature, measurements)
    elif base.waypoints_pred_head == "gru":
        waypoints = base.waypoints_generator(waypoints_feature, target_point)
    elif base.waypoints_pred_head == "gru-command":
        waypoints = base.waypoints_generator(waypoints_feature, target_point, measurements)
    else:
        waypoints = base.waypoints_generator(waypoints_feature, measurements)

    is_junction         = base.junction_pred_head(is_junction_feature)
    traffic_light_state = base.traffic_light_pred_head(traffic_light_feature)
    stop_sign           = base.stop_sign_head(stop_sign_feature)

    velocity = measurements[:, 6:7].unsqueeze(-1).repeat(1, 400, 32)
    traffic_feature_with_vel = torch.cat([traffic_feature, velocity], dim=2)
    traffic = base.traffic_pred_head(traffic_feature_with_vel)

    return traffic, waypoints, is_junction, traffic_light_state, stop_sign, traffic_feature


# -------------------------------------------------------------------------------------
# Waypoint metrics (identical to eval.py)
# -------------------------------------------------------------------------------------

def waypoint_l1_l2(pred_wp, target_wp):
    invalid   = target_wp.ge(1000)
    pred_wp   = pred_wp.clone()
    target_wp = target_wp.clone()
    pred_wp[invalid]   = 0.0
    target_wp[invalid] = 0.0

    l1_per_coord = (pred_wp - target_wp).abs()
    l1_per_coord[invalid] = 0.0
    valid_count = (~invalid).sum().clamp(min=1)
    mean_l1 = l1_per_coord.sum() / valid_count

    diff = pred_wp - target_wp
    dist = diff.pow(2).sum(dim=-1).sqrt()
    invalid_steps = invalid.any(dim=-1)
    dist[invalid_steps] = 0.0
    valid_steps = (~invalid_steps).sum().clamp(min=1)
    mean_l2 = dist.sum() / valid_steps

    return mean_l1.item(), mean_l2.item()


# -------------------------------------------------------------------------------------
# Evaluation loop
# -------------------------------------------------------------------------------------

@torch.no_grad()
def evaluate(base, loader, loss_fns, device):
    base.eval()

    total = dict(
        waypoint_l1=0.0, waypoint_l2=0.0, weighted_waypoint_l1=0.0,
        loss_traffic=0.0, loss_waypoints=0.0, loss_junction=0.0,
        loss_traffic_light=0.0, loss_stop_sign=0.0, loss_total=0.0,
    )
    steps = 0

    for inputs, target in loader:
        # inputs is list of 1 dict (T=1 window)
        x = {k: v.to(device) if isinstance(v, torch.Tensor) else v
             for k, v in inputs[0].items()}
        target = [t.to(device) if isinstance(t, torch.Tensor) else t for t in target]

        output = baseline_forward(base, x)

        l1, l2 = waypoint_l1_l2(output[1], target[1])
        total["waypoint_l1"] += l1
        total["waypoint_l2"] += l2

        loss_traffic, _ = loss_fns["traffic"](output[0], target[4])
        loss_wp         = loss_fns["waypoints"](output[1], target[1])
        loss_junc       = loss_fns["cls"](output[2], target[2])
        loss_tl         = loss_fns["cls"](output[3], target[3])
        loss_stop       = loss_fns["stop_cls"](output[4], target[6])

        loss_total = (
            loss_traffic * 0.5 + loss_wp * 0.2 + loss_junc * 0.05
            + loss_tl * 0.1 + loss_stop * 0.01
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

    out = {k: v / max(steps, 1) for k, v in total.items()}
    out["steps"] = steps
    return out


# -------------------------------------------------------------------------------------
# Main
# -------------------------------------------------------------------------------------

def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device:     {device}")
    print(f"Checkpoint: {args.checkpoint}")

    # ---- Load baseline model ----
    base = timm.create_model("interfuser_baseline", pretrained=False)
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    state_dict = ckpt.get("state_dict", ckpt.get("model", ckpt))
    missing, unexpected = base.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"  [WARN] Missing keys: {len(missing)} — {missing[:3]}")
    if unexpected:
        print(f"  [WARN] Unexpected keys: {len(unexpected)} — {unexpected[:3]}")
    base = base.to(device)
    base.eval()
    print(f"Baseline params: {sum(p.numel() for p in base.parameters()):,}")

    # ---- Dataset (T=1 temporal window = single-frame) ----
    print(f"Loading val dataset: towns={args.val_towns}, weathers={args.val_weathers}")
    val_base = build_val_dataset(args.data_dir, args.val_towns, args.val_weathers)
    ep_lengths = episode_lengths_from_carla_dataset(val_base)
    print(f"  Routes: {len(ep_lengths)}, frames: {sum(ep_lengths)}")

    val_ds = TemporalWindowDataset(
        val_base, num_frames=1, frame_stride=1, episode_lengths=ep_lengths,
    )
    print(f"  Windows (T=1): {len(val_ds)}")

    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, collate_fn=collate_temporal,
        pin_memory=(device.type == "cuda"), drop_last=False,
    )

    # ---- Evaluate ----
    loss_fns = build_loss_fns()
    print("\nRunning evaluation...")
    results = evaluate(base, val_loader, loss_fns, device)

    # ---- Report ----
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS — BASELINE InterFuser")
    print("=" * 60)
    print(f"  Checkpoint:           {args.checkpoint}")
    print(f"  Steps evaluated:      {results['steps']}")
    print()
    print("  -- Waypoint Metrics --")
    print(f"  waypoint_l1:          {results['waypoint_l1']:.6f}  (m)")
    print(f"  waypoint_l2:          {results['waypoint_l2']:.6f}  (m)")
    print(f"  weighted_waypoint_l1: {results['weighted_waypoint_l1']:.6f}")
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
            os.path.dirname(args.checkpoint), "eval_baseline_results.json"
        )
    results.update({
        "checkpoint": args.checkpoint,
        "model": "interfuser_baseline",
        "val_towns": args.val_towns,
        "val_weathers": args.val_weathers,
    })
    with open(args.output_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {args.output_json}")


if __name__ == "__main__":
    main()