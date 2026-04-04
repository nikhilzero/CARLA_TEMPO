"""
temporal/train.py — Training entry point for InterFuserTemporal.

Mirrors the baseline InterFuser training as closely as possible:
  - Same CarlaMVDetDataset / create_carla_loader pipeline
  - Same 5-head loss (traffic, waypoints, junction, traffic_light, stop_sign)
  - Same AdamW + cosine schedule hyperparameters by default

Usage (from repo root):
    python -m temporal.train --data-dir /path/to/dataset [options]

SLURM:
    sbatch scripts/slurm/temporal_smoke.sbatch
    sbatch scripts/slurm/temporal_train.sbatch
"""

import argparse
import csv
import os
import sys
import math
from datetime import datetime

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
from temporal.utils.losses import build_loss_fns


# -------------------------------------------------------------------------------------
# Argument parsing
# -------------------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Train InterFuserTemporal")

    # Data
    p.add_argument("--data-dir", required=True, help="Dataset root (contains dataset_index.txt)")
    p.add_argument("--train-towns", type=int, nargs="+", default=[1])
    p.add_argument("--train-weathers", type=int, nargs="+", default=[18])
    p.add_argument("--val-towns", type=int, nargs="+", default=[1])
    p.add_argument("--val-weathers", type=int, nargs="+", default=[18])

    # Model
    p.add_argument("--model-type", choices=["concat", "crossattn"], default="concat", help="Temporal fusion type")
    p.add_argument("--dropout", type=float, default=0.1, help="Dropout for temporal encoder")
    p.add_argument("--temporal-frames", type=int, default=4, help="Window size T")
    p.add_argument("--frame-stride", type=int, default=1)
    p.add_argument("--temporal-depth", type=int, default=2, help="Temporal encoder layers")
    p.add_argument("--pretrained-backbone", default=None, help="Path to baseline checkpoint")

    # Training
    p.add_argument("--epochs", type=int, default=12)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--lr", type=float, default=0.0005)
    p.add_argument("--backbone-lr", type=float, default=0.0002)
    p.add_argument("--weight-decay", type=float, default=0.05)
    p.add_argument("--clip-grad", type=float, default=10.0)
    p.add_argument("--warmup-epochs", type=int, default=1)
    p.add_argument("--grad-accum", type=int, default=1, help="Gradient accumulation steps")
    p.add_argument("--workers", type=int, default=4)

    # Output
    p.add_argument("--output", default="/scratch/nd967/CARLA_TEMPO/runs/temporal",
                   help="Base output dir; a timestamped subdir is created inside")
    p.add_argument("--experiment", default="temporal_t4")
    p.add_argument("--log-interval", type=int, default=10)
    p.add_argument("--save-interval", type=int, default=1, help="Save checkpoint every N epochs")

    # Resume
    p.add_argument("--resume", default=None, help="Path to checkpoint to resume from")

    return p.parse_args()


# -------------------------------------------------------------------------------------
# Dataset
# -------------------------------------------------------------------------------------

def build_carla_dataset(data_dir, towns, weathers, is_training):
    """
    Create and configure a CarlaMVDetDataset with the same settings as the baseline.
    Transforms are applied to base_ds as a side effect of create_carla_loader.
    Returns the configured base dataset (NOT yet wrapped in TemporalWindowDataset).
    """
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
            f"Dataset is empty for towns={towns}, weathers={weathers} at {data_dir}. "
            "Check dataset_index.txt."
        )

    # Call create_carla_loader to set rgb_transform / multi_view_transform / etc.
    # on base_ds as instance attributes. The returned loader is discarded — we
    # build our own temporal loader below.
    create_carla_loader(
        base_ds,
        input_size=[3, 224, 224],
        batch_size=1,                         # placeholder, loader is discarded
        multi_view_input_size=[3, 128, 128],
        is_training=is_training,
        scale=[0.9, 1.1] if is_training else [1.0, 1.0],
        color_jitter=0.1 if is_training else 0.0,
        mean=IMAGENET_DEFAULT_MEAN,
        std=IMAGENET_DEFAULT_STD,
        num_workers=1,
        persistent_workers=False,
    )
    return base_ds


# -------------------------------------------------------------------------------------
# Training / validation
# -------------------------------------------------------------------------------------

def move_inputs_to_device(inputs, device):
    """Move list-of-T-dicts to device."""
    return [
        {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in f.items()}
        for f in inputs
    ]


def train_one_epoch(model, loader, optimizer, scheduler, loss_fns, device,
                    epoch, grad_accum, log_interval, output_dir):
    model.train()
    optimizer.zero_grad()

    total_loss = 0.0
    total_loss_traffic = 0.0
    total_loss_waypoints = 0.0
    steps = 0

    for step, (inputs, target) in enumerate(loader):
        inputs = move_inputs_to_device(inputs, device)
        target = [x.to(device) if isinstance(x, torch.Tensor) else x for x in target]

        # Forward
        output = model(inputs)
        # output: (traffic, waypoints, is_junction, traffic_light_state, stop_sign, traffic_feature)

        loss_traffic, _loss_vel = loss_fns["traffic"](output[0], target[4])
        loss_waypoints          = loss_fns["waypoints"](output[1], target[1])
        loss_junction           = loss_fns["cls"](output[2], target[2])
        loss_traffic_light      = loss_fns["cls"](output[3], target[3])
        loss_stop_sign          = loss_fns["stop_cls"](output[4], target[6])

        loss = (
            loss_traffic       * 0.5
            + loss_waypoints   * 0.2
            + loss_junction    * 0.05
            + loss_traffic_light * 0.1
            + loss_stop_sign   * 0.01
        )

        (loss / grad_accum).backward()

        if (step + 1) % grad_accum == 0 or (step + 1) == len(loader):
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            optimizer.step()
            optimizer.zero_grad()

        total_loss          += loss.item()
        total_loss_traffic  += loss_traffic.item() if isinstance(loss_traffic, torch.Tensor) else float(loss_traffic)
        total_loss_waypoints += loss_waypoints.item()
        steps += 1

        if step % log_interval == 0:
            print(
                f"  [epoch {epoch} step {step}/{len(loader)}] "
                f"loss={loss.item():.4f}  "
                f"traffic={loss_traffic.item() if isinstance(loss_traffic, torch.Tensor) else loss_traffic:.4f}  "
                f"waypoints={loss_waypoints.item():.4f}"
            )

    if scheduler is not None:
        scheduler.step()

    return total_loss / max(steps, 1)


@torch.no_grad()
def validate(model, loader, loss_fns, device):
    model.eval()
    total_loss = 0.0
    steps = 0

    for inputs, target in loader:
        inputs = move_inputs_to_device(inputs, device)
        target = [x.to(device) if isinstance(x, torch.Tensor) else x for x in target]

        output = model(inputs)

        loss_traffic, _ = loss_fns["traffic"](output[0], target[4])
        loss_waypoints  = loss_fns["waypoints"](output[1], target[1])
        loss_junction   = loss_fns["cls"](output[2], target[2])
        loss_tl         = loss_fns["cls"](output[3], target[3])
        loss_stop       = loss_fns["stop_cls"](output[4], target[6])

        loss = (
            loss_traffic   * 0.5
            + loss_waypoints * 0.2
            + loss_junction  * 0.05
            + loss_tl        * 0.1
            + loss_stop      * 0.01
        )
        total_loss += loss.item()
        steps += 1

    return total_loss / max(steps, 1)


# -------------------------------------------------------------------------------------
# Cosine LR scheduler (mirrors baseline)
# -------------------------------------------------------------------------------------

def build_cosine_scheduler(optimizer, epochs, warmup_epochs, steps_per_epoch):
    """Simple cosine annealing with linear warmup, computed per-epoch."""
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / max(warmup_epochs, 1)
        progress = (epoch - warmup_epochs) / max(epochs - warmup_epochs, 1)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# -------------------------------------------------------------------------------------
# Main
# -------------------------------------------------------------------------------------

def main():
    args = parse_args()

    # ---- Output directory ----
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    exp_name  = f"{timestamp}-{args.experiment}-T{args.temporal_frames}"
    output_dir = os.path.join(args.output, exp_name)
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output dir: {output_dir}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ---- Model ----
    if args.model_type == "crossattn":
        from temporal.models.interfuser_temporal_attn import build_interfuser_temporal_crossattn
        model = build_interfuser_temporal_crossattn(
            num_frames=args.temporal_frames,
            num_attn_layers=args.temporal_depth,
            pretrained_path=args.pretrained_backbone,
            dropout=args.dropout,
        )
    else:
        model = build_interfuser_temporal(
            num_frames=args.temporal_frames,
            temporal_encoder_depth=args.temporal_depth,
            pretrained_path=args.pretrained_backbone,
            dropout=args.dropout,
        )
    model = model.to(device)

    # ---- Optimizer: lower LR for pretrained backbone, full LR for new params ----
    base_ids = {id(p) for p in model.base.parameters()}
    temporal_only = [p for p in model.parameters() if id(p) not in base_ids]

    optimizer = torch.optim.AdamW(
        [
            {"params": list(model.base.parameters()), "lr": args.backbone_lr},
            {"params": temporal_only, "lr": args.lr},
        ],
        weight_decay=args.weight_decay,
        eps=1e-8,
    )

    # ---- Loss functions (identical to baseline) ----
    loss_fns = build_loss_fns()

    # ---- Datasets ----
    print("Building train dataset...")
    train_base = build_carla_dataset(
        args.data_dir, args.train_towns, args.train_weathers, is_training=True
    )
    ep_lengths_train = episode_lengths_from_carla_dataset(train_base)
    print(f"  Train routes: {len(ep_lengths_train)}, frames: {sum(ep_lengths_train)}")

    train_ds = TemporalWindowDataset(
        train_base,
        num_frames=args.temporal_frames,
        frame_stride=args.frame_stride,
        episode_lengths=ep_lengths_train,
    )
    print(f"  Temporal windows (train): {len(train_ds)}")

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        collate_fn=collate_temporal,
        pin_memory=True,
        drop_last=True,
    )

    print("Building val dataset...")
    val_base = build_carla_dataset(
        args.data_dir, args.val_towns, args.val_weathers, is_training=False
    )
    ep_lengths_val = episode_lengths_from_carla_dataset(val_base)
    print(f"  Val routes: {len(ep_lengths_val)}, frames: {sum(ep_lengths_val)}")

    val_ds = TemporalWindowDataset(
        val_base,
        num_frames=args.temporal_frames,
        frame_stride=args.frame_stride,
        episode_lengths=ep_lengths_val,
    )
    print(f"  Temporal windows (val): {len(val_ds)}")

    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        collate_fn=collate_temporal,
        pin_memory=True,
        drop_last=False,
    )

    # ---- LR scheduler ----
    scheduler = build_cosine_scheduler(
        optimizer,
        epochs=args.epochs,
        warmup_epochs=args.warmup_epochs,
        steps_per_epoch=len(train_loader),
    )

    # ---- Resume ----
    start_epoch = 1
    best_val_loss = float("inf")
    if args.resume and os.path.isfile(args.resume):
        ckpt = torch.load(args.resume, map_location="cpu")
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch = ckpt.get("epoch", 0) + 1
        best_val_loss = ckpt.get("best_val_loss", float("inf"))
        print(f"Resumed from {args.resume} at epoch {start_epoch}")

    # ---- CSV summary (mirrors baseline summary.csv) ----
    summary_path = os.path.join(output_dir, "summary.csv")
    with open(summary_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "eval_loss"])

    # ---- Training loop ----
    for epoch in range(start_epoch, args.epochs + 1):
        print(f"\n[Epoch {epoch}/{args.epochs}]")

        train_loss = train_one_epoch(
            model, train_loader, optimizer, scheduler, loss_fns, device,
            epoch, args.grad_accum, args.log_interval, output_dir,
        )
        val_loss = validate(model, val_loader, loss_fns, device)

        print(f"  => train_loss={train_loss:.4f}  val_loss={val_loss:.4f}")

        # Append to CSV
        with open(summary_path, "a", newline="") as f:
            csv.writer(f).writerow([epoch, train_loss, val_loss])

        # Save checkpoint
        is_best = val_loss < best_val_loss
        best_val_loss = min(val_loss, best_val_loss)

        ckpt = {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "train_loss": train_loss,
            "val_loss": val_loss,
            "best_val_loss": best_val_loss,
            "args": vars(args),
        }

        last_path = os.path.join(output_dir, "last.pth.tar")
        torch.save(ckpt, last_path)

        if is_best:
            best_path = os.path.join(output_dir, "model_best.pth.tar")
            torch.save(ckpt, best_path)
            print(f"  ** New best val_loss={best_val_loss:.4f} — saved model_best.pth.tar")

        if epoch % args.save_interval == 0:
            ep_path = os.path.join(output_dir, f"checkpoint-{epoch}.pth.tar")
            torch.save(ckpt, ep_path)

    print(f"\nTraining complete. Best val_loss={best_val_loss:.4f}")
    print(f"Outputs: {output_dir}")


if __name__ == "__main__":
    main()
