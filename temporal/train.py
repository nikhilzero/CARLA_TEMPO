"""
temporal/train.py — Training entry point for InterFuserTemporal.

Usage:
    python -m temporal.train --config configs/temporal_t4.yaml [overrides...]
"""

import argparse
import os
import sys
import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from temporal.models.interfuser_temporal import build_interfuser_temporal
from temporal.data.temporal_dataset import TemporalWindowDataset, collate_temporal


def parse_args():
    p = argparse.ArgumentParser(description="Train InterFuserTemporal")
    p.add_argument("--config", required=True, help="Path to YAML config file")
    p.add_argument("--data-dir", required=True, help="Training data root")
    p.add_argument("--val-data-dir", default=None, help="Validation data root")
    p.add_argument("--output", default="output/temporal", help="Output directory")
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--num-workers", type=int, default=None)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--temporal-frames", type=int, default=None)
    p.add_argument("--frame-stride", type=int, default=None)
    p.add_argument("--pretrained-backbone", default=None, help="Path to baseline checkpoint")
    p.add_argument("--resume", default=None, help="Path to checkpoint to resume from")
    return p.parse_args()


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def apply_cli_overrides(cfg: dict, args) -> dict:
    """CLI args override YAML config values."""
    overrides = {
        ("training", "batch_size"): args.batch_size,
        ("training", "num_workers"): args.num_workers,
        ("training", "epochs"): args.epochs,
        ("training", "lr"): args.lr,
        ("model", "temporal_frames"): args.temporal_frames,
        ("model", "frame_stride"): args.frame_stride,
        ("model", "pretrained_backbone"): args.pretrained_backbone,
    }
    for (section, key), val in overrides.items():
        if val is not None:
            cfg.setdefault(section, {})[key] = val
    return cfg


def build_dataset(data_dir: str, cfg: dict, split: str = "train"):
    """
    Build the temporal dataset.
    TODO: replace the placeholder below with your actual LMDrive dataset class.
    """
    # Placeholder — swap in the real LMDrive dataset once the data loader is ready.
    raise NotImplementedError(
        "Implement build_dataset() with your LMDrive dataset class.\n"
        f"data_dir={data_dir}, split={split}"
    )


def train_one_epoch(model, loader, optimizer, device, epoch):
    model.train()
    total_loss = 0.0
    for step, batch in enumerate(loader):
        frames = [{k: v.to(device) for k, v in f.items()} for f in batch["frames"]]
        waypoints = batch["waypoints"].to(device)

        pred = model(frames)
        loss = nn.functional.l1_loss(pred, waypoints)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        if step % 100 == 0:
            print(f"  [epoch {epoch} step {step}] loss={loss.item():.4f}")

    return total_loss / max(len(loader), 1)


def main():
    args = parse_args()
    cfg = load_config(args.config)
    cfg = apply_cli_overrides(cfg, args)

    os.makedirs(args.output, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # ---- Model ----
    model_cfg = cfg["model"]
    model = build_interfuser_temporal(
        num_frames=model_cfg.get("temporal_frames", 4),
        fusion_strategy=model_cfg.get("fusion_strategy", "concat"),
        pretrained_path=model_cfg.get("pretrained_backbone", None),
    )
    model = model.to(device)

    train_cfg = cfg["training"]

    # ---- Optimizer ----
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_cfg.get("lr", 1e-4),
        weight_decay=train_cfg.get("weight_decay", 0.05),
    )

    # ---- Dataset ----
    # NOTE: build_dataset raises NotImplementedError until implemented.
    try:
        train_ds = build_dataset(args.data_dir, cfg, split="train")
        train_ds = TemporalWindowDataset(
            train_ds,
            num_frames=model_cfg.get("temporal_frames", 4),
            frame_stride=model_cfg.get("frame_stride", 1),
        )
        train_loader = DataLoader(
            train_ds,
            batch_size=train_cfg.get("batch_size", 8),
            shuffle=True,
            num_workers=train_cfg.get("num_workers", 4),
            collate_fn=collate_temporal,
            pin_memory=True,
        )
    except NotImplementedError as e:
        print(f"[WARN] Dataset not implemented yet:\n  {e}")
        print("  Skipping training loop. Implement build_dataset() first.")
        return

    # ---- Training loop ----
    epochs = train_cfg.get("epochs", 30)
    for epoch in range(1, epochs + 1):
        avg_loss = train_one_epoch(model, train_loader, optimizer, device, epoch)
        print(f"[Epoch {epoch}/{epochs}] avg_loss={avg_loss:.4f}")

        if epoch % cfg.get("logging", {}).get("save_interval", 5) == 0:
            ckpt_path = os.path.join(args.output, f"checkpoint_ep{epoch:03d}.pth")
            torch.save({"epoch": epoch, "model": model.state_dict(), "optimizer": optimizer.state_dict()}, ckpt_path)
            print(f"  Saved checkpoint: {ckpt_path}")


if __name__ == "__main__":
    main()
