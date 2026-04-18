"""
diagnose_d01_collapse.py
========================
Diagnoses why dropout=0.1 (d01) temporal models produce DS=0.0 in CARLA.

Runs inference on real dataset samples and prints the raw model outputs
(waypoints, junction prob, traffic_light prob, stop_sign prob) for a d01
checkpoint vs a d03 checkpoint side-by-side.

Also simulates the InterfuserController decision to show EXACTLY why
the vehicle never moves.

Usage (on Amarel, from repo root):
    source ~/miniconda3/bin/activate interfuser
    cd /scratch/nd967/CARLA_TEMPO
    python scripts/diagnose_d01_collapse.py

Edit D01_CKPT / D03_CKPT / DATA_DIR below if paths change.
"""

import os
import sys
import numpy as np
import torch

# ── Paths ─────────────────────────────────────────────────────────────────────

# NOTE: Two possible d01 run directories (check which exists on Amarel)
_D01_CANDIDATES = [
    "/scratch/nd967/CARLA_TEMPO/runs/temporal/20260307-195110-temporal_research_T4_s5-T4/model_best.pth.tar",
    "/scratch/nd967/CARLA_TEMPO/runs/temporal/20260308-124902-temporal_research_T4_s5-T4/model_best.pth.tar",
]
D01_CKPT = next((p for p in _D01_CANDIDATES if os.path.exists(p)), _D01_CANDIDATES[0])
D03_CKPT = "/scratch/nd967/CARLA_TEMPO/runs/temporal/20260329-132750-temporal_reg_research_T4_s5-T4/model_best.pth.tar"
DATA_DIR  = "/scratch/nd967/CARLA_TEMPO/InterFuser/dataset"

NUM_FRAMES   = 4
FRAME_STRIDE = 5
TEMPORAL_DEPTH = 2
N_SAMPLES = 20    # how many windows to probe

# ── Path setup ─────────────────────────────────────────────────────────────────

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
INTERFUSER_PKG = os.path.join(REPO_ROOT, "InterFuser", "interfuser")
for p in [REPO_ROOT, INTERFUSER_PKG]:
    if p not in sys.path:
        sys.path.insert(0, p)

from timm.data import create_carla_dataset, create_carla_loader
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from temporal.models.interfuser_temporal import build_interfuser_temporal
from temporal.data.temporal_dataset import (
    TemporalWindowDataset,
    collate_temporal,
    episode_lengths_from_carla_dataset,
)


# ── Controller helpers (reproduced from interfuser_controller.py for standalone diagnosis) ──

def downsample_waypoints(waypoints, precision=0.2):
    out = [np.array([0.0, 0.0])]
    last = np.array([0.0, 0.0])
    for i in range(len(waypoints)):
        wp = np.array(waypoints[i])
        dis = np.linalg.norm(wp - last)
        if dis > precision:
            interval = int(dis / precision)
            mv = (wp - last) / (interval + 1)
            for j in range(interval):
                out.append(last + mv * (j + 1))
        out.append(wp)
        last = wp
    return out


def simulate_controller(waypoints_np, junction_prob, tl_prob, stop_prob, speed=0.0):
    """
    Simplified controller decision (no CARLA maps — traffic meta assumed clear).
    Returns: (brake: bool, desired_speed: float, reason: str)
    """
    # Waypoint-based safe distance
    dsampled = downsample_waypoints(waypoints_np)
    # Without CARLA map, traffic meta is zero → get_max_safe_distance returns norm of last point
    # Simulate: d_0 ≈ np.linalg.norm(dsampled[-1])
    d_0 = np.linalg.norm(dsampled[-1]) if len(dsampled) > 0 else 0.0
    d_0_adj = max(0.0, d_0 - 2.0)  # subtract collision buffer

    if d_0_adj < max(3.0, speed):
        return True, 0.0, f"waypoint_d0={d_0:.3f} < threshold={max(3.0, speed):.1f}"

    # Traffic light check (only when in junction)
    if junction_prob > 0.0 and tl_prob > 0.3:
        return True, 0.0, f"red_light: junction_prob={junction_prob:.3f}, tl_prob={tl_prob:.3f}"

    # Stop sign check
    if stop_prob < 0.6:
        return True, 0.0, f"stop_sign: stop_prob={stop_prob:.3f} < 0.6"

    desired_speed = min(5.0, 4 * max(0, d_0_adj - 2.0))
    return False, desired_speed, "OK"


# ── Load model ────────────────────────────────────────────────────────────────

def load_model(ckpt_path, label):
    print(f"\nLoading {label} from:\n  {ckpt_path}")
    model = build_interfuser_temporal(
        num_frames=NUM_FRAMES,
        temporal_encoder_depth=TEMPORAL_DEPTH,
        pretrained_path=None,
        dropout=0.1,  # architecture is the same; dropout only matters at training time
    )
    ckpt = torch.load(ckpt_path, map_location="cpu")
    sd = ckpt.get("state_dict", ckpt.get("model", ckpt))
    missing, unexpected = model.load_state_dict(sd, strict=False)
    print(f"  Missing={len(missing)}, Unexpected={len(unexpected)}")
    model.cpu()
    model.eval()
    return model


# ── Dataset ───────────────────────────────────────────────────────────────────

def build_dataset():
    print(f"\nBuilding dataset from: {DATA_DIR}")
    # Mirrors temporal/train.py exactly
    base_ds = create_carla_dataset(
        "carla",
        root=DATA_DIR,
        towns=[5],
        weathers=[18],
        with_lidar=True,
        multi_view=True,
        augment_prob=0.0,
    )
    print(f"  Base dataset size: {len(base_ds)}")
    # create_carla_loader sets rgb_transform / multi_view_transform on base_ds
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
    ep_lens = episode_lengths_from_carla_dataset(base_ds)
    print(f"  Episodes: {len(ep_lens)}, Frames: {sum(ep_lens)}")
    ds = TemporalWindowDataset(base_ds, num_frames=NUM_FRAMES, frame_stride=FRAME_STRIDE,
                                episode_lengths=ep_lens)
    print(f"  Valid windows: {len(ds)}")
    return ds


# ── Main ──────────────────────────────────────────────────────────────────────

def run_inference(model, ds, n_samples, label):
    softmax = torch.nn.Softmax(dim=1)
    results = []

    # Sample evenly across the dataset
    indices = np.linspace(0, len(ds) - 1, n_samples, dtype=int)

    with torch.no_grad():
        for idx in indices:
            frames, _ = ds[idx]
            # Add batch dimension (CPU inference — runs on login node)
            window = []
            for frame in frames:
                f = {}
                for k, v in frame.items():
                    if isinstance(v, torch.Tensor):
                        f[k] = v.unsqueeze(0).float()
                    elif isinstance(v, np.ndarray):
                        f[k] = torch.from_numpy(v).unsqueeze(0).float()
                    else:
                        f[k] = v
                window.append(f)

            traffic, waypoints, is_junction, tl_state, stop_sign, _ = model(window)

            wp_np = waypoints.numpy()[0]                  # (10, 2)
            junc  = softmax(is_junction).numpy().reshape(-1)[0]
            tl    = softmax(tl_state).numpy().reshape(-1)[0]
            stop  = softmax(stop_sign).numpy().reshape(-1)[0]

            brake, desired_speed, reason = simulate_controller(wp_np, junc, tl, stop)

            results.append({
                "idx": idx,
                "wp_mean_x": float(wp_np[:, 0].mean()),
                "wp_mean_y": float(wp_np[:, 1].mean()),
                "wp_max_x":  float(wp_np[:, 0].max()),
                "wp_norm":   float(np.linalg.norm(wp_np[-1])),  # last waypoint distance
                "junction":  float(junc),
                "tl_state":  float(tl),
                "stop_sign": float(stop),
                "brake":     brake,
                "desired_speed": desired_speed,
                "reason":    reason,
            })

    return results


def print_results(results, label):
    print(f"\n{'='*70}")
    print(f"  {label}  ({len(results)} samples)")
    print(f"{'='*70}")
    print(f"{'idx':>5} {'wp_mean_x':>10} {'wp_max_x':>9} {'wp_norm':>8} "
          f"{'junc':>6} {'tl':>6} {'stop':>6} {'brake':>6} {'reason'}")
    print(f"{'-'*70}")

    brake_count = 0
    wp_reasons, tl_reasons, other_reasons = 0, 0, 0

    for r in results:
        brake_str = "BRAKE" if r["brake"] else "GO"
        if r["brake"]:
            brake_count += 1
            if "waypoint" in r["reason"]:
                wp_reasons += 1
            elif "red_light" in r["reason"]:
                tl_reasons += 1
            else:
                other_reasons += 1

        print(f"{r['idx']:>5} {r['wp_mean_x']:>10.4f} {r['wp_max_x']:>9.4f} "
              f"{r['wp_norm']:>8.4f} {r['junction']:>6.3f} {r['tl_state']:>6.3f} "
              f"{r['stop_sign']:>6.3f} {brake_str:>6} {r['reason'][:50]}")

    print(f"\nSummary for [{label}]:")
    print(f"  Braking {brake_count}/{len(results)} samples ({100*brake_count/len(results):.0f}%)")
    if brake_count > 0:
        print(f"  Brake causes — waypoint_d0: {wp_reasons}, red_light: {tl_reasons}, other: {other_reasons}")
    wp_mean_xs = [r["wp_mean_x"] for r in results]
    wp_norms   = [r["wp_norm"]   for r in results]
    print(f"  Waypoint mean_x:  avg={np.mean(wp_mean_xs):.4f}  std={np.std(wp_mean_xs):.4f}  "
          f"min={np.min(wp_mean_xs):.4f}  max={np.max(wp_mean_xs):.4f}")
    print(f"  Last-wp norm:     avg={np.mean(wp_norms):.4f}   std={np.std(wp_norms):.4f}   "
          f"min={np.min(wp_norms):.4f}   max={np.max(wp_norms):.4f}")
    print(f"  junction avg: {np.mean([r['junction'] for r in results]):.3f}")
    print(f"  tl_state avg: {np.mean([r['tl_state'] for r in results]):.3f}")
    print(f"  stop_sign avg: {np.mean([r['stop_sign'] for r in results]):.3f}")


def main():
    print("=" * 70)
    print("  d01 Collapse Diagnostic")
    print("=" * 70)

    ds = build_dataset()

    model_d01 = load_model(D01_CKPT, "d01 (RES-002, dropout=0.1)")
    model_d03 = load_model(D03_CKPT, "d03 (RES-004, dropout=0.3)")

    print(f"\nRunning inference on {N_SAMPLES} samples from Town05 weather=18...")

    results_d01 = run_inference(model_d01, ds, N_SAMPLES, "d01")
    results_d03 = run_inference(model_d03, ds, N_SAMPLES, "d03")

    print_results(results_d01, "d01 — RES-002 (DS=0.000 in CARLA)")
    print_results(results_d03, "d03 — RES-004 (DS=8.157 in CARLA)")

    # ── Focused comparison ──────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print("  HEAD-TO-HEAD COMPARISON (same samples)")
    print(f"{'='*70}")
    print(f"{'metric':<25} {'d01':>10} {'d03':>10} {'diff':>10}")
    print("-" * 55)

    metrics = [
        ("wp_mean_x (avg)",   "wp_mean_x"),
        ("wp_norm / last wp", "wp_norm"),
        ("junction prob",     "junction"),
        ("tl_state prob",     "tl_state"),
        ("stop_sign prob",    "stop_sign"),
    ]
    for name, key in metrics:
        v01 = np.mean([r[key] for r in results_d01])
        v03 = np.mean([r[key] for r in results_d03])
        print(f"  {name:<23} {v01:>10.4f} {v03:>10.4f} {v03-v01:>+10.4f}")

    brk01 = sum(r["brake"] for r in results_d01) / len(results_d01)
    brk03 = sum(r["brake"] for r in results_d03) / len(results_d03)
    print(f"  {'brake rate':<23} {brk01:>10.1%} {brk03:>10.1%}")

    # ── Simulate d01 with tl_calibration_threshold=0.55 patch ─────────────────
    TL_PATCH = 0.55
    print(f"\n{'='*70}")
    print(f"  PATCHED SIMULATION: d01 with tl_calibration_threshold={TL_PATCH}")
    print(f"  (clamp tl_state to 0 if < {TL_PATCH} — mirrors temporal_d01_patched_config.py)")
    print(f"{'='*70}")
    patched_brake = 0
    for r in results_d01:
        tl_patched = 0.0 if r["tl_state"] < TL_PATCH else r["tl_state"]
        b, _, reason = simulate_controller(
            np.array([[r["wp_mean_x"], r["wp_mean_y"]]] * 10),  # approximate
            r["junction"], tl_patched, r["stop_sign"]
        )
        if b:
            patched_brake += 1
    print(f"  Patched brake rate: {patched_brake}/{len(results_d01)} "
          f"({100*patched_brake/len(results_d01):.0f}%)")
    print(f"  Original brake rate: {sum(r['brake'] for r in results_d01)}/{len(results_d01)} (100%)")
    print(f"\n  → If patched brake rate ≈ 0%, the vehicle WOULD drive with this fix.")
    print(f"  → CARLA eval submitted as carla_eval_d01_patched_gpu.sbatch to confirm.")

    print("\nDone.")


if __name__ == "__main__":
    main()
