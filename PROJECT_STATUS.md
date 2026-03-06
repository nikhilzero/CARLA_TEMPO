# PROJECT_STATUS.md
# CARLA_TEMPO — Current Project State

_Last updated: 2026-03-06_

---

## 1. Project Overview

**Goal**: Extend InterFuser (autonomous driving transformer) with temporal reasoning.
Instead of processing a single camera snapshot per timestep, the temporal model receives a
window of T=4 consecutive frames and learns motion, velocity, and intent from the sequence.

**Thesis**: Master's at Rutgers University–Camden.
**HPC**: Rutgers Amarel cluster (`nd967@amarel.hpc.rutgers.edu`).
**Dataset**: LMDrive-style CARLA recordings (Town01, Weather 18, 210 frames) at
`/scratch/nd967/CARLA_TEMPO/InterFuser/dataset/`.

---

## 2. Repository Structure

```
CARLA_TEMPO/
├── CLAUDE.md                        # Project context for Claude Code
├── RUNBOOK.md                       # Operational runbook (commands reference)
├── README.md                        # Quick-start guide
├── requirements.txt                 # pyyaml, einops
├── .gitignore
│
├── InterFuser/                      # Upstream baseline — DO NOT MODIFY
│   └── interfuser/
│       ├── train.py                 # Baseline training entry point
│       └── timm/
│           ├── models/interfuser.py # Core InterFuser architecture
│           └── data/carla_dataset.py + carla_loader.py + ...
│
├── temporal/                        # OUR temporal extension (main work area)
│   ├── train.py                     # Temporal training entry point  ✅ WORKING
│   ├── models/interfuser_temporal.py  # InterFuserTemporal class     ✅ WORKING
│   ├── data/temporal_dataset.py    # TemporalWindowDataset + collate ✅ WORKING
│   └── utils/losses.py             # 5-head loss functions           ✅ WORKING
│
├── configs/
│   └── temporal_t4.yaml            # Config (NOT YET WIRED into train.py) ⚠️ PARTIAL
│
└── scripts/
    ├── setup_amarel.sh             # One-time Amarel env setup        ✅ DONE
    ├── train_baseline.sh           # Legacy — superseded by slurm/   ⚠️ OLD
    ├── train_temporal.sh           # Legacy — superseded by slurm/   ⚠️ OLD
    └── slurm/
        ├── baseline_smoke.sbatch  # ✅ TESTED, PASSING
        ├── baseline_train.sbatch  # ✅ TESTED, COMPLETED (job 50020281)
        ├── temporal_smoke.sbatch  # ✅ TESTED, PASSING (job 50020296)
        └── temporal_train.sbatch  # ✅ TESTED, COMPLETED (job 50020302)
```

---

## 3. System Architecture

### 3.1 Baseline InterFuser (unchanged)

```
Multi-view RGB (front + left + right + rear) + LiDAR
    ↓
Camera CNN backbones (independent per view)
    ↓
Token projection → patch tokens per view
    ↓
Multi-view token fusion (Transformer Encoder)
    ↓
Decoder → 5 prediction heads:
  [0] traffic detection (multi-view L1)
  [1] waypoints (L1 per step)
  [2] is_junction (binary CrossEntropy)
  [3] traffic_light_state (CrossEntropy)
  [4] stop_sign (CrossEntropy)
```

### 3.2 InterFuserTemporal (Approach 1 — Concat Fusion)

```
T=4 consecutive frames, each as a full data dict
    ↓
For each frame t ∈ {0,1,2,3}:
    base.forward_features(frame_t) → (N, B, D)
    + temporal_pos_embed[t]          # learnable (T, 1, 1, D)
    ↓
Concatenate across T: (4N, B, D)
    ↓
Temporal Encoder (2-layer Transformer, 8 heads) — cross-frame attention
    ↓
Temporal Pooling (learned weighted average across T) → (N, B, D)
    ↓
Base Encoder + Decoder (original InterFuser) → 5 heads (unchanged)
```

**Parameter count**: ~54M base + ~1.58M temporal = ~55.58M total (2.9% new params).

---

## 4. Experiment Results

### 4.1 Experiments Run on Amarel

| Job ID | Script | Status | Wall time | Epochs | Result |
|---|---|---|---|---|---|
| 50020270 | baseline_smoke | COMPLETED | ~5m | 11* | train converges |
| 50020281 | baseline_train | COMPLETED | 6m 57s | 22* | eval_l1_error=0.0145 |
| 50020282 | temporal_smoke | FAILED | ~2m | 0 | Bug: numpy→tensor in collate |
| 50020294 | temporal_smoke | FAILED | ~2m | 0 | Bug: scalar targets in collate |
| 50020296 | temporal_smoke | COMPLETED | ~6m | 11* | 103 steps, val_loss=0.1216 |
| 50020302 | temporal_train | COMPLETED | 13m 17s | 12 | best val_loss=0.0961 |

*timm scheduler has `cooldown_epochs=10` by default — baseline smoke and baseline_train ran extra epochs.

### 4.2 Best Checkpoint Results

**Baseline:**
- Run dir: `runs/baseline/20260301-101807-interfuser_baseline-224-baseline_full/`
- Metric: `eval_l1_error` (waypoint L1, meters — lower is better)
- **Best: 0.014530** at epoch 20

**Temporal T=4 (Concat Fusion):**
- Run dir: `runs/temporal/20260301-102924-temporal_full-T4/`
- Metric: `eval_loss` (composite 5-head weighted loss — lower is better)
- **Best: 0.096125** at epoch 12 (still improving — monotonically decreasing)

⚠️ **Metrics are not directly comparable.** Baseline uses timm's `eval_l1_error` (waypoint-only L1).
Temporal uses the composite 5-head `eval_loss`. Fair comparison requires the CARLA leaderboard
harness or a unified eval script.

### 4.3 Results Package (Amarel)

```
/scratch/nd967/CARLA_TEMPO/results_pack/          # 2.3 GB
/scratch/nd967/CARLA_TEMPO/results_pack.tar.gz    # archived
```

Contains: model_best.pth.tar + last.pth.tar + summary.csv + SLURM logs for both runs.

---

## 5. What Is Working ✅

| Component | Status | Notes |
|---|---|---|
| InterFuser baseline training | ✅ Working | Runs via SLURM, achieves eval_l1_error=0.0145 |
| TemporalWindowDataset | ✅ Working | Episode-aware, no boundary crossing |
| collate_temporal | ✅ Working | Handles tensor/numpy/scalar types correctly |
| InterFuserTemporal model | ✅ Working | Forward pass verified end-to-end |
| 5-head loss (temporal) | ✅ Working | Matches baseline loss weighting |
| temporal/train.py | ✅ Working | Full training loop, checkpointing, CSV log |
| Differential LR (backbone vs temporal) | ✅ Working | 2e-4 / 5e-4 |
| Gradient accumulation | ✅ Working | grad_accum=2 → effective batch=4 |
| Cosine LR schedule + warmup | ✅ Working | 1-epoch linear warmup |
| SLURM scripts | ✅ Working | smoke + full for both baseline and temporal |
| Pretrained backbone loading | ✅ Working | Loads from 12-epoch baseline checkpoint |
| results_pack / packaging | ✅ Done | Archived at results_pack.tar.gz |

---

## 6. What Is Partially Implemented ⚠️

| Component | Status | Gap |
|---|---|---|
| `configs/temporal_t4.yaml` | ⚠️ Exists but unused | `train.py` uses argparse; YAML not wired in |
| Evaluation metric | ⚠️ Inconsistent | Temporal uses composite loss; baseline uses timm eval |
| Multi-town/weather training | ⚠️ Not tested | Config lists towns 1–4,6,7,10 but scripts only use town1 |
| Mixed precision training | ⚠️ Not implemented | Config mentions it, train.py does not use AMP |
| TensorBoard logging | ⚠️ Not implemented | Only CSV logging currently |
| Temporal Approach 2 (attn) | ⚠️ Not started | Architecture planned but not coded |

---

## 7. What Is Not Yet Done ❌

| Component | Priority | Notes |
|---|---|---|
| Unified eval script (waypoint L1 + L2) | HIGH | Needed for fair baseline vs temporal comparison |
| CARLA closed-loop evaluation | HIGH | RouteCompletion, InfractionsPerKm on CARLA server |
| Temporal ablations (T=2, T=8) | MEDIUM | Needed for thesis |
| Frame stride ablations (stride=2) | MEDIUM | Different temporal context lengths |
| Temporal Approach 2 (temporal attn) | MEDIUM | More expressive fusion strategy |
| Multi-town training (generalization) | MEDIUM | Currently only Town01 / Weather 18 |
| YAML config integration in train.py | LOW | Quality-of-life improvement |
| AMP (mixed precision) training | LOW | Speedup but not required for correctness |

---

## 8. Known Issues / Gotchas

1. **cooldown_epochs=10**: timm's LR scheduler default adds 10 cooldown epochs. `--epochs 12`
   actually runs 22 total. This affects baseline, not temporal (which uses a custom scheduler).

2. **Metric mismatch**: Cannot directly compare `eval_l1_error` (baseline) vs `eval_loss`
   (temporal). A unified evaluation pass is required.

3. **Local git 5 commits ahead of origin**: Need to push. (Mac has no GitHub SSH key;
   push must be done from Amarel.)

4. **Dataset scope**: Only 210 frames (Town01, Weather 18) used. Full LMDrive dataset
   has many more routes and weathers. Training on small dataset limits generalization.

5. **Temporal model still improving at epoch 12**: val_loss monotonically decreasing at
   final epoch — longer training (24+ epochs) may yield better results.

6. **Configs/temporal_t4.yaml discrepancy**: Config specifies batch_size=8, epochs=30,
   multi-town training — none of these match the current sbatch configuration.

---

## 9. Git History Summary

```
6f3e5ae  fix: collate_temporal numpy→tensor (scalar targets)
0ff316d  fix: collate_temporal numpy→tensor (numpy arrays)
2f50669  docs: add RUNBOOK.md and .gitignore
eaa306a  temporal: implement dataset wrapper, full 5-head loss, working train loop
cf78662  baseline scripts: add slurm smoke+train with correct data path and output dir
ea98410  fix: corrected temporal model wrapping real InterFuser architecture
e9eea4f  project structure: InterFuser baseline + temporal extension scaffold
33b5ad6  initial commit
```

---

## 10. Next Priority Actions

1. **Push local commits to GitHub** (via Amarel)
2. **Write unified eval script** that computes waypoint L1/L2 for both models
3. **Run temporal model longer** (24 epochs) — still converging at epoch 12
4. **Run ablations**: T=2, T=8, stride=2
5. **Implement Approach 2** (temporal cross-attention between frames)
6. **Set up CARLA closed-loop evaluation** for definitive comparison
