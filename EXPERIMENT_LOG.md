# EXPERIMENT_LOG.md
# CARLA_TEMPO — Training Runs, Results, and Observations

_Last updated: 2026-03-06_

---

## Naming Convention

`EXP-NNN` — sequential experiment number
Format per entry: Job ID, config, result, observations, next steps

---

## EXP-000 — Baseline Pretrain (pre-CARLA_TEMPO repo)

**Date**: 2026-02-10
**Location**: `/scratch/nd967/interfuser_project/`
**Purpose**: Establish baseline performance before temporal work
**Config**:
```
Model:     interfuser_baseline
Dataset:   LMDrive (Town01, Weather18), 210 frames
Epochs:    12
LR:        5e-4 (backbone 2e-4)
Batch:     4
Schedule:  cosine, warmup 1 epoch
```
**Result**: `eval_l1_error = 0.0265`
**Checkpoint**: `/scratch/nd967/interfuser_project/InterFuser/interfuser/output/20260210-110341-interfuser_baseline-224-real_data_test/model_best.pth.tar` (636 MB)
**Notes**:
- This checkpoint is used as pretrained backbone for all temporal experiments
- eval_l1_error is waypoint-only L1 from timm harness (lower = better)

---

## EXP-001 — Baseline Smoke Test

**Date**: 2026-03-01
**Job ID**: 50020270
**Script**: `scripts/slurm/baseline_smoke.sbatch`
**Purpose**: Sanity check baseline env + data on CARLA_TEMPO repo
**Config**:
```
Model:     interfuser_baseline
Dataset:   Town01, Weather18, 210 frames
Epochs:    1 (specified) → ~11 actual (cooldown_epochs=10)
Batch:     4
```
**Result**: COMPLETED — train loss decreasing, checkpoint saved
**Wall time**: ~5 min 35 sec
**Output dir**: `runs/baseline/<timestamp>-baseline_smoke/`
**Notes**:
- `cooldown_epochs=10` in timm LR scheduler default causes `--epochs 1` to run ~11 epochs
- Not a bug per se; expected timm behavior. Use `--cooldown-epochs 0` to suppress in future.
- Confirmed data pipeline works end-to-end

---

## EXP-002 — Baseline Full Training

**Date**: 2026-03-01
**Job ID**: 50020281
**Script**: `scripts/slurm/baseline_train.sbatch`
**Purpose**: Reproduce baseline performance in CARLA_TEMPO repo (for fair comparison)
**Config**:
```
Model:     interfuser_baseline
Dataset:   Town01, Weather18, 210 frames
Epochs:    12 specified → 22 actual (cooldown_epochs=10)
LR:        5e-4 (backbone 2e-4)
Batch:     4
Schedule:  cosine, warmup 1 epoch
AdamW:     weight_decay=0.05, eps=1e-8
Clip grad: 10.0
```
**Results**:
| Epoch | train_loss | eval_loss | eval_l1_error |
|---|---|---|---|
| 1 | ~0.35 | — | — |
| ... | decreasing | — | — |
| 20 | — | — | **0.014530** ← best |
| 21 | — | — | 0.016369 |
| 22 | — | — | — |

**Best checkpoint**: epoch 20, eval_l1_error=0.014530
**Wall time**: 6m 57s
**Output dir**: `runs/baseline/20260301-101807-interfuser_baseline-224-baseline_full/`
**Notes**:
- Slightly better than EXP-000 (0.0145 vs 0.0265) — likely due to checkpoint warm-start + more effective LR scheduling on 22-epoch run
- eval_l1_error peaks at epoch 20 then slightly degrades (expected — late-stage cooldown)
- ⚠️ This metric uses timm's evaluation harness which computes waypoint L1 only

---

## EXP-003 — Temporal Smoke Test v1 (FAILED)

**Date**: 2026-03-01
**Job ID**: 50020282
**Script**: `scripts/slurm/temporal_smoke.sbatch`
**Purpose**: First end-to-end test of temporal model
**Config**: T=4, batch=2, depth=2, stride=1, Town01/Weather18
**Result**: FAILED
**Error**: `TypeError: conv2d() received list instead of Tensor`
**Root cause**: `lidar` field in data_dict is `np.ndarray`; `collate_temporal` only handled `torch.Tensor` type, fell through to `else: vals` (returns list)
**Fix applied**: Added numpy branch in `collate_temporal` (commit `0ff316d`)

---

## EXP-004 — Temporal Smoke Test v2 (FAILED)

**Date**: 2026-03-01
**Job ID**: 50020294
**Script**: `scripts/slurm/temporal_smoke.sbatch` (post-numpy fix)
**Purpose**: Second attempt after numpy fix
**Result**: FAILED
**Error**: `TypeError: cross_entropy_loss() argument 'target' must be Tensor, not list`
**Root cause**: `is_junction`, `traffic_light_state`, `stop_sign` are Python `int` scalars from `CarlaMVDetDataset`. Target tuple collation also fell through to `else: list` branch.
**Fix applied**: Added int/float/bool branch in target collation (commit `6f3e5ae`)

---

## EXP-005 — Temporal Smoke Test v3 (PASSED)

**Date**: 2026-03-01
**Job ID**: 50020296
**Script**: `scripts/slurm/temporal_smoke.sbatch` (both fixes applied)
**Purpose**: Validate temporal model end-to-end
**Config**:
```
Model:     InterFuserTemporal, T=4, depth=2
Pretrained backbone: EXP-000 checkpoint (20260210-...)
Dataset:   Town01, Weather18, 210 frames
Epochs:    1 specified → ~11 actual (cooldown)
Batch:     2, grad_accum=1 (effective batch=2)
LR:        5e-4 (temporal), 2e-4 (backbone)
```
**Results**:
- Steps: 103
- train_loss = 0.1223
- val_loss = 0.1216
- Checkpoint saved

**Wall time**: ~6 min
**Notes**:
- Both fixes (numpy + scalar targets) were necessary and sufficient
- Model forward pass works end-to-end
- Loss is reasonable (not NaN, not exploding)

---

## EXP-006 — Temporal Full Training

**Date**: 2026-03-01
**Job ID**: 50020302
**Script**: `scripts/slurm/temporal_train.sbatch`
**Purpose**: Full 12-epoch training of InterFuserTemporal
**Config**:
```
Model:     InterFuserTemporal, T=4, depth=2, stride=1
Pretrained backbone: EXP-000 checkpoint
Dataset:   Town01, Weather18, 210 frames
Epochs:    12
LR:        5e-4 (temporal), 2e-4 (backbone)
Batch:     2, grad_accum=2 (effective batch=4)
Schedule:  cosine, warmup 1 epoch
AdamW:     weight_decay=0.05, eps=1e-8
Clip grad: 10.0
```
**Results**:
| Epoch | train_loss | eval_loss |
|---|---|---|
| 1 | ~0.30+ | ~0.25+ |
| ... | decreasing | decreasing |
| 12 | — | **0.096125** ← best |

**Best checkpoint**: epoch 12 (final), eval_loss=0.096125
**Wall time**: 13m 17s (2× baseline, as expected for T=4)
**Output dir**: `runs/temporal/20260301-102924-temporal_full-T4/`
**Notes**:
- Loss is monotonically decreasing — model has NOT converged at epoch 12
- **Suggests: run for 24+ epochs** to reach plateau
- eval_loss is the composite 5-head loss — ⚠️ NOT directly comparable to baseline eval_l1_error
- Wall time is 2× baseline as expected (4 forward passes per sample)
- No NaN/divergence — training is stable

---

## Planned Experiments

| ID | Name | Config | Purpose | Status |
|---|---|---|---|---|
| EXP-007 | temporal_24epochs | T=4, 24 epochs | Check convergence | ⬜ Planned |
| EXP-008 | temporal_T2 | T=2, 12 epochs | Ablation: window size | ⬜ Planned |
| EXP-009 | temporal_T8 | T=8, 12 epochs | Ablation: window size | ⬜ Planned |
| EXP-010 | temporal_stride2 | T=4, stride=2 | Ablation: frame stride | ⬜ Planned |
| EXP-011 | temporal_depth1 | T=4, depth=1 | Ablation: encoder depth | ⬜ Planned |
| EXP-012 | temporal_depth4 | T=4, depth=4 | Ablation: encoder depth | ⬜ Planned |
| EXP-013 | temporal_approach2 | T=4, cross-attn | Approach 2 test | ⬜ Planned |
| EXP-014 | unified_eval | both checkpoints | Fair metric comparison | ⬜ Planned — URGENT |

---

## Observations Summary

1. **Temporal model trains stably** — no instability, NaN, or divergence observed
2. **Still converging at epoch 12** — val_loss monotonically decreasing, no plateau yet
3. **Wall time scales linearly with T** — T=4 → ~2× baseline, as expected
4. **Pretrained init is critical** — backbone warm-start enables fast convergence (val_loss <0.1 by epoch 12)
5. **Metric incompatibility is the key gap** — cannot compare baseline and temporal without unified eval

---

## Questions for Investigation

- Q1: Does temporal model achieve lower `eval_l1_error` (waypoint L1) than baseline? [UNKNOWN — need EXP-014]
- Q2: What is the optimal T? Does more history always help? [UNKNOWN — need EXP-008/009]
- Q3: Is Approach 2 (cross-attention) better than Approach 1 (concat)? [UNKNOWN — need EXP-013]
- Q4: Does temporal model generalize better to unseen towns/weathers? [UNKNOWN — need multi-town data]
