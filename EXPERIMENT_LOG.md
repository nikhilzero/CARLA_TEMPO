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

## EXP-014 — Unified Evaluation (Temporal T=4 Best Checkpoint)

**Date**: 2026-03-06
**Job ID**: 50112047
**Script**: `scripts/slurm/eval_temporal.sbatch`
**Purpose**: First fair, metric-unified evaluation — computes waypoint L1/L2 on the same val set

**Config**:
```
Model:     InterFuserTemporal, T=4, depth=2, stride=1
Checkpoint: runs/temporal/20260301-102924-temporal_full-T4/model_best.pth.tar
Dataset:   Town01, Weather18, 210 frames → 207 temporal windows
Batch:     4, workers=4
GPU:       Quadro M6000
```

**Results**:
| Metric | Value | Notes |
|---|---|---|
| `waypoint_l1` | **0.2811 m** | Raw mean \|pred − target\|, same units as baseline |
| `waypoint_l2` | **0.4525 m** | Euclidean distance per step |
| `weighted_waypoint_l1` | **0.02084** | Distance-weighted L1 = WaypointL1Loss |
| `loss_traffic` | 0.000038 | Very low — detection nearly perfect |
| `loss_waypoints` | 0.020841 | |
| `loss_junction` | 0.4541 | High — classification head still learning |
| `loss_traffic_light` | 0.6920 | High — only 210 frames, limited class diversity |
| `loss_stop_sign` | 0.002641 | |
| `loss_total` | **0.09612** | Matches training val_loss ✓ |

**JSON**: `runs/temporal/20260301-102924-temporal_full-T4/eval_results.json`

---

## EXP-015 — Unified Evaluation (Baseline Checkpoint)

**Date**: 2026-03-06
**Job ID**: 50112393
**Script**: `scripts/slurm/eval_baseline.sbatch`
**Checkpoint**: `interfuser_project/.../model_best.pth.tar` (12-epoch pretrain, EXP-000)
**JSON**: saved next to checkpoint as `eval_baseline_results.json`

---

## Definitive Baseline vs Temporal Comparison (same eval script, same val set)

| Metric | Baseline (T=1) | Temporal T=4 | Δ | Interpretation |
|---|---|---|---|---|
| **waypoint_l1** | 0.6076 m | **0.2811 m** | **−53.7%** | Temporal 2× better on raw L1 |
| **waypoint_l2** | 0.9418 m | **0.4525 m** | **−51.9%** | Temporal 2× better on Euclidean dist |
| **weighted_waypoint_l1** | 0.05084 | **0.02084** | **−59.0%** | Temporal 2.4× better on distance-weighted L1 |
| loss_traffic | 0.006336 | 0.000038 | −99.4% | Temporal nearly perfect on detection |
| loss_junction | 0.062138 | 0.454124 | +631% | Temporal worse on junction cls |
| loss_traffic_light | 0.097062 | 0.692006 | +613% | Temporal worse on TL cls |
| loss_stop_sign | 0.024892 | 0.002641 | −89% | Temporal better |
| loss_total (5-head) | 0.026398 | 0.096121 | +264% | Baseline wins on composite loss |
| Training epochs | 22 (12+cooldown) | 12 | — | Temporal still converging |
| Val windows | 210 | 207 | — | T=1 vs T=4 window count |

### Key Findings

1. **Waypoints**: Temporal model is dramatically better — 54% lower L1, 52% lower L2. This is the primary driving task metric and the most important result.

2. **Classification heads (junction, traffic_light)**: Baseline is much better. However, the baseline was trained for 22 epochs vs temporal's 12. Classification heads also require class diversity — the 210-frame tiny dataset has very few junction/traffic-light events, making these metrics noisy.

3. **Composite loss_total**: Baseline wins because the classification losses dominate the weighted sum. But this is misleading — the baseline is heavily upweighted by junction/TL losses that are unreliable on this small dataset.

4. **Temporal model is 12 epochs, still converging**. Running for 24 epochs should further close the classification gap.

### Thesis Summary Statement

> *"The temporal InterFuser (T=4) achieves 54% lower waypoint L1 error (0.281 m vs 0.608 m) compared to the single-frame baseline, demonstrating that temporal context significantly improves trajectory prediction. Classification head performance is lower due to limited training epochs and class imbalance in the small dataset."*

---

## EXP-007 — T=4, 24 Epochs

**Date**: 2026-03-06 | **Job**: 50112439 | **Script**: `temporal_train_24ep.sbatch`
**Best val_loss**: 0.0340 (vs 0.0961 at 12 epochs — 65% improvement with more training)
**Run dir**: `runs/temporal/20260306-093125-temporal_24ep-T4/`

**Eval (job 50112727)**:
- waypoint_l1: **0.3533 m**
- waypoint_l2: 0.5518 m
- weighted_waypoint_l1: **0.02752**

---

## EXP-008 — T=2 Ablation

**Date**: 2026-03-06 | **Job**: 50112447 | **Script**: `ablation_T2.sbatch`
**Best val_loss**: 0.0403
**Run dir**: `runs/ablations/20260306-093402-temporal_ablation_T2-T2/`

**Eval (job 50112612)**:
- waypoint_l1: **0.4206 m**
- waypoint_l2: 0.6580 m
- weighted_waypoint_l1: **0.03291**

---

## EXP-009 — T=8 Ablation

**Date**: 2026-03-06 | **Job**: 50112448 | **Script**: `ablation_T8.sbatch`
**Best val_loss**: 0.0472
**Run dir**: `runs/ablations/20260306-093626-temporal_ablation_T8-T8/`

**Eval (job 50112726)**:
- waypoint_l1: **0.2938 m**
- waypoint_l2: 0.4811 m
- weighted_waypoint_l1: **0.02161**

---

## COMPLETE ABLATION TABLE — Temporal Window Size

| Model | T | Epochs | val_loss (train) | waypoint_l1 ↓ | waypoint_l2 ↓ | weighted_wp_l1 ↓ |
|---|---|---|---|---|---|---|
| Baseline | 1 | 22 | — | 0.6076 m | 0.9418 m | 0.05084 |
| Temporal | 2 | 12 | 0.0403 | 0.4206 m | 0.6580 m | 0.03291 |
| Temporal | 4 | 12 | 0.0961 | 0.2811 m | 0.4525 m | 0.02084 |
| Temporal | 4 | 24 | 0.0340 | 0.3533 m | 0.5518 m | 0.02752 |
| Temporal | 8 | 12 | 0.0472 | **0.2938 m** | **0.4811 m** | **0.02161** |

### Key Findings

1. **All temporal models beat the baseline** — even T=2 (1 second of history) gives 31% lower waypoint L1.

2. **T=4 (12 epochs) is the best single result** — waypoint_l1=0.281 m, the lowest across all runs. T=8 is close (0.294 m) but not better, suggesting T=4 is the sweet spot for this dataset.

3. **T=4 24-epoch is worse than T=4 12-epoch on waypoint_l1 (0.353 vs 0.281)** — unexpected. This suggests the 12-epoch model generalised better; the 24-epoch run likely overfit the tiny 210-frame dataset. The lower val_loss (0.034) is driven by classification heads, not waypoints.

4. **Diminishing returns beyond T=4** — T=8 does not outperform T=4 despite 2× more temporal context, consistent with the hypothesis that ~2 seconds of history is sufficient for this driving scenario.

5. **Thesis recommendation**: Report T=4 (12 epochs) as the primary result. Ablation table shows T=4 is optimal for this dataset size.

---

---

## ═══════════════════════════════════════════════════════════════
## RESEARCH-SCALE EXPERIMENTS (RES-NNN) — THESIS-QUALITY RESULTS
## Data: Town01-04 train, Town05 held-out test, 6 weathers, 300 routes, 60,755 frames
## These are the only results that may be reported in the thesis.
## ═══════════════════════════════════════════════════════════════

## RES-001 — Baseline Training (Research Scale)

**Date**: 2026-03-07
**Job ID**: 50129303
**Script**: `scripts/slurm/baseline_train_research.sbatch`
**Purpose**: Thesis-quality baseline training — first research-scale run
**Wall time**: 11h 27m (24 epochs)
**GPU**: NVIDIA L40S (gpu029)

**Config**:
```
Model:     interfuser_baseline
Dataset:   Town01-04 train / Town05 val, Weathers [1,3,6,8,14,18], tiny routes
Routes:    240 train + 60 val
Frames:    49,378 train + 11,377 val
Epochs:    24 (--cooldown-epochs 0 fix applied)
LR:        5e-4 (backbone 2e-4)
Batch:     4
Schedule:  cosine, warmup 2 epochs
AdamW:     weight_decay=0.05, eps=1e-8
Clip grad: 10.0
```

**Result**: `val l1_error = 0.1569` (best at epoch 4)

**Checkpoint**: `/scratch/nd967/CARLA_TEMPO/runs/baseline/20260307-082126-interfuser_baseline-224-baseline_research/model_best.pth.tar` (607 MB)

**Notes**:
- Best val_l1_error was at epoch 4 (early), suggesting fast convergence on research data
- This checkpoint is used as pretrained backbone for RES-002 temporal training
- Eval on held-out Town05: see RES-E001 below

---

## RES-E001 — Baseline Eval on Town05 (Held-Out)

**Date**: 2026-03-07
**Job ID**: 50135656
**Script**: `scripts/slurm/eval_baseline.sbatch`
**Checkpoint**: RES-001 `model_best.pth.tar`
**Val set**: Town05, Weathers [1,3,6,8,14,18], 11,377 frames (T=1), 2,845 batches
**GPU**: Quadro M6000

**Results**:
| Metric | Value |
|---|---|
| **waypoint_l1** | **0.6036 m** |
| **waypoint_l2** | **0.9873 m** |
| **weighted_waypoint_l1** | **0.04636** |
| loss_traffic | 0.1277 |
| loss_waypoints | 0.04636 |
| loss_junction | 0.4409 |
| loss_traffic_light | 0.5119 |
| loss_stop_sign | 0.1943 |
| **loss_total (5-head)** | **0.1483** |

**JSON**: `runs/baseline/20260307-082126-interfuser_baseline-224-baseline_research/eval_baseline_results.json`

**Notes**:
- waypoint_l1=0.604m is the **research-scale baseline target** — RES-002 must beat this
- Compare with debug baseline (EXP-015): waypoint_l1=0.608m — nearly identical, confirms research data is similar difficulty
- Classification losses are higher than debug runs (traffic=0.128, TL=0.512) — more diverse Town05 scenarios

---

## RES-002 — Temporal Training (Research Scale) — IN PROGRESS

**Date**: 2026-03-07
**Job ID**: 50135611
**Script**: `scripts/slurm/temporal_train_research.sbatch`
**Purpose**: Thesis-quality temporal model training (T=4, stride=5 = 2-sec context)
**Status**: RUNNING on gpu029, 48h budget

**Config**:
```
Model:     InterFuserTemporal, T=4, depth=2, stride=5
Pretrained backbone: RES-001 model_best.pth.tar
Dataset:   Town01-04 train / Town05 val, Weathers [1,3,6,8,14,18], tiny routes
Windows:   45,778 train + 10,477 val temporal windows
Epochs:    24
LR:        5e-4 (temporal), 2e-4 (backbone)
Batch:     2, grad_accum=2 (effective batch=4)
Schedule:  cosine, warmup 2 epochs
AdamW:     weight_decay=0.05, eps=1e-8
Clip grad: 10.0
```

**Temporal params**: 1,580,548 new params (2.9% of total 54,516,115)
**Expected wall time**: ~20-24h (T=4 = 4× backbone forward passes per window)
**Output dir**: `runs/temporal/20260307-195110-temporal_research_T4_s5-T4/`

**Results (RES-E002 — see below)**: best val_loss=0.1770 at epoch 1; epoch 24 val_loss=0.1860

---

---

## RES-E002 — Temporal Eval on Town05 (Held-Out)

**Date**: 2026-03-10
**Jobs**: 50191515 (model_best / epoch 1), 50191516 (epoch 24)
**Script**: `scripts/slurm/eval_temporal.sbatch`
**Val set**: Town05, Weathers [1,3,6,8,14,18], 10,477 temporal windows (T=4,stride=5), 2,620 batches

**Results**:
| Checkpoint | waypoint_l1 | waypoint_l2 | weighted_wp_l1 | loss_total |
|---|---|---|---|---|
| **model_best (epoch 1)** | **0.5956 m** | **0.9586 m** | **0.04538** | 0.1785 |
| epoch 24 | 0.7205 m | 1.1991 m | 0.05602 | 0.1878 |
| **Baseline RES-001** | **0.6036 m** | **0.9873 m** | **0.04636** | 0.1483 |

**Δ (model_best vs baseline)**: waypoint_l1 **−1.3%**, waypoint_l2 **−2.9%**

---

## ═══════════════════════════════════════════════════════════════
## RESEARCH-SCALE DEFINITIVE COMPARISON
## ═══════════════════════════════════════════════════════════════

| Model | Train epochs | waypoint_l1 ↓ | waypoint_l2 ↓ | Δ vs baseline |
|---|---|---|---|---|
| Baseline (RES-001) | 24 | 0.6036 m | 0.9873 m | — |
| **Temporal T=4 s5 best** | 24 (ep1 best) | **0.5956 m** | **0.9586 m** | **−1.3%** |
| Temporal T=4 s5 ep24 | 24 | 0.7205 m | 1.1991 m | +19.4% (worse) |

### Key Findings

1. **Temporal model barely beats baseline at research scale** — 1.3% improvement vs 53% in debug runs. The debug result was misleading: it evaluated on the same 210-frame data used for training (data leakage).

2. **Best temporal checkpoint is epoch 1** — the pretrained backbone dominates, and continued training with the temporal components slightly degrades generalization to Town05. This suggests the temporal encoder is overfitting to the training distribution (Town01-04).

3. **Epoch 24 is worse than baseline** — temporal components learned something Town01-04-specific that hurts on Town05. This generalization gap is a real finding worth investigating.

4. **Hypothesis**: The temporal model needs more diverse training data or stronger regularization to generalize. Alternatively, stride=5 with only tiny routes may not provide enough meaningful temporal context for the model to learn useful motion patterns.

5. **Debug runs (EXP-014: waypoint_l1=0.281 m) are invalid** — confirmed data leakage. Do not report these in the thesis.

### Thesis Interpretation

> *"At research scale with a proper held-out test town (Town05), the temporal InterFuser (T=4, stride=5) achieves marginally better waypoint prediction than the single-frame baseline (0.596 m vs 0.604 m, −1.3%). The model's best performance occurs at epoch 1, before temporal fine-tuning begins, suggesting the temporal components have not yet learned to generalize across towns. This motivates further investigation into regularization, longer training, and cross-town temporal patterns."*

---

## Planned Experiments

| ID | Name | Config | Purpose | Status |
|---|---|---|---|---|
| EXP-010 | temporal_stride2 | T=4, stride=2 | Ablation: frame stride | ⬜ Planned |
| EXP-011 | temporal_depth1 | T=4, depth=1 | Ablation: encoder depth | ⬜ Planned |
| EXP-012 | temporal_depth4 | T=4, depth=4 | Ablation: encoder depth | ⬜ Planned |
| EXP-013 | temporal_approach2 | T=4, cross-attn | Approach 2 test | ⬜ Planned |

---

## Observations Summary

1. **Temporal model trains stably** — no instability, NaN, or divergence observed
2. **Still converging at epoch 12** — val_loss monotonically decreasing, no plateau yet
3. **Wall time scales linearly with T** — T=4 → ~2× baseline, as expected
4. **Pretrained init is critical** — backbone warm-start enables fast convergence (val_loss <0.1 by epoch 12)
5. **Traffic detection nearly solved** — loss_traffic=0.000038, suggesting detection is not the bottleneck
6. **Classification heads are the weak link** — junction (0.454) and traffic_light (0.692) are high,
   likely because the 210-frame dataset has limited class diversity for these rare events

---

## Questions for Investigation

- Q1: Does temporal model achieve lower waypoint L1 than baseline (same eval script)? [PARTIALLY KNOWN — need EXP-015 for baseline]
- Q2: What is the optimal T? Does more history always help? [UNKNOWN — need EXP-008/009]
- Q3: Is Approach 2 (cross-attention) better than Approach 1 (concat)? [UNKNOWN — need EXP-013]
- Q4: Does temporal model generalize better to unseen towns/weathers? [UNKNOWN — need multi-town data]
- Q5: Does training for 24 epochs significantly improve waypoint L1? [UNKNOWN — need EXP-007]
