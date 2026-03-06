# TODO_MASTER.md
# CARLA_TEMPO — Master Task Checklist

_Last updated: 2026-03-06_
Status legend: ✅ Done | 🔄 In Progress | ⬜ Not Started | ⚠️ Blocked

---

## PHASE 0 — Infrastructure & Setup

- [x] ✅ Clone InterFuser baseline into repo
- [x] ✅ Set up `interfuser` conda env on Amarel (Python 3.8, PyTorch 2.1.0+cu121, CUDA 12.1)
- [x] ✅ Convert dataset: `rgb_full` → `rgb_front/left/right/rear` channels
- [x] ✅ Confirm dataset at `/scratch/nd967/CARLA_TEMPO/InterFuser/dataset/`
- [x] ✅ Fix baseline `train.py` bugs (distributed training checks, validation loss variables)
- [x] ✅ Create `.gitignore` (excludes runs/, *.pth, logs/, data files)
- [x] ✅ Create `RUNBOOK.md` (operational reference)
- [x] ✅ Set up Mac → Amarel git sync workflow
- [ ] ⬜ Push 5 pending local commits to GitHub (via Amarel, since Mac has no SSH key)

---

## PHASE 1 — Baseline Training

- [x] ✅ Write `scripts/slurm/baseline_smoke.sbatch` (1 epoch sanity check)
- [x] ✅ Write `scripts/slurm/baseline_train.sbatch` (12-epoch full run)
- [x] ✅ Run baseline smoke (job 50020270) — PASSED
- [x] ✅ Run baseline full (job 50020281) — COMPLETED, best eval_l1_error=0.014530
- [x] ✅ Checkpoint stored: `runs/baseline/20260301-101807-interfuser_baseline-224-baseline_full/model_best.pth.tar`
- [ ] ⬜ Document baseline eval_l1_error curve in EXPERIMENT_LOG.md ← do this

---

## PHASE 2 — Temporal Model (Approach 1: Concat Fusion)

### 2a. Model Architecture
- [x] ✅ Implement `InterFuserTemporal` in `temporal/models/interfuser_temporal.py`
- [x] ✅ Temporal position embeddings (learnable, shape T×1×1×D)
- [x] ✅ 2-layer temporal Transformer encoder (8 heads, 4× feedforward)
- [x] ✅ Temporal pooling layer (learned weighted average → N tokens)
- [x] ✅ `build_interfuser_temporal()` factory with pretrained backbone loading
- [x] ✅ Verified 1.58M new parameters (~2.9% of total)

### 2b. Data Pipeline
- [x] ✅ `TemporalWindowDataset` — episode-aware window builder
- [x] ✅ `episode_lengths_from_carla_dataset()` — infers route boundaries
- [x] ✅ `collate_temporal()` — handles tensor/numpy/scalar types
- [x] ✅ Fix: numpy arrays (lidar) → `torch.from_numpy()`
- [x] ✅ Fix: scalar targets (is_junction, traffic_light, stop_sign) → `torch.tensor()`

### 2c. Training
- [x] ✅ `temporal/train.py` — full training loop
- [x] ✅ `temporal/utils/losses.py` — 5-head losses (WaypointL1, MVTL1, CrossEntropy)
- [x] ✅ Differential LR: backbone=2e-4, temporal params=5e-4
- [x] ✅ Gradient accumulation (grad_accum=2 → effective batch=4)
- [x] ✅ Cosine LR schedule with linear warmup
- [x] ✅ Checkpointing: model_best + last + epoch checkpoints
- [x] ✅ CSV loss logging (epoch, train_loss, eval_loss)

### 2d. SLURM & Experiments
- [x] ✅ `scripts/slurm/temporal_smoke.sbatch` — 1-epoch sanity check
- [x] ✅ `scripts/slurm/temporal_train.sbatch` — 12-epoch full run
- [x] ✅ Temporal smoke PASSED (job 50020296, val_loss=0.1216)
- [x] ✅ Temporal full COMPLETED (job 50020302, best val_loss=0.096125 at epoch 12)
- [x] ✅ Results packaged: `results_pack/` + `results_pack.tar.gz` (2.3 GB)

---

## PHASE 3 — Evaluation & Comparison (HIGH PRIORITY)

- [ ] ⬜ **Write unified eval script** `temporal/eval.py` that:
  - Loads either baseline or temporal checkpoint
  - Runs on val set
  - Reports waypoint L1, waypoint L2, and composite loss
  - Outputs comparison-ready metrics
- [ ] ⬜ Run unified eval on baseline `model_best.pth.tar`
- [ ] ⬜ Run unified eval on temporal `model_best.pth.tar`
- [ ] ⬜ Plot loss curves (baseline vs temporal) side-by-side
- [ ] ⬜ Add `eval_l1_error` computation to `temporal/train.py` validation loop
- [ ] ⬜ Re-run temporal training with `eval_l1_error` logged to CSV (apples-to-apples)

---

## PHASE 4 — Ablation Studies

### 4a. Temporal Window Size
- [ ] ⬜ T=2 (1 second history @ 2 Hz) — new sbatch + run
- [ ] ⬜ T=4 (2 seconds) — ✅ already done
- [ ] ⬜ T=8 (4 seconds) — new sbatch + run
- [ ] ⬜ Compare: val_loss and eval_l1_error vs T

### 4b. Frame Stride
- [ ] ⬜ stride=1 (every frame) — ✅ already done
- [ ] ⬜ stride=2 (skip 1 frame → 4 sec history with T=4) — new run
- [ ] ⬜ Compare: val_loss vs stride

### 4c. Temporal Encoder Depth
- [ ] ⬜ depth=1 — run
- [ ] ⬜ depth=2 — ✅ already done
- [ ] ⬜ depth=4 — run
- [ ] ⬜ Compare: val_loss vs depth

### 4d. Training Duration
- [ ] ⬜ Train temporal T=4 for 24 epochs (model still improving at epoch 12)
- [ ] ⬜ Verify convergence / plateau

---

## PHASE 5 — Approach 2: Temporal Cross-Attention

- [ ] ⬜ Design temporal cross-attention module:
  - Between-frame attention (each frame attends to all other frames)
  - Query from current frame, Key/Value from history frames
- [ ] ⬜ Implement `InterFuserTemporalAttn` in `temporal/models/interfuser_temporal_attn.py`
- [ ] ⬜ Add training support in `temporal/train.py` (model type selector)
- [ ] ⬜ Run smoke test for Approach 2
- [ ] ⬜ Run full 12-epoch training for Approach 2
- [ ] ⬜ Compare Approach 1 vs Approach 2 on eval_l1_error

---

## PHASE 6 — Multi-Town & Generalization

- [ ] ⬜ Confirm Amarel dataset has multi-town data (check dataset_index.txt)
  - UNKNOWN: does `/scratch/nd967/CARLA_TEMPO/InterFuser/dataset/` have towns 2,3,4,...?
  - Verification: `cat /scratch/nd967/CARLA_TEMPO/InterFuser/dataset/dataset_index.txt`
- [ ] ⬜ Train temporal model on all available towns (if data exists)
- [ ] ⬜ Evaluate on held-out town (Town05 as val)
- [ ] ⬜ Document generalization gap

---

## PHASE 7 — CARLA Closed-Loop Evaluation

- [ ] ⬜ Set up CARLA server (on Amarel or separate machine)
  - UNKNOWN: Is CARLA server available on Amarel? Check `module avail carla`
- [ ] ⬜ Integrate checkpoint with CARLA leaderboard harness
  - Files: `InterFuser/leaderboard/` — needs audit
- [ ] ⬜ Run RouteCompletion evaluation for baseline checkpoint
- [ ] ⬜ Run RouteCompletion evaluation for temporal checkpoint
- [ ] ⬜ Report: RouteCompletion %, InfractionsPerKm, DrivingScore

---

## PHASE 8 — Documentation & Thesis

- [x] ✅ `RUNBOOK.md` — operational reference
- [x] ✅ `CLAUDE.md` — project context
- [ ] ⬜ `PROJECT_STATUS.md` — this file (write and maintain)
- [ ] ⬜ `EXPERIMENT_LOG.md` — all runs with settings and results
- [ ] ⬜ `DECISIONS.md` — architectural decisions with rationale
- [ ] ⬜ Architecture diagram (PDF/PNG) for thesis
- [ ] ⬜ Thesis chapter: Temporal Extension (method section)
- [ ] ⬜ Thesis chapter: Experiments and Results

---

## PHASE 9 — Code Quality & GitHub

- [ ] ⬜ Push 5 pending local commits to GitHub
- [ ] ⬜ Wire `configs/temporal_t4.yaml` into `train.py` (replace argparse with YAML+override)
- [ ] ⬜ Add AMP (mixed precision) training to `temporal/train.py`
- [ ] ⬜ Add TensorBoard logging to `temporal/train.py`
- [ ] ⬜ Unit tests for `collate_temporal`, `TemporalWindowDataset`
- [ ] ⬜ Remove legacy `scripts/train_baseline.sh` and `scripts/train_temporal.sh`

---

## Immediate Next Actions (this week)

1. ⬜ Push commits to GitHub (from Amarel)
2. ⬜ Write `temporal/eval.py` unified evaluation script
3. ⬜ Re-run temporal training with eval_l1_error logged
4. ⬜ Run T=2 and T=8 ablations
5. ⬜ Check if GitHub repo (when provided) has useful components
