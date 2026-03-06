# CHANGELOG.md
# CARLA_TEMPO — Structured Development Log

_Format: [YYYY-MM-DD] TYPE: description_
Types: FEAT | FIX | INFRA | DOCS | EXPERIMENT | REFACTOR

---

## 2026-03-06

**[2026-03-06] DOCS: Generated project tracking documents**
- Created PROJECT_STATUS.md, TODO_MASTER.md, CHANGELOG.md, MIGRATION_PLAN.md, DECISIONS.md, EXPERIMENT_LOG.md
- Full project state audit performed

---

## 2026-03-01

**[2026-03-01] EXPERIMENT: Temporal full training run completed (job 50020302)**
- 12 epochs on Town01/Weather18, T=4, batch=2+grad_accum=2
- Best val_loss=0.096125 at epoch 12 (still converging)
- Checkpoint: `runs/temporal/20260301-102924-temporal_full-T4/model_best.pth.tar`
- Wall time: 13m 17s

**[2026-03-01] EXPERIMENT: Baseline full training run completed (job 50020281)**
- 22 epochs total (12 + 10 cooldown due to timm scheduler default)
- Best eval_l1_error=0.014530 at epoch 20 (final epoch 21: 0.016369)
- Checkpoint: `runs/baseline/20260301-101807-interfuser_baseline-224-baseline_full/model_best.pth.tar`
- Wall time: 6m 57s

**[2026-03-01] INFRA: Created results_pack on Amarel**
- Packaged both run dirs, SLURM logs, COMPARISON.md, comparison.csv
- Archived to `/scratch/nd967/CARLA_TEMPO/results_pack.tar.gz` (2.3 GB)

**[2026-03-01] FIX: collate_temporal — scalar targets (job 50020294 → 50020296)**
- Bug: `cross_entropy_loss() argument 'target' must be Tensor, not list`
- Root cause: is_junction, traffic_light_state, stop_sign are Python int scalars from
  CarlaMVDetDataset; target tuple collation fell through to `else: list` branch
- Fix: Added `elif isinstance(..., (int, float, bool)): torch.tensor([...])` in target collation
- Commit: `6f3e5ae fix: collate_temporal numpy→tensor`

**[2026-03-01] FIX: collate_temporal — numpy arrays (job 50020282 → 50020294)**
- Bug: `TypeError: conv2d() received list instead of Tensor`
- Root cause: lidar field is `np.ndarray`; collate_temporal only had torch.Tensor branch
- Fix: Added `import numpy as np` + `elif isinstance(vals[0], np.ndarray): torch.stack([torch.from_numpy(v) for v in vals])`
- Commit: `0ff316d fix: collate_temporal numpy→tensor`

**[2026-03-01] EXPERIMENT: Temporal smoke FAILED twice, then PASSED**
- Job 50020282: numpy array bug in collate_temporal → FAILED
- Job 50020294: scalar target bug in collate_temporal → FAILED
- Job 50020296: both fixes applied → PASSED (103 steps, train_loss=0.1223, val_loss=0.1216)

**[2026-03-01] EXPERIMENT: Baseline smoke completed (job 50020270)**
- 1-epoch sanity check passed; ran ~11 epochs due to cooldown_epochs=10 default
- Checkpoints saved in `runs/baseline/`

**[2026-03-01] FEAT: temporal/train.py — complete working training loop**
- Implemented `build_carla_dataset()` using `create_carla_dataset` + `create_carla_loader`
  (loader discarded; transforms set as side-effect on base_ds)
- Full 5-head loss matching baseline (traffic×0.5 + waypoints×0.2 + junction×0.05 + tl×0.1 + stop×0.01)
- Differential LR: backbone=2e-4, temporal params=5e-4
- Gradient accumulation, cosine LR schedule, CSV logging, checkpointing
- Commit: `eaa306a temporal: implement dataset wrapper, full 5-head loss, working train loop`

**[2026-03-01] FEAT: temporal/utils/losses.py — 5-head loss functions**
- WaypointL1Loss and MVTL1Loss copied verbatim from baseline for fair comparison
- `build_loss_fns()` returns dict with traffic/waypoints/cls/stop_cls
- CrossEntropyLoss confirmed (not BCEWithLogitsLoss) based on baseline inspection

**[2026-03-01] FEAT: temporal/data/temporal_dataset.py — TemporalWindowDataset**
- Episode-aware window builder (never crosses route boundaries)
- `episode_lengths_from_carla_dataset()` reads `ds.route_frames`
- `collate_temporal()` handles all data types
- Commit: `eaa306a temporal: implement dataset wrapper, full 5-head loss, working train loop`

**[2026-03-01] INFRA: SLURM scripts created**
- `scripts/slurm/baseline_smoke.sbatch` — 30 min, 1 epoch, torchrun
- `scripts/slurm/baseline_train.sbatch` — 2 hr, 12 epochs, torchrun
- `scripts/slurm/temporal_smoke.sbatch` — 30 min, 1 epoch, python -m temporal.train
- `scripts/slurm/temporal_train.sbatch` — 3 hr, 12 epochs, python -m temporal.train
- Commit: `cf78662 baseline scripts: add slurm smoke+train with correct data path and output dir`

**[2026-03-01] DOCS: RUNBOOK.md and .gitignore**
- Full operational reference for Mac→Amarel workflow, data paths, job submission, monitoring
- Commit: `2f50669 docs: add RUNBOOK.md and .gitignore`

**[2026-03-01] INFRA: Files pushed via Amarel (Mac SSH key issue)**
- Local Mac SSH key not authorized for GitHub
- Workaround: SCP files → Amarel → commit+push from Amarel

---

## 2026-02-27

**[2026-02-27] FEAT: InterFuserTemporal model (Approach 1 — Concat Fusion)**
- `temporal/models/interfuser_temporal.py` implemented
- Architecture: per-frame forward_features → temporal pos embeddings → concat → 2-layer temporal encoder → pooling → original decoder
- `build_interfuser_temporal()` factory with pretrained backbone support
- ~1.58M new parameters (2.9% of total ~54M)
- Commit: `ea98410 fix: corrected temporal model wrapping real InterFuser architecture`

**[2026-02-27] INFRA: Project structure scaffold**
- `temporal/` package created with models/, data/, utils/
- `configs/temporal_t4.yaml` created
- `requirements.txt` (pyyaml, einops)
- Commit: `e9eea4f project structure: InterFuser baseline + temporal extension scaffold`

---

## 2026-02-10

**[2026-02-10] EXPERIMENT: Baseline InterFuser trained on Amarel (interfuser_project)**
- 12 epochs on LMDrive dataset (Town01, Weather 18)
- Result: eval_l1_error=0.0265
- Checkpoint: `/scratch/nd967/interfuser_project/InterFuser/interfuser/output/20260210-110341-interfuser_baseline-224-real_data_test/model_best.pth.tar` (636 MB)
- Known fixes applied: distributed training checks, validation loss variables

**[2026-02-10] INFRA: Dataset converted to baseline format**
- `rgb_full` → separate `rgb_front/left/right/rear` directories
- Dataset at `/scratch/nd967/CARLA_TEMPO/InterFuser/dataset/`
- 210 frames, Town01, Weather 18 (`weather-18_data_town01_tiny_w18/`)

---

## Initial

**[Initial] INFRA: Repository initialized**
- Cloned `opendilab/InterFuser` into `InterFuser/`
- Set up CARLA_TEMPO repo structure
- Commit: `33b5ad6 initial commit`
