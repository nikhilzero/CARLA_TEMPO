# CARLA_TEMPO Runbook

Master's thesis — Rutgers Camden. Temporal extension of InterFuser for autonomous driving.

---

## Quick Reference

| Step | Command |
|---|---|
| Push from Mac | `git add . && git commit -m "msg" && git push` |
| Pull on Amarel | `ssh nd967@amarel.hpc.rutgers.edu "cd /scratch/nd967/CARLA_TEMPO && git pull"` |
| Submit baseline smoke | `sbatch scripts/slurm/baseline_smoke.sbatch` |
| Submit baseline full | `sbatch scripts/slurm/baseline_train.sbatch` |
| Submit temporal smoke | `sbatch scripts/slurm/temporal_smoke.sbatch` |
| Submit temporal full | `sbatch scripts/slurm/temporal_train.sbatch` |
| Check queue | `squeue -u nd967` |
| Tail log | `tail -f /scratch/nd967/CARLA_TEMPO/logs/<logfile>.out` |

---

## Environment

- **Cluster**: Amarel HPC, Rutgers University
- **Login**: `nd967@amarel.hpc.rutgers.edu`
- **Working dir**: `/scratch/nd967/CARLA_TEMPO`
- **Conda env**: `interfuser` (Python 3.8, PyTorch 2.1.0+cu121, CUDA 12.1)
- **GPU partition**: `--partition=gpu --gres=gpu:1`

```bash
ssh nd967@amarel.hpc.rutgers.edu
cd /scratch/nd967/CARLA_TEMPO
source ~/miniconda3/bin/activate interfuser
```

---

## Data

Located at: `/scratch/nd967/CARLA_TEMPO/InterFuser/dataset/`

```
dataset_index.txt              # lists routes and frame counts
weather-18_data_town01_tiny_w18/   # 210 frames, town 1, weather 18
  ├── rgb_front/  rgb_left/  rgb_right/  rgb_rear/
  ├── lidar/
  ├── measurements/  measurements_full/
  ├── affordances/  actors_data/  3d_bbs/
  └── topdown/  birdview/
```

The `dataset_index.txt` is the registry used by `CarlaMVDetDataset`.
Pattern in the index: `weather-<W>_data_town<TT>_*/  <N_frames>`

---

## Mac → Amarel Workflow

```bash
# 1. Make changes locally
# 2. Commit and push
git add -p                          # stage specific changes
git commit -m "feat: description"
git push

# 3. Pull on Amarel (one-liner)
ssh nd967@amarel.hpc.rutgers.edu "cd /scratch/nd967/CARLA_TEMPO && git pull"

# 4. Submit job (see sections below)
```

---

## Baseline InterFuser

### What it does
Runs the upstream InterFuser `train.py` with the same hyperparameters used in
the 12-epoch run that achieved `eval_l1_error = 0.0265`.

### Smoke test (always run this first)
```bash
# From /scratch/nd967/CARLA_TEMPO on Amarel:
sbatch scripts/slurm/baseline_smoke.sbatch

# Monitor:
squeue -u nd967
tail -f /scratch/nd967/CARLA_TEMPO/logs/baseline_smoke_<JOBID>.out

# Verify success: checkpoint should appear in
ls /scratch/nd967/CARLA_TEMPO/runs/baseline/
```

Expected smoke output:
```
Epoch 1/1 — train_loss drops, checkpoint saved in runs/baseline/<timestamp>-baseline_smoke/
```

### Full training run
```bash
sbatch scripts/slurm/baseline_train.sbatch
# Time limit: 2 hours  |  12 epochs  |  ~30 min actual runtime
```

### Resume a crashed run
```bash
# Find the last checkpoint:
ls -lt /scratch/nd967/CARLA_TEMPO/runs/baseline/<run_dir>/

# Re-run baseline train.py with --resume:
cd /scratch/nd967/CARLA_TEMPO/InterFuser/interfuser
torchrun --nproc_per_node=1 --master_port=29500 train.py \
    /scratch/nd967/CARLA_TEMPO/InterFuser/dataset/ \
    --dataset carla --train-towns 1 --val-towns 1 \
    --train-weathers 18 --val-weathers 18 \
    --model interfuser_baseline \
    --sched cosine --epochs 12 --warmup-epochs 1 --lr 0.0005 --batch-size 4 \
    --no-prefetcher --eval-metric l1_error \
    --opt adamw --opt-eps 1e-8 --weight-decay 0.05 \
    --scale 0.9 1.1 --saver-decreasing --clip-grad 10 --freeze-num -1 \
    --with-backbone-lr --backbone-lr 0.0002 \
    --multi-view --with-lidar --multi-view-input-size 3 128 128 \
    --experiment baseline_full \
    --output /scratch/nd967/CARLA_TEMPO/runs/baseline \
    --workers 4 --pretrained \
    --resume /scratch/nd967/CARLA_TEMPO/runs/baseline/<run_dir>/last.pth.tar
```

### Output locations
```
runs/baseline/<timestamp>-baseline_smoke-*/
runs/baseline/<timestamp>-baseline_full-*/
  ├── model_best.pth.tar   # best checkpoint (lowest val loss)
  ├── last.pth.tar         # latest checkpoint
  ├── checkpoint-N.pth.tar # epoch N checkpoint
  ├── summary.csv          # epoch, train_loss, eval_loss, eval_l1_error
  └── args.yaml            # all hyperparameters
```

---

## Temporal InterFuser

### Architecture
- T=4 consecutive frames fed through the same InterFuser backbone
- Temporal position embeddings added per frame
- 2-layer temporal transformer encoder fuses T×N tokens → N tokens
- Original InterFuser decoder + all 5 prediction heads unchanged
- Same loss as baseline (traffic + waypoints + junction + traffic_light + stop_sign)

### Pretrained backbone
```
/scratch/nd967/interfuser_project/InterFuser/interfuser/output/
  20260210-110341-interfaser_baseline-224-real_data_test/model_best.pth.tar
```
12-epoch baseline, `eval_l1_error = 0.0265`.

### Smoke test
```bash
sbatch scripts/slurm/temporal_smoke.sbatch

# Monitor:
squeue -u nd967
tail -f /scratch/nd967/CARLA_TEMPO/logs/temporal_smoke_<JOBID>.out

# Verify: checkpoint in runs/temporal/
ls /scratch/nd967/CARLA_TEMPO/runs/temporal/
```

### Full training run
```bash
sbatch scripts/slurm/temporal_train.sbatch
# Time limit: 3 hours  |  12 epochs  |  batch=2, grad_accum=2 (effective batch=4)
```

### Resume
```bash
cd /scratch/nd967/CARLA_TEMPO
python -m temporal.train \
    --data-dir InterFuser/dataset \
    --train-towns 1 --val-towns 1 \
    --train-weathers 18 --val-weathers 18 \
    --temporal-frames 4 --frame-stride 1 --temporal-depth 2 \
    --epochs 12 --warmup-epochs 1 \
    --batch-size 2 --grad-accum 2 \
    --lr 0.0005 --backbone-lr 0.0002 --weight-decay 0.05 --clip-grad 10 \
    --workers 4 --experiment temporal_full \
    --output runs/temporal \
    --pretrained-backbone /scratch/nd967/interfuser_project/InterFuser/interfuser/output/20260210-110341-interfuser_baseline-224-real_data_test/model_best.pth.tar \
    --resume runs/temporal/<run_dir>/last.pth.tar
```

### Output locations
```
runs/temporal/<timestamp>-temporal_smoke-T4/
runs/temporal/<timestamp>-temporal_full-T4/
  ├── model_best.pth.tar
  ├── last.pth.tar
  ├── checkpoint-N.pth.tar
  └── summary.csv          # epoch, train_loss, eval_loss
```

---

## Monitoring

```bash
# Job queue
squeue -u nd967

# Live log tail
tail -f /scratch/nd967/CARLA_TEMPO/logs/<logfile>_<JOBID>.out

# Cancel a job
scancel <JOBID>

# Check GPU usage on node (if job is running)
srun --jobid=<JOBID> nvidia-smi

# Disk usage in scratch
du -sh /scratch/nd967/CARLA_TEMPO/
du -sh /scratch/nd967/CARLA_TEMPO/runs/
```

---

## SLURM Script Reference

| Script | Time | Epochs | Purpose |
|---|---|---|---|
| `scripts/slurm/baseline_smoke.sbatch` | 30 min | 1 | Sanity check baseline env + data |
| `scripts/slurm/baseline_train.sbatch` | 2 hr | 12 | Full baseline training run |
| `scripts/slurm/temporal_smoke.sbatch` | 30 min | 1 | Sanity check temporal model + data |
| `scripts/slurm/temporal_train.sbatch` | 3 hr | 12 | Full temporal training run |

All scripts:
- Accept `DATA_ROOT` env var override: `DATA_ROOT=/other/path sbatch ...`
- Write SLURM logs to `/scratch/nd967/CARLA_TEMPO/logs/`
- Write model outputs to `/scratch/nd967/CARLA_TEMPO/runs/<baseline|temporal>/`

---

## Key File Locations

| File | Purpose |
|---|---|
| `InterFuser/interfuser/train.py` | Baseline training entrypoint |
| `InterFuser/interfuser/timm/models/interfuser.py` | InterFuser model architecture |
| `temporal/models/interfuser_temporal.py` | Temporal model (wraps baseline) |
| `temporal/data/temporal_dataset.py` | Temporal window dataset wrapper |
| `temporal/utils/losses.py` | Loss functions (copied from baseline) |
| `temporal/train.py` | Temporal training entrypoint |
| `configs/temporal_t4.yaml` | Default temporal config (T=4) |
| `CLAUDE.md` | Full project context for Claude Code |
