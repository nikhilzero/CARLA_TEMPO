# CARLA_TEMPO — Temporal Reasoning for Autonomous Driving

Master's thesis, Rutgers University–Camden (2026).
Extends [InterFuser](https://github.com/opendilab/InterFuser) (Shao et al., CoRL 2022) with multi-frame temporal reasoning and evaluates it in closed-loop CARLA simulation.

---

## Motivation

Transformer-based autonomous driving models such as InterFuser process a **single snapshot** of sensor input at each timestep. This makes it impossible for the model to directly perceive motion, velocity, or the intent of surrounding agents — information that human drivers routinely use. A vehicle that was stationary one second ago but is now accelerating presents a fundamentally different situation than one that has been stationary for ten seconds.

**This work adds temporal reasoning** by feeding T consecutive frames into the model, allowing the transformer to attend across time and encode motion context. We evaluate whether temporal history improves closed-loop driving performance on held-out Town05 routes in CARLA.

---

## Architecture

CARLA_TEMPO wraps InterFuser with a temporal encoder that processes a sliding window of T frames. Each frame is encoded independently by the shared multi-view encoder; the resulting token sequences are concatenated and fed into the transformer encoder.

### Temporal Concat Fusion (`temporal/models/interfuser_temporal.py`)

Each of the T frames is encoded independently by the shared InterFuser multi-view encoder. The resulting token sequences are concatenated along the sequence dimension and fed into the transformer encoder as a single, longer context.

```
Frame t-3 ──┐
Frame t-2 ──┤  Multi-view   ┌── concat ──►  Transformer  ──►  Waypoints
Frame t-1 ──┤   Encoder ×T  │               Encoder           Traffic
Frame t   ──┘               └──────────────────────────────►  Light State
                                                               Stop Sign
```

The temporal model shares the same dataset loader (`temporal/data/temporal_dataset.py`) and 5-head loss (`temporal/utils/losses.py`). Training is controlled by `temporal/train.py`.

---

## Experiments

All experiments use the same setup:
- **Dataset**: LMDrive subset — 300 routes, 60,755 frames from Towns 01–04, 6 weather conditions
- **Evaluation**: 10 held-out routes in Town05 (RouteScenario_16–25), never seen during training
- **Metrics**: Driving Score (DS = RC × IS), Route Completion (RC), Infraction Score (IS)
- **Frame stride**: 5 frames (0.5 s at 10 Hz) between temporal samples

### Closed-Loop Results on Town05 (10 routes)

| Model | T | Stride | Dropout | Avg DS | Avg RC | Avg IS | vs Baseline |
|-------|---|--------|---------|-------:|-------:|-------:|-------------|
| **Baseline (RES-001)** | 1 | — | — | **4.251** | 7.3% | 0.598 | — |
| T=2, d=0.3 | 2 | 5 | 0.3 | 5.672 | 28.6% | 0.294 | +33% |
| stride=1, d=0.3 | 4 | 1 | 0.3 | 6.473 | 20.3% | 0.346 | +52% |
| **T=4, d=0.3 (RES-004)** | 4 | 5 | 0.3 | **8.157** | 27.2% | 0.415 | **+92%** |

*DS = Driving Score (0–100). RC = Route Completion (%). IS = Infraction Score (penalty multiplier, 1.0 = no infractions).*

---

## Repository Structure

```
CARLA_TEMPO/
├── InterFuser/                      # Upstream baseline (git submodule)
│   └── interfuser/timm/models/      # Core InterFuser architecture (read-only)
├── temporal/                        # Our temporal extension
│   ├── models/
│   │   └── interfuser_temporal.py   # Temporal concat fusion
│   ├── data/
│   │   └── temporal_dataset.py      # TemporalWindowDataset + collate
│   ├── utils/
│   │   └── losses.py                # 5-head loss functions
│   ├── agents/                      # CARLA eval agents and configs
│   ├── train.py                     # Training entry point
│   └── eval.py                      # Offline evaluation
├── scripts/
│   ├── slurm/                       # SLURM job scripts (Amarel HPC)
│   ├── download_lmdrive.py          # Dataset download
│   └── plot_results.py              # Result visualization
├── configs/
│   └── temporal_t4.yaml             # Experiment config
├── figures/                         # Result plots and diagrams
├── RUNBOOK.md                       # Step-by-step reproduction guide
├── CHANGELOG.md                     # Experiment history
└── requirements.txt
```

---

## Reproducing Results

### Prerequisites

- Amarel HPC (or any Linux machine with SLURM + GPU)
- CARLA 0.9.10.1
- Conda environment: Python 3.8, PyTorch 2.1.0+cu121

```bash
# Clone repo with InterFuser submodule
git clone --recurse-submodules https://github.com/nikhilzero/CARLA_TEMPO.git
cd CARLA_TEMPO

# One-time environment setup (Amarel)
bash scripts/setup_amarel.sh

# Download dataset
sbatch scripts/download_lmdrive.sbatch
```

### Training

```bash
# Train baseline (InterFuser, 24 epochs)
sbatch scripts/slurm/baseline_train_research.sbatch

# Train temporal model — T=4, dropout=0.3 (RES-004 configuration)
sbatch scripts/slurm/temporal_train_research.sbatch
# Override dropout:  edit --dropout 0.3 in the sbatch script
```

### CARLA Closed-Loop Evaluation

```bash
# Evaluate baseline on Town05 routes 16–25
sbatch scripts/slurm/carla_eval_baseline_gpu.sbatch

# Evaluate RES-004 temporal model
sbatch scripts/slurm/carla_eval_temporal_reg_gpu.sbatch
```

See `RUNBOOK.md` for full step-by-step instructions, checkpoint paths, and result parsing.

---

## Citation / Acknowledgement

This work builds on **InterFuser** (Shao et al., CoRL 2022):

```bibtex
@inproceedings{shao2022interfuser,
  title     = {Safety-Enhanced Autonomous Driving Using Interpretable Sensor Fusion Transformer},
  author    = {Shao, Hao and Wang, Letian and Chen, Ruobing and Li, Hongsheng and Liu, Yu},
  booktitle = {Conference on Robot Learning (CoRL)},
  year      = {2022}
}
```

Dataset: **LMDrive** (Jia et al., NeurIPS 2023) — [HuggingFace](https://huggingface.co/datasets/opendilab/LMDrive).

**Thesis advisor**: Prof. Desmond Lun, Rutgers University–Camden.
