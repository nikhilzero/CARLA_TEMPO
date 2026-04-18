# CARLA_TEMPO — Temporal Reasoning for Autonomous Driving

Master's thesis, Rutgers University–Camden (2026).
Extends [InterFuser](https://github.com/opendilab/InterFuser) (Shao et al., CoRL 2022) with multi-frame temporal reasoning and evaluates it in closed-loop CARLA simulation.

---

## Motivation

Transformer-based autonomous driving models such as InterFuser process a **single snapshot** of sensor input at each timestep. This makes it impossible for the model to directly perceive motion, velocity, or the intent of surrounding agents — information that human drivers routinely use. A vehicle that was stationary one second ago but is now accelerating presents a fundamentally different situation than one that has been stationary for ten seconds.

**This work adds temporal reasoning** by feeding T consecutive frames into the model, allowing the transformer to attend across time and encode motion context. We evaluate whether temporal history improves closed-loop driving performance on held-out Town05 routes in CARLA.

---

## Architecture

CARLA_TEMPO wraps InterFuser with a temporal encoder that processes a sliding window of T frames. Two approaches are implemented and evaluated:

### Approach 1 — Temporal Concat Fusion (`temporal/models/interfuser_temporal.py`)

Each of the T frames is encoded independently by the shared InterFuser multi-view encoder. The resulting token sequences are concatenated along the sequence dimension and fed into the transformer encoder as a single, longer context.

```
Frame t-3 ──┐
Frame t-2 ──┤  Multi-view   ┌── concat ──►  Transformer  ──►  Waypoints
Frame t-1 ──┤   Encoder ×T  │               Encoder           Traffic
Frame t   ──┘               └──────────────────────────────►  Light State
                                                               Stop Sign
```

### Approach 2 — Cross-Frame Attention (`temporal/models/interfuser_temporal_attn.py`)

A dedicated temporal cross-attention module is inserted between the per-frame encoder and the fusion transformer. The current frame's tokens attend to past frame tokens before entering the main encoder.

Both approaches share the same dataset loader (`temporal/data/temporal_dataset.py`) and 5-head loss (`temporal/utils/losses.py`). Training is controlled by `temporal/train.py` with `--model-type {concat,crossattn}`.

---

## Experiments

All experiments use the same setup:
- **Dataset**: LMDrive subset — 300 routes, 60,755 frames from Towns 01–04, 6 weather conditions
- **Evaluation**: 10 held-out routes in Town05 (RouteScenario_16–25), never seen during training
- **Metrics**: Driving Score (DS = RC × IS), Route Completion (RC), Infraction Score (IS)
- **Frame stride**: 5 frames (0.5 s at 10 Hz) between temporal samples

### Closed-Loop Results on Town05 (10 routes)

| Model | T | Dropout | Avg DS | Avg RC | Avg IS | vs Baseline |
|-------|---|---------|-------:|-------:|-------:|-------------|
| **Baseline (RES-001)** | 1 | — | **4.251** | 7.3% | 0.598 | — |
| Temporal T=4 (RES-002) | 4 | 0.1 | 0.000 | 0.0% | 1.000 | −100% |
| **Temporal T=4 + Dropout (RES-004)** | 4 | 0.3 | **8.157** | 27.2% | 0.415 | **+92%** |

*DS = Driving Score (0–100). RC = Route Completion (%). IS = Infraction Score (penalty multiplier, 1.0 = no infractions).*

### Per-Route Breakdown — Baseline vs. Best Model (RES-004)

| Route | Baseline DS | RES-004 DS |
|-------|------------:|-----------:|
| S16 | 0.406 | 9.215 |
| S17 | 4.910 | 7.578 |
| S18 | 2.788 | 3.035 |
| S19 | 6.192 | 11.291 |
| S20 | 5.960 | 0.034 |
| S21 | 11.864 | **24.393** |
| S22 | 1.806 | 5.997 |
| S23 | 0.397 | 10.393 |
| S24 | 6.291 | 5.801 |
| S25 | 1.898 | 3.833 |
| **Avg** | **4.251** | **8.157** |

---

## Key Findings

1. **Temporal context improves driving by +92%** (Driving Score 4.251 → 8.157) when regularized with dropout=0.3.

2. **Dropout is critical.** Without it (RES-002, dropout=0.1), the model's traffic-light head becomes overconfident and outputs a constant "red" signal, causing the vehicle to never move. The waypoint predictions are geometrically valid; the failure is entirely in the auxiliary head. Dropout=0.3 breaks this degeneracy.

3. **More frames help.** Ablations show T=4 (RES-004, +92%) outperforms T=2 (+33%) under identical regularization, confirming that temporal context is the causal factor.

4. **Cross-attention (Approach 2) requires stronger regularization.** The cross-attention model with dropout=0.1 also produces DS=0.000, consistent with the dropout finding rather than an architectural failure.

---

## Repository Structure

```
CARLA_TEMPO/
├── InterFuser/                      # Upstream baseline (git submodule)
│   └── interfuser/timm/models/      # Core InterFuser architecture (read-only)
├── temporal/                        # Our temporal extension
│   ├── models/
│   │   ├── interfuser_temporal.py   # Approach 1: concat fusion
│   │   └── interfuser_temporal_attn.py  # Approach 2: cross-attention
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

**Thesis advisor**: Prof. Lun Li, Rutgers University–Camden.
