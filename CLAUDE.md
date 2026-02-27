# CARLA_TEMPO — Project Context for Claude Code

## Project Summary
Master's thesis at Rutgers University - Camden.
Goal: Extend InterFuser (autonomous driving transformer) with **temporal reasoning** — processing multiple consecutive frames instead of single snapshots.

## Repository Layout
```
CARLA_TEMPO/
├── CLAUDE.md                        # This file
├── README.md
├── InterFuser/                      # Cloned upstream baseline (opendilab/InterFuser)
│   ├── interfuser/                  # Main package
│   │   ├── timm/models/interfuser.py   # ← Core model architecture
│   │   ├── train.py                    # ← Training entry point
│   │   └── scripts/train.sh
│   ├── dataset/                     # Data utilities
│   └── leaderboard/                 # CARLA eval harness
├── temporal/                        # OUR temporal extension (main work area)
│   ├── models/                      # Temporal model variants
│   ├── data/                        # Temporal data loaders
│   └── utils/                       # Shared helpers
├── scripts/                         # SLURM job scripts for Amarel HPC
│   ├── setup_amarel.sh              # One-time environment setup
│   ├── train_baseline.sh            # Reproduce InterFuser baseline
│   └── train_temporal.sh            # Train temporal model
└── configs/                         # YAML experiment configs
```

## Key Files to Know
- **Core model**: `InterFuser/interfuser/timm/models/interfuser.py`
- **Training script**: `InterFuser/interfuser/train.py`
- **Our models go in**: `temporal/models/`
- **Our data loaders go in**: `temporal/data/`

## Amarel HPC Details
- **Login**: `nd967@amarel.rutgers.edu`
- **Working dir**: `/scratch/nd967/CARLA_TEMPO`
- **GPU partition**: `--partition=gpu --gres=gpu:1`
- **CUDA module**: `cuda/12.1.0`
- **Conda env**: `interfuser` (Python 3.8, PyTorch 2.1.0+cu121, CUDA 12.1)
- **Conda activate**: `source ~/miniconda3/bin/activate interfuser`

## Baseline Already Trained on Amarel
- Trained InterFuser for 12 epochs on real CARLA data (LMDrive dataset from HuggingFace)
- Checkpoint: `/scratch/nd967/interfuser_project/InterFuser/interfuser/output/20260210-110341-interfuser_baseline-224-real_data_test/`
- Data converted from `rgb_full` → separate `rgb_front/left/right/rear` channels
- Known bug fixes in `train.py`: distributed training checks, validation loss variables

## Git / Sync Workflow
```bash
# Local (this machine)
git add . && git commit -m "message" && git push

# On Amarel
cd /scratch/nd967/CARLA_TEMPO && git pull && sbatch scripts/train_temporal.sh
```
GitHub: https://github.com/nikhilzero/CARLA_TEMPO.git

## Temporal Extension Plan
The core idea: feed T consecutive frames (e.g., T=4) into the model so it can learn motion, velocity, and intent from the temporal context.

**Approach 1 (start here)**: Modify the tokenizer to concatenate multi-frame features before the transformer encoder. Simple, low-risk.

**Approach 2**: Add a temporal attention module between the frame encoder and the fusion transformer. More expressive.

**Key questions to answer**:
- How many frames T? (start with T=4 at 2 Hz → 2 seconds of history)
- Frame stride? (every frame, every other, etc.)
- How to handle the dataset (LMDrive has sequential frames)?

## Development Conventions
- Python 3.8, PyTorch 2.1.0
- Keep temporal modifications isolated in `temporal/` — don't modify `InterFuser/` unless fixing a known bug
- SLURM scripts use `#SBATCH` directives for Amarel gpu partition
- Config files use YAML
