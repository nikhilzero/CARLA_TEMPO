# CARLA_TEMPO — Temporal Extension for InterFuser

Master's thesis (Rutgers University - Camden): extending the [InterFuser](https://github.com/opendilab/InterFuser) autonomous driving model with temporal reasoning — multi-frame input instead of single snapshots.

## Repository Structure

```
CARLA_TEMPO/
├── InterFuser/          # Cloned upstream baseline (opendilab/InterFuser)
├── temporal/            # Our temporal extension (main work area)
│   ├── models/          # Temporal model variants
│   ├── data/            # Temporal data loaders
│   ├── utils/           # Shared helpers
│   └── train.py         # Training entry point
├── scripts/             # SLURM job scripts for Amarel HPC
│   ├── setup_amarel.sh
│   ├── train_baseline.sh
│   └── train_temporal.sh
├── configs/             # YAML experiment configs
│   └── temporal_t4.yaml
├── CLAUDE.md            # Context for Claude Code
└── requirements.txt
```

## Quick Start (Amarel)

```bash
# One-time setup
cd /scratch/nd967/CARLA_TEMPO
bash scripts/setup_amarel.sh

# Train temporal model (T=4 frames)
sbatch scripts/train_temporal.sh
```

## Sync Workflow

```bash
# Local → Amarel
git add . && git commit -m "your message" && git push

# Amarel
cd /scratch/nd967/CARLA_TEMPO && git pull && sbatch scripts/train_temporal.sh
```

## Temporal Approach

**Approach 1 (implemented)**: Concat fusion — encode each of T frames independently, concatenate all tokens, feed into the transformer encoder.

**Approach 2 (planned)**: Temporal attention module between frame encoder and fusion transformer.

See `CLAUDE.md` for full project context.
