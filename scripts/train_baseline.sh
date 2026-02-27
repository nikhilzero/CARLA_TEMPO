#!/bin/bash
# =============================================================================
# train_baseline.sh — Reproduce InterFuser baseline on Amarel
# Usage: sbatch scripts/train_baseline.sh
# =============================================================================

#SBATCH --job-name=interfuser_baseline
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=logs/baseline_%j.out
#SBATCH --error=logs/baseline_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=nd967@rutgers.edu

set -e

PROJECT_DIR="/scratch/nd967/CARLA_TEMPO"
CONDA_ENV="interfuser"

# ---- Environment ----
module purge
module load cuda/12.1.0
source ~/miniconda3/bin/activate ${CONDA_ENV}

echo "Job ID: ${SLURM_JOB_ID}"
echo "Node:   $(hostname)"
echo "GPU:    $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "Python: $(python --version)"

# ---- Create log dir ----
mkdir -p ${PROJECT_DIR}/logs

# ---- Data paths ----
DATA_ROOT="/scratch/nd967/lmdrive_data"   # adjust if needed
TRAIN_DATA="${DATA_ROOT}/train"
VAL_DATA="${DATA_ROOT}/val"

# ---- Train ----
cd ${PROJECT_DIR}/InterFuser/interfuser

python train.py \
    --model interfuser_baseline \
    --data-dir ${TRAIN_DATA} \
    --val-data-dir ${VAL_DATA} \
    --output output \
    --batch-size 16 \
    --num-workers 8 \
    --epochs 30 \
    --lr 1e-4 \
    --img-size 224 \
    --log-interval 100 \
    "$@"
