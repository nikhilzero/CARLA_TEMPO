#!/bin/bash
# =============================================================================
# train_temporal.sh — Train temporal extension of InterFuser on Amarel
# Usage: sbatch scripts/train_temporal.sh
# =============================================================================

#SBATCH --job-name=interfuser_temporal
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=48:00:00
#SBATCH --output=logs/temporal_%j.out
#SBATCH --error=logs/temporal_%j.err
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
DATA_ROOT="/scratch/nd967/lmdrive_data"
TRAIN_DATA="${DATA_ROOT}/train"
VAL_DATA="${DATA_ROOT}/val"

# ---- Temporal config ----
TEMPORAL_FRAMES=4       # Number of consecutive frames to use
FRAME_STRIDE=1          # Sample every Nth frame (1 = every frame at ~2 Hz = 2 sec history)
CONFIG="${PROJECT_DIR}/configs/temporal_t4.yaml"

# ---- Pretrained baseline weights (optional warmstart) ----
BASELINE_CKPT="/scratch/nd967/interfuser_project/InterFuser/interfuser/output/20260210-110341-interfuser_baseline-224-real_data_test/model_best.pth.tar"

# ---- Train ----
cd ${PROJECT_DIR}

python -m temporal.train \
    --config ${CONFIG} \
    --data-dir ${TRAIN_DATA} \
    --val-data-dir ${VAL_DATA} \
    --output output/temporal \
    --batch-size 8 \
    --num-workers 8 \
    --epochs 30 \
    --lr 1e-4 \
    --temporal-frames ${TEMPORAL_FRAMES} \
    --frame-stride ${FRAME_STRIDE} \
    --pretrained-backbone ${BASELINE_CKPT} \
    "$@"
