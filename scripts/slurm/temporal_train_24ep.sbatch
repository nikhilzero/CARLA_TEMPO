#!/bin/bash
# =============================================================================
# temporal_train_24ep.sbatch — Extended T=4 run (24 epochs).
# Use when the 12-epoch run shows val_loss still decreasing at the final epoch.
# Can resume from the 12-epoch checkpoint via --resume.
# Submit: sbatch scripts/slurm/temporal_train_24ep.sbatch
# =============================================================================

#SBATCH --job-name=temporal_24ep
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --output=/scratch/nd967/CARLA_TEMPO/logs/temporal_24ep_%j.out
#SBATCH --error=/scratch/nd967/CARLA_TEMPO/logs/temporal_24ep_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=nd967@rutgers.edu

set -e

PROJECT_DIR="/scratch/nd967/CARLA_TEMPO"
DATA_ROOT="${DATA_ROOT:-${PROJECT_DIR}/InterFuser/dataset}"
OUTPUT_DIR="${PROJECT_DIR}/runs/temporal"
BASELINE_CKPT="/scratch/nd967/interfuser_project/InterFuser/interfuser/output/20260210-110341-interfuser_baseline-224-real_data_test/model_best.pth.tar"
CONDA_ENV="interfuser"

# Set RESUME_CKPT to resume from existing 12-epoch run, or leave blank to start fresh.
# Example: RESUME_CKPT=/scratch/nd967/CARLA_TEMPO/runs/temporal/.../last.pth.tar
RESUME_CKPT="${RESUME_CKPT:-}"

echo "=========================================="
echo "EXTENDED TRAINING: T=4, 24 epochs"
echo "Job ID:      ${SLURM_JOB_ID}"
echo "Node:        $(hostname)"
echo "GPU:         $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo unknown)"
echo "DATA_ROOT:   ${DATA_ROOT}"
echo "RESUME_CKPT: ${RESUME_CKPT:-none}"
echo "=========================================="

module purge
module load cuda/12.1.0
source ~/miniconda3/bin/activate ${CONDA_ENV}

if [ ! -f "${DATA_ROOT}/dataset_index.txt" ]; then
    echo "ERROR: dataset_index.txt not found at ${DATA_ROOT}"
    exit 1
fi

if [ ! -f "${BASELINE_CKPT}" ]; then
    echo "WARNING: baseline checkpoint not found — training from scratch"
    PRETRAINED_ARG=""
else
    PRETRAINED_ARG="--pretrained-backbone ${BASELINE_CKPT}"
fi

if [ -n "${RESUME_CKPT}" ] && [ -f "${RESUME_CKPT}" ]; then
    RESUME_ARG="--resume ${RESUME_CKPT}"
    echo "Resuming from: ${RESUME_CKPT}"
else
    RESUME_ARG=""
    echo "Starting fresh (no resume checkpoint)"
fi

mkdir -p "${OUTPUT_DIR}"
mkdir -p "${PROJECT_DIR}/logs"

cd ${PROJECT_DIR}

python -m temporal.train \
    --data-dir "${DATA_ROOT}" \
    --train-towns 1 --val-towns 1 \
    --train-weathers 18 --val-weathers 18 \
    --temporal-frames 4 \
    --frame-stride 1 \
    --temporal-depth 2 \
    --epochs 24 \
    --warmup-epochs 1 \
    --batch-size 2 \
    --grad-accum 2 \
    --lr 0.0005 \
    --backbone-lr 0.0002 \
    --weight-decay 0.05 \
    --clip-grad 10 \
    --workers 4 \
    --log-interval 10 \
    --save-interval 2 \
    --experiment temporal_24ep \
    --output "${OUTPUT_DIR}" \
    ${PRETRAINED_ARG} \
    ${RESUME_ARG}

echo "=========================================="
echo "Extended training DONE. Outputs in ${OUTPUT_DIR}/"
echo "=========================================="
