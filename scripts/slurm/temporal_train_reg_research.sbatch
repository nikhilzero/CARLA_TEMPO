#!/bin/bash
# =============================================================================
# temporal_train_reg_research.sbatch — Approach 1 with stronger regularization
#
# Addresses the overfitting observed in RES-002 (best at epoch 1, epoch 24 worse
# than baseline on Town05). Interventions:
#   - dropout=0.3 (vs 0.1 in RES-002)
#   - weight_decay=0.1 (vs 0.05 in RES-002)
#   - temporal LR halved: 0.00025 (vs 0.0005 in RES-002)
#
# Research-scale: Town01-04 train, Town05 val, 6 weathers, T=4, stride=5.
#
# Submit: sbatch scripts/slurm/temporal_train_reg_research.sbatch
# =============================================================================

#SBATCH --job-name=temporal_reg
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=48:00:00
#SBATCH --requeue
#SBATCH --output=/scratch/nd967/CARLA_TEMPO/logs/temporal_reg_%j.out
#SBATCH --error=/scratch/nd967/CARLA_TEMPO/logs/temporal_reg_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=nd967@rutgers.edu

set -e

PROJECT_DIR="/scratch/nd967/CARLA_TEMPO"
DATA_ROOT="${DATA_ROOT:-${PROJECT_DIR}/InterFuser/dataset}"
OUTPUT_DIR="${PROJECT_DIR}/runs/temporal"

# RES-001 research-scale baseline checkpoint
BASELINE_CKPT="/scratch/nd967/CARLA_TEMPO/runs/baseline/20260307-082126-interfuser_baseline-224-baseline_research/model_best.pth.tar"

CONDA_ENV="interfuser"

echo "=========================================="
echo "Job ID:          ${SLURM_JOB_ID}"
echo "Node:            $(hostname)"
echo "GPU:             $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo unknown)"
echo "DATA_ROOT:       ${DATA_ROOT}"
echo "BASELINE_CKPT:   ${BASELINE_CKPT}"
echo "Run:             temporal_reg_research (Approach 1 + dropout=0.3, wd=0.1, lr=0.00025)"
echo "Motivation:      Address RES-002 overfitting (epoch1 best, epoch24 worse than baseline)"
echo "=========================================="

module purge
module load cuda/12.1.0
source ~/miniconda3/bin/activate ${CONDA_ENV}

if [ ! -f "${DATA_ROOT}/dataset_index.txt" ]; then
    echo "ERROR: dataset_index.txt not found."
    exit 1
fi

if [ ! -f "${BASELINE_CKPT}" ]; then
    echo "ERROR: Baseline checkpoint not found at ${BASELINE_CKPT}"
    exit 1
fi

# Auto-resume if a previous run exists
RESUME_ARG=""
LATEST_RUN=$(ls -td ${OUTPUT_DIR}/*temporal_reg_research_T4_s5* 2>/dev/null | head -1)
if [ -n "${LATEST_RUN}" ] && [ -f "${LATEST_RUN}/last.pth.tar" ]; then
    RESUME_ARG="--resume ${LATEST_RUN}/last.pth.tar"
    echo "Resuming from: ${LATEST_RUN}/last.pth.tar"
else
    echo "No checkpoint found — starting fresh."
fi

mkdir -p "${OUTPUT_DIR}" "${PROJECT_DIR}/logs"

cd ${PROJECT_DIR}

python -m temporal.train \
    --model-type concat \
    --data-dir "${DATA_ROOT}" \
    --train-towns 1 2 3 4 \
    --val-towns 5 \
    --train-weathers 1 3 6 8 14 18 \
    --val-weathers 1 3 6 8 14 18 \
    --temporal-frames 4 \
    --frame-stride 5 \
    --temporal-depth 2 \
    --dropout 0.3 \
    --epochs 24 \
    --warmup-epochs 2 \
    --batch-size 2 \
    --grad-accum 2 \
    --lr 0.00025 \
    --backbone-lr 0.0001 \
    --weight-decay 0.1 \
    --clip-grad 10 \
    --workers 8 \
    --log-interval 50 \
    --save-interval 1 \
    --experiment temporal_reg_research_T4_s5 \
    --output "${OUTPUT_DIR}" \
    --pretrained-backbone "${BASELINE_CKPT}" \
    ${RESUME_ARG}

echo "=========================================="
echo "Temporal regularization research training DONE."
echo "Outputs in ${OUTPUT_DIR}/"
echo "=========================================="
