#!/bin/bash
# =============================================================================
# temporal_train_res005.sbatch — RES-005: T=4, d=0.3, rebalanced loss weights
#
# Hypothesis: RES-004 traffic head overfits (weight=0.5), producing phantom
# vehicle detections that cause chronic braking. Rebalancing loss weights
# (traffic↓, waypoints↑, tl↑) should reduce false positives and improve
# traffic-light state accuracy.
#
# Changes vs RES-004:
#   --w-traffic   0.25  (was 0.50) — less emphasis on BEV vehicle detection
#   --w-waypoints 0.45  (was 0.20) — prioritise trajectory quality
#   --w-tl        0.20  (was 0.10) — improve tl_state head calibration
#   --w-junction  0.05  (unchanged)
#   --w-stop      0.05  (was 0.01) — slight increase for safety
#   --dropout     0.3   (same as RES-004)
#   --epochs      24    (same)
#
# Submit: sbatch scripts/slurm/temporal_train_res005.sbatch
# =============================================================================

#SBATCH --job-name=res005
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=72:00:00
#SBATCH --requeue
#SBATCH --output=/scratch/nd967/CARLA_TEMPO/logs/res005_%j.out
#SBATCH --error=/scratch/nd967/CARLA_TEMPO/logs/res005_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=nd967@rutgers.edu
#SBATCH --exclude=gpu005,gpu006,gpu018,gpu019,gpu024,gpu029,gpu030

set -e

PROJECT_DIR="/scratch/nd967/CARLA_TEMPO"
DATA_ROOT="${DATA_ROOT:-${PROJECT_DIR}/InterFuser/dataset}"
OUTPUT_DIR="${PROJECT_DIR}/runs/temporal"

BASELINE_CKPT="/scratch/nd967/CARLA_TEMPO/runs/baseline/20260307-082126-interfuser_baseline-224-baseline_research/model_best.pth.tar"
CONDA_ENV="interfuser"

echo "=========================================="
echo "Job ID:    ${SLURM_JOB_ID}"
echo "Node:      $(hostname)"
echo "GPU:       $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo unknown)"
echo "Exp:       RES-005 (T=4, d=0.3, rebalanced loss)"
echo "=========================================="

module purge
module load cuda/12.1.0
source ~/miniconda3/bin/activate ${CONDA_ENV}

if [ ! -f "${DATA_ROOT}/dataset_index.txt" ]; then
    echo "ERROR: dataset_index.txt not found at ${DATA_ROOT}"; exit 1
fi
if [ ! -f "${BASELINE_CKPT}" ]; then
    echo "ERROR: Baseline checkpoint not found"; exit 1
fi

# Auto-resume from most recent res005 run
RESUME_ARG=""
LATEST_RUN=$(ls -td ${OUTPUT_DIR}/*res005* 2>/dev/null | head -1)
if [ -n "${LATEST_RUN}" ] && [ -f "${LATEST_RUN}/last.pth.tar" ]; then
    RESUME_ARG="--resume ${LATEST_RUN}/last.pth.tar"
    echo "Resuming from: ${LATEST_RUN}/last.pth.tar"
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
    --lr 0.0005 \
    --backbone-lr 0.0002 \
    --weight-decay 0.05 \
    --clip-grad 10 \
    --workers 8 \
    --log-interval 50 \
    --save-interval 1 \
    --w-traffic   0.25 \
    --w-waypoints 0.45 \
    --w-junction  0.05 \
    --w-tl        0.20 \
    --w-stop      0.05 \
    --experiment res005_T4_d03_rebalanced \
    --output "${OUTPUT_DIR}" \
    --pretrained-backbone "${BASELINE_CKPT}" \
    ${RESUME_ARG}

echo "RES-005 training DONE. Outputs: ${OUTPUT_DIR}/"
