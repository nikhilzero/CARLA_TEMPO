#!/bin/bash
# =============================================================================
# ablation_T2.sbatch — Temporal window ablation: T=2 (1 second history @ 2 Hz)
# Submit: sbatch scripts/slurm/ablation_T2.sbatch
# =============================================================================

#SBATCH --job-name=temporal_T2
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/nd967/CARLA_TEMPO/logs/ablation_T2_%j.out
#SBATCH --error=/scratch/nd967/CARLA_TEMPO/logs/ablation_T2_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=nd967@rutgers.edu

set -e

PROJECT_DIR="/scratch/nd967/CARLA_TEMPO"
DATA_ROOT="${DATA_ROOT:-${PROJECT_DIR}/InterFuser/dataset}"
OUTPUT_DIR="${PROJECT_DIR}/runs/ablations"
BASELINE_CKPT="/scratch/nd967/interfuser_project/InterFuser/interfuser/output/20260210-110341-interfuser_baseline-224-real_data_test/model_best.pth.tar"
CONDA_ENV="interfuser"

echo "=========================================="
echo "ABLATION: T=2 (1 second history)"
echo "Job ID:    ${SLURM_JOB_ID}"
echo "Node:      $(hostname)"
echo "GPU:       $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo unknown)"
echo "DATA_ROOT: ${DATA_ROOT}"
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

mkdir -p "${OUTPUT_DIR}"
mkdir -p "${PROJECT_DIR}/logs"

cd ${PROJECT_DIR}

python -m temporal.train \
    --data-dir "${DATA_ROOT}" \
    --train-towns 1 --val-towns 1 \
    --train-weathers 18 --val-weathers 18 \
    --temporal-frames 2 \
    --frame-stride 1 \
    --temporal-depth 2 \
    --epochs 12 \
    --warmup-epochs 1 \
    --batch-size 2 \
    --grad-accum 2 \
    --lr 0.0005 \
    --backbone-lr 0.0002 \
    --weight-decay 0.05 \
    --clip-grad 10 \
    --workers 4 \
    --log-interval 10 \
    --save-interval 1 \
    --experiment temporal_ablation_T2 \
    --output "${OUTPUT_DIR}" \
    ${PRETRAINED_ARG}

echo "=========================================="
echo "Ablation T=2 DONE. Outputs in ${OUTPUT_DIR}/"
echo "=========================================="
