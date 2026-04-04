#!/bin/bash
# =============================================================================
# ablation_T2_research.sbatch — Research-scale ablation: T=2 (1-second context)
#
# Research-scale: Town01-04 train, Town05 val, 6 weathers, stride=5.
# Paired with RES-002 (T=4, stride=5) and ablation_T8_research to form the
# thesis ablation table on research-scale data.
#
# Submit: sbatch scripts/slurm/ablation_T2_research.sbatch
# =============================================================================

#SBATCH --job-name=abl_T2_d03
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=36:00:00
#SBATCH --requeue
#SBATCH --output=/scratch/nd967/CARLA_TEMPO/logs/ablation_T2_research_d03_%j.out
#SBATCH --error=/scratch/nd967/CARLA_TEMPO/logs/ablation_T2_research_d03_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=nd967@rutgers.edu

set -e

PROJECT_DIR="/scratch/nd967/CARLA_TEMPO"
DATA_ROOT="${DATA_ROOT:-${PROJECT_DIR}/InterFuser/dataset}"
OUTPUT_DIR="${PROJECT_DIR}/runs/ablations"

BASELINE_CKPT="/scratch/nd967/CARLA_TEMPO/runs/baseline/20260307-082126-interfuser_baseline-224-baseline_research/model_best.pth.tar"

CONDA_ENV="interfuser"

echo "=========================================="
echo "Job ID:       ${SLURM_JOB_ID}"
echo "Node:         $(hostname)"
echo "GPU:          $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo unknown)"
echo "Ablation:     T=2, stride=5 (research scale)"
echo "=========================================="

module purge
module load cuda/12.1.0
source ~/miniconda3/bin/activate ${CONDA_ENV}

if [ ! -f "${DATA_ROOT}/dataset_index.txt" ]; then
    echo "ERROR: dataset_index.txt not found."
    exit 1
fi

if [ ! -f "${BASELINE_CKPT}" ]; then
    echo "ERROR: Baseline checkpoint not found."
    exit 1
fi

RESUME_ARG=""
LATEST_RUN=$(ls -td ${OUTPUT_DIR}/*ablation_T2_research* 2>/dev/null | head -1)
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
    --temporal-frames 2 \
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
    --experiment ablation_T2_research_d03 \
    --output "${OUTPUT_DIR}" \
    --pretrained-backbone "${BASELINE_CKPT}" \
    ${RESUME_ARG}

echo "Ablation T=2 research DONE. Outputs: ${OUTPUT_DIR}/"
