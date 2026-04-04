#!/bin/bash
# =============================================================================
# ablation_stride1_research.sbatch — Research-scale ablation: T=4, stride=1
#
# Tests whether denser sampling (every frame at 10Hz vs every 5th frame)
# helps generalization to Town05. With stride=1, T=4 covers 0.4 seconds
# (vs 2.0 seconds at stride=5). Different temporal context, not better/worse a priori.
#
# Research-scale: Town01-04 train, Town05 val, 6 weathers.
#
# Submit: sbatch scripts/slurm/ablation_stride1_research.sbatch
# =============================================================================

#SBATCH --job-name=abl_s1_d03
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=48:00:00
#SBATCH --requeue
#SBATCH --output=/scratch/nd967/CARLA_TEMPO/logs/ablation_stride1_research_d03_%j.out
#SBATCH --error=/scratch/nd967/CARLA_TEMPO/logs/ablation_stride1_research_d03_%j.err
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
echo "Ablation:     T=4, stride=1 (dense, 0.4-sec context at 10Hz; research scale)"
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
LATEST_RUN=$(ls -td ${OUTPUT_DIR}/*ablation_T4_stride1_research_d03* 2>/dev/null | head -1)
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
    --frame-stride 1 \
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
    --experiment ablation_T4_stride1_research_d03 \
    --output "${OUTPUT_DIR}" \
    --pretrained-backbone "${BASELINE_CKPT}" \
    ${RESUME_ARG}

echo "Ablation T=4 stride=1 research DONE. Outputs: ${OUTPUT_DIR}/"
