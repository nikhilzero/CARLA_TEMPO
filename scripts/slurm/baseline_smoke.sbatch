#!/bin/bash
# =============================================================================
# baseline_smoke.sbatch — 1-epoch smoke test for InterFuser baseline
# Submit from repo root: sbatch scripts/slurm/baseline_smoke.sbatch
# =============================================================================

#SBATCH --job-name=baseline_smoke
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=/scratch/nd967/CARLA_TEMPO/logs/baseline_smoke_%j.out
#SBATCH --error=/scratch/nd967/CARLA_TEMPO/logs/baseline_smoke_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=nd967@rutgers.edu

set -e

PROJECT_DIR="/scratch/nd967/CARLA_TEMPO"
DATA_ROOT="${DATA_ROOT:-${PROJECT_DIR}/InterFuser/dataset}"
OUTPUT_DIR="${PROJECT_DIR}/runs/baseline"
CONDA_ENV="interfuser"

echo "=========================================="
echo "Job ID:     ${SLURM_JOB_ID}"
echo "Node:       $(hostname)"
echo "GPU:        $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo unknown)"
echo "Python:     $(python --version 2>&1)"
echo "DATA_ROOT:  ${DATA_ROOT}"
echo "OUTPUT_DIR: ${OUTPUT_DIR}"
echo "=========================================="

# ---- Environment ----
module purge
module load cuda/12.1.0
source ~/miniconda3/bin/activate ${CONDA_ENV}

# ---- Sanity: confirm data exists ----
if [ ! -f "${DATA_ROOT}/dataset_index.txt" ]; then
    echo "ERROR: dataset_index.txt not found at ${DATA_ROOT}"
    exit 1
fi
echo "dataset_index.txt: $(cat ${DATA_ROOT}/dataset_index.txt)"

# ---- Create output dirs ----
mkdir -p "${OUTPUT_DIR}"
mkdir -p "${PROJECT_DIR}/logs"

# ---- Train (1 epoch smoke) ----
cd ${PROJECT_DIR}/InterFuser/interfuser

torchrun --nproc_per_node=1 --master_port=29501 train.py \
    "${DATA_ROOT}" \
    --dataset carla \
    --train-towns 1 --val-towns 1 \
    --train-weathers 18 --val-weathers 18 \
    --model interfuser_baseline \
    --sched cosine --epochs 1 --warmup-epochs 0 --lr 0.0005 --batch-size 4 \
    --no-prefetcher --eval-metric l1_error \
    --opt adamw --opt-eps 1e-8 --weight-decay 0.05 \
    --scale 0.9 1.1 --saver-decreasing --clip-grad 10 --freeze-num -1 \
    --with-backbone-lr --backbone-lr 0.0002 \
    --multi-view --with-lidar --multi-view-input-size 3 128 128 \
    --experiment baseline_smoke \
    --output "${OUTPUT_DIR}" \
    --workers 4 \
    --pretrained

echo "=========================================="
echo "Smoke test DONE. Check ${OUTPUT_DIR}/ for checkpoint."
echo "=========================================="
