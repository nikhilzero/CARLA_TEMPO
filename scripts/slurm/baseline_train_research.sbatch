#!/bin/bash
# =============================================================================
# baseline_train_research.sbatch — Thesis-quality baseline training
#
# Research-scale: Town01-04 train, Town05 val, 6 weather conditions.
# This produces the baseline numbers reported in the thesis.
#
# Prerequisites:
#   1. sbatch scripts/download_lmdrive.sbatch (and wait for completion)
#   2. Confirm dataset_index.txt has Town01-05 entries
#
# Submit: sbatch scripts/slurm/baseline_train_research.sbatch
# =============================================================================

#SBATCH --job-name=baseline_research
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=24:00:00
#SBATCH --output=/scratch/nd967/CARLA_TEMPO/logs/baseline_research_%j.out
#SBATCH --error=/scratch/nd967/CARLA_TEMPO/logs/baseline_research_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
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
echo "DATA_ROOT:  ${DATA_ROOT}"
echo "Run:        baseline_research (Town01-04 train, Town05 val, 6 weathers)"
echo "=========================================="

module purge
module load cuda/12.1.0
source ~/miniconda3/bin/activate ${CONDA_ENV}

if [ ! -f "${DATA_ROOT}/dataset_index.txt" ]; then
    echo "ERROR: dataset_index.txt not found. Run download_lmdrive.sbatch first."
    exit 1
fi

# Verify multi-town data is present
for town in 01 02 03 04 05; do
    count=$(grep -c "town${town}" "${DATA_ROOT}/dataset_index.txt" || true)
    echo "  Town${town}: ${count} routes in index"
done

mkdir -p "${OUTPUT_DIR}" "${PROJECT_DIR}/logs"

cd ${PROJECT_DIR}/InterFuser/interfuser

torchrun --nproc_per_node=1 --master_port=29500 train.py \
    "${DATA_ROOT}" \
    --dataset carla \
    --train-towns 1 2 3 4 \
    --val-towns 5 \
    --train-weathers 1 3 6 8 14 18 \
    --val-weathers 1 3 6 8 14 18 \
    --model interfuser_baseline \
    --sched cosine --epochs 24 --warmup-epochs 2 --lr 0.0005 --batch-size 4 \
    --no-prefetcher --eval-metric l1_error \
    --opt adamw --opt-eps 1e-8 --weight-decay 0.05 \
    --scale 0.9 1.1 --saver-decreasing --clip-grad 10 --freeze-num -1 \
    --with-backbone-lr --backbone-lr 0.0002 \
    --multi-view --with-lidar --multi-view-input-size 3 128 128 \
    --experiment baseline_research \
    --output "${OUTPUT_DIR}" \
    --workers 8 \
    --pretrained

echo "=========================================="
echo "Baseline research training DONE."
echo "Outputs in ${OUTPUT_DIR}/"
echo "=========================================="
