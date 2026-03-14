#!/bin/bash
# =============================================================================
# eval_baseline.sbatch — Evaluate RES-001 baseline on held-out Town05.
# Produces waypoint_l1, waypoint_l2, and head losses for thesis comparison.
#
# Usage:
#   sbatch scripts/slurm/eval_baseline.sbatch
#   CKPT=/path/to/model_best.pth.tar sbatch scripts/slurm/eval_baseline.sbatch
# =============================================================================

#SBATCH --job-name=eval_baseline
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=/scratch/nd967/CARLA_TEMPO/logs/eval_baseline_%j.out
#SBATCH --error=/scratch/nd967/CARLA_TEMPO/logs/eval_baseline_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=nd967@rutgers.edu

set -e

PROJECT_DIR="/scratch/nd967/CARLA_TEMPO"
DATA_ROOT="${DATA_ROOT:-${PROJECT_DIR}/InterFuser/dataset}"
CONDA_ENV="interfuser"

# RES-001: research-scale baseline checkpoint
CKPT="${CKPT:-/scratch/nd967/CARLA_TEMPO/runs/baseline/20260307-082126-interfuser_baseline-224-baseline_research/model_best.pth.tar}"

echo "=========================================="
echo "EVAL: Baseline InterFuser (RES-001)"
echo "Job ID:    ${SLURM_JOB_ID}"
echo "Node:      $(hostname)"
echo "GPU:       $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo unknown)"
echo "CKPT:      ${CKPT}"
echo "DATA_ROOT: ${DATA_ROOT}"
echo "Val:       Town05 (held-out), all 6 weathers"
echo "=========================================="

module purge
module load cuda/12.1.0
source ~/miniconda3/bin/activate ${CONDA_ENV}

if [ ! -f "${DATA_ROOT}/dataset_index.txt" ]; then
    echo "ERROR: dataset_index.txt not found at ${DATA_ROOT}"
    exit 1
fi

if [ ! -f "${CKPT}" ]; then
    echo "ERROR: checkpoint not found: ${CKPT}"
    exit 1
fi

cd ${PROJECT_DIR}

python -m temporal.eval_baseline \
    --data-dir "${DATA_ROOT}" \
    --checkpoint "${CKPT}" \
    --val-towns 5 \
    --val-weathers 1 3 6 8 14 18 \
    --batch-size 4 \
    --workers 4

echo "=========================================="
echo "Baseline eval DONE. JSON saved next to checkpoint."
echo "=========================================="
