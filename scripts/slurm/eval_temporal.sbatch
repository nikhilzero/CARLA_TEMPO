#!/bin/bash
# =============================================================================
# eval_temporal.sbatch — Run temporal/eval.py on a saved checkpoint.
#
# Usage:
#   sbatch scripts/slurm/eval_temporal.sbatch
#
# Override checkpoint path:
#   CKPT=/path/to/model_best.pth.tar sbatch scripts/slurm/eval_temporal.sbatch
# =============================================================================

#SBATCH --job-name=eval_temporal
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=/scratch/nd967/CARLA_TEMPO/logs/eval_temporal_%j.out
#SBATCH --error=/scratch/nd967/CARLA_TEMPO/logs/eval_temporal_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=nd967@rutgers.edu

set -e

PROJECT_DIR="/scratch/nd967/CARLA_TEMPO"
DATA_ROOT="${DATA_ROOT:-${PROJECT_DIR}/InterFuser/dataset}"
CONDA_ENV="interfuser"

# Default: best checkpoint from the T=4 full run. Override with CKPT env var.
CKPT="${CKPT:-${PROJECT_DIR}/runs/temporal/20260301-102924-temporal_full-T4/model_best.pth.tar}"

echo "=========================================="
echo "EVAL: InterFuserTemporal"
echo "Job ID:    ${SLURM_JOB_ID}"
echo "Node:      $(hostname)"
echo "GPU:       $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo unknown)"
echo "CKPT:      ${CKPT}"
echo "DATA_ROOT: ${DATA_ROOT}"
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

python -m temporal.eval \
    --data-dir "${DATA_ROOT}" \
    --checkpoint "${CKPT}" \
    --val-towns 1 \
    --val-weathers 18 \
    --temporal-frames 4 \
    --frame-stride 1 \
    --temporal-depth 2 \
    --batch-size 4 \
    --workers 4

echo "=========================================="
echo "Eval DONE. JSON saved next to checkpoint."
echo "=========================================="
