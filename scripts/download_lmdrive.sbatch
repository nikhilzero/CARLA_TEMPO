#!/bin/bash
#SBATCH --job-name=lmdrive_download
#SBATCH --partition=main          # CPU partition — no GPU needed for download
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=12:00:00           # downloads can take hours
#SBATCH --output=/scratch/nd967/CARLA_TEMPO/logs/download_%j.out
#SBATCH --error=/scratch/nd967/CARLA_TEMPO/logs/download_%j.err

# =============================================================================
# download_lmdrive.sbatch
#
# Downloads a selected subset of the LMDrive dataset from HuggingFace.
# Edit TOWNS and WEATHERS below to control what gets downloaded.
#
# Submit: sbatch scripts/download_lmdrive.sbatch
# Monitor: tail -f logs/download_<jobid>.out
# =============================================================================

set -e

DATASET_DIR="/scratch/nd967/CARLA_TEMPO/InterFuser/dataset"
LOG_DIR="/scratch/nd967/CARLA_TEMPO/logs"
mkdir -p "$LOG_DIR"
mkdir -p "$DATASET_DIR"

source ~/miniconda3/bin/activate interfuser

# Install huggingface_hub if not present
pip install huggingface_hub --quiet

echo "=== LMDrive Download Job ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Start: $(date)"
echo "Target dir: $DATASET_DIR"
echo ""

# Check available disk space before starting
echo "=== Disk space before download ==="
df -h /scratch/nd967
echo ""

python /scratch/nd967/CARLA_TEMPO/scripts/download_lmdrive.py \
    --output-dir "$DATASET_DIR"

echo ""
echo "=== Disk space after download ==="
df -h /scratch/nd967

echo ""
echo "=== Download complete: $(date) ==="
