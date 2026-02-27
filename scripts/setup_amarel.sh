#!/bin/bash
# =============================================================================
# setup_amarel.sh — One-time environment setup on Amarel HPC
# Run this ONCE after cloning the repo on Amarel:
#   cd /scratch/nd967/CARLA_TEMPO && bash scripts/setup_amarel.sh
# =============================================================================

set -e

PROJECT_DIR="/scratch/nd967/CARLA_TEMPO"
CONDA_ENV="interfuser"

echo "=== CARLA_TEMPO Amarel Setup ==="

# ---- 1. Load modules ----
module purge
module load cuda/12.1.0
echo "[OK] Loaded cuda/12.1.0"

# ---- 2. Activate conda env ----
source ~/miniconda3/bin/activate ${CONDA_ENV}
echo "[OK] Activated conda env: ${CONDA_ENV}"

# ---- 3. Install InterFuser dependencies (skip if already installed) ----
cd ${PROJECT_DIR}/InterFuser/interfuser
pip install -e . --quiet
echo "[OK] InterFuser package installed"

# ---- 4. Install temporal extension deps ----
cd ${PROJECT_DIR}
if [ -f requirements.txt ]; then
    pip install -r requirements.txt --quiet
    echo "[OK] temporal requirements installed"
fi

# ---- 5. Verify CUDA access ----
python -c "import torch; print(f'[OK] PyTorch {torch.__version__}, CUDA available: {torch.cuda.is_available()}')"

echo ""
echo "=== Setup complete. You can now run sbatch scripts/train_baseline.sh ==="
