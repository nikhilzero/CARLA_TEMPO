#!/bin/bash
# =============================================================================
# install_carla.sh — Install CARLA 0.9.10.1 on Amarel HPC
#
# Run this ONCE from a login node (not a SLURM job):
#   bash scripts/install_carla.sh
#
# What this does:
#   1. Downloads CARLA 0.9.10.1 + AdditionalMaps to /scratch/nd967/carla/
#   2. Installs the CARLA Python API into the interfuser conda env
#   3. Installs required packages (pygame, easydict)
#   4. Verifies the installation
# =============================================================================

set -e

CARLA_DIR="/scratch/nd967/carla"
CONDA_ENV="interfuser"

echo "=== CARLA 0.9.10.1 Installation ==="
echo "Target dir: ${CARLA_DIR}"
echo ""

# ---- 1. Create target dir ----
mkdir -p "${CARLA_DIR}"
cd "${CARLA_DIR}"

# ---- 2. Download CARLA (skip if already downloaded) ----
if [ ! -f "CarlaUE4.sh" ]; then
    echo "[1/4] Downloading CARLA 0.9.10.1 (~12 GB)..."
    wget -q --show-progress https://tiny.carla.org/carla-0-9-10-1-linux -O CARLA_0.9.10.1.tar.gz
    echo "[1/4] Extracting..."
    tar -xf CARLA_0.9.10.1.tar.gz
    rm CARLA_0.9.10.1.tar.gz
    echo "[1/4] CARLA extracted."
else
    echo "[1/4] CARLA already extracted — skipping download."
fi

# ---- 3. Download AdditionalMaps (Town06, Town07, Town10) ----
if [ ! -d "Import" ] || [ -z "$(ls Import/*.tar.gz 2>/dev/null)" ]; then
    echo "[2/4] Downloading AdditionalMaps (~3 GB)..."
    wget -q --show-progress https://tiny.carla.org/additional-maps-0-9-10-1-linux -O AdditionalMaps_0.9.10.1.tar.gz
    tar -xf AdditionalMaps_0.9.10.1.tar.gz
    rm AdditionalMaps_0.9.10.1.tar.gz
    echo "[2/4] AdditionalMaps extracted."
else
    echo "[2/4] AdditionalMaps already present — skipping."
fi

# ---- 4. Install CARLA Python API ----
echo "[3/4] Installing CARLA Python API..."
source ~/miniconda3/bin/activate ${CONDA_ENV}

# pip package works with Python 3.8 (unlike the .egg which is 3.7-only)
pip install carla==0.9.10.1 --quiet
echo "[3/4] CARLA Python API installed."

# ---- 5. Install leaderboard dependencies ----
echo "[4/4] Installing leaderboard dependencies..."
pip install pygame easydict py-trees==0.8.3 networkx==2.2 --quiet
echo "[4/4] Dependencies installed."

# ---- 6. Verify ----
echo ""
echo "=== Verification ==="
python -c "import carla; print(f'carla module OK: {carla.__file__}')"
ls "${CARLA_DIR}/CarlaUE4.sh" && echo "CarlaUE4.sh found OK"
echo ""
echo "=== Installation complete ==="
echo "CARLA dir: ${CARLA_DIR}"
echo ""
echo "Test with:"
echo "  bash scripts/slurm/carla_eval_baseline.sbatch  (submit as SLURM job)"
