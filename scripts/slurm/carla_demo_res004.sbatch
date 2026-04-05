#!/bin/bash
# =============================================================================
# carla_demo_res004.sbatch — Demo video recording: RES-004 on RouteScenario_21
#
# Runs the temporal model (T=4, dropout=0.3) on Town05 route 21 (DS=24.393,
# no collisions — the best route for demo purposes).
# Saves front-camera PNGs to DEMO_FRAMES_DIR, then stitches them into demo.mp4
# with ffmpeg at 20 fps.
#
# Submit: sbatch scripts/slurm/carla_demo_res004.sbatch
# Output video: /scratch/nd967/CARLA_TEMPO/demo/res004_s21_demo.mp4
# =============================================================================

#SBATCH --job-name=carla_demo_res004
#SBATCH --partition=gpu
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=/scratch/nd967/CARLA_TEMPO/logs/carla_demo_res004_%j.out
#SBATCH --error=/scratch/nd967/CARLA_TEMPO/logs/carla_demo_res004_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=nd967@rutgers.edu
#SBATCH --exclude=gpu005,gpu006,gpu018,gpu029,gpu030

set -e

PROJECT_DIR="/scratch/nd967/CARLA_TEMPO"
CARLA_ROOT="/scratch/nd967/carla"
CONDA_ENV="interfuser"
PORT=4200
TM_PORT=4700

ROUTE_FILE="${PROJECT_DIR}/InterFuser/leaderboard/data/training_routes/routes_town05_demo_s21.xml"
SCENARIO_FILE="${PROJECT_DIR}/InterFuser/leaderboard/data/scenarios/town05_all_scenarios.json"
RESULTS_DIR="${PROJECT_DIR}/demo"
DEMO_FRAMES_DIR="${PROJECT_DIR}/demo/frames_s21"
OUTPUT_VIDEO="${PROJECT_DIR}/demo/res004_s21_demo.mp4"

echo "=========================================="
echo "Job:      ${SLURM_JOB_ID}"
echo "Node:     $(hostname)"
echo "Route:    RouteScenario_21 (Town05)"
echo "Model:    RES-004 (temporal, T=4, dropout=0.3)"
echo "Frames →  ${DEMO_FRAMES_DIR}"
echo "Video  →  ${OUTPUT_VIDEO}"
echo "=========================================="

module purge
module load cuda/12.1.0
source ~/miniconda3/bin/activate ${CONDA_ENV}
export LD_LIBRARY_PATH=~/miniconda3/envs/${CONDA_ENV}/lib:${LD_LIBRARY_PATH}

export SDL_VIDEODRIVER=offscreen
export SDL_HINT_CUDA_DEVICE=0
export PYTHONUNBUFFERED=1
export ROUTES="${ROUTE_FILE}"
export SAVE_PATH="${RESULTS_DIR}/save"
export DEMO_FRAMES_PATH="${DEMO_FRAMES_DIR}"

mkdir -p "${RESULTS_DIR}" "${DEMO_FRAMES_DIR}" "${PROJECT_DIR}/logs"

# ---- GPU status ----
echo "=== GPU Status ==="
nvidia-smi --query-gpu=index,name,memory.free,memory.total --format=csv 2>/dev/null || echo "nvidia-smi failed"

# ---- Find working GPUs (skip broken PCI devices) ----
WORKING_GPUS=""
NUM_GPUS=$(nvidia-smi --list-gpus 2>/dev/null | wc -l)
for gpu_idx in $(seq 0 $((NUM_GPUS - 1))); do
    if nvidia-smi -i ${gpu_idx} --query-gpu=index --format=csv,noheader 2>/dev/null | grep -qE "^[0-9]+$"; then
        WORKING_GPUS="${WORKING_GPUS}${gpu_idx},"
    fi
done
WORKING_GPUS=${WORKING_GPUS%,}
if [ -z "${WORKING_GPUS}" ]; then
    echo "ERROR: No working GPUs found on $(hostname)"
    nvidia-smi 2>&1 || true
    exit 1
fi
echo "Working GPUs: ${WORKING_GPUS}"
export CUDA_VISIBLE_DEVICES="${WORKING_GPUS}"

# ---- Verify CARLA ----
if [ ! -f "${CARLA_ROOT}/CarlaUE4.sh" ]; then
    echo "ERROR: CARLA not found at ${CARLA_ROOT}"
    exit 1
fi

# ---- Set PYTHONPATH ----
export CARLA_ROOT="${CARLA_ROOT}"
export PYTHONPATH="${CARLA_ROOT}/PythonAPI:${CARLA_ROOT}/PythonAPI/carla:${PROJECT_DIR}/InterFuser/leaderboard:${PROJECT_DIR}/InterFuser/leaderboard/team_code:${PROJECT_DIR}/InterFuser/scenario_runner:${PROJECT_DIR}/InterFuser/interfuser:${PROJECT_DIR}"

# ---- Start CARLA server ----
echo "Starting CARLA on port ${PORT}..."
SDL_VIDEODRIVER=offscreen "${CARLA_ROOT}/CarlaUE4.sh" \
    --world-port=${PORT} \
    -RenderOffScreen \
    -nosound \
    -carla-streaming-port=0 \
    -quality-level=Epic \
    > /tmp/carla_demo_stdout.log 2>&1 &
CARLA_PID=$!
echo "CARLA PID: ${CARLA_PID}"

# ---- Wait for CARLA (up to 5 minutes) ----
echo "Waiting for CARLA server (up to 5 min)..."
CARLA_READY=0
for i in $(seq 1 60); do
    sleep 5
    if python -c "import carla; c=carla.Client('localhost',${PORT}); c.set_timeout(5.0); c.get_server_version(); print('CARLA ready')" 2>/dev/null; then
        CARLA_READY=1
        echo "CARLA ready after $((i * 5)) seconds"
        break
    fi
    if (( i % 6 == 0 )); then
        echo "  Waiting... ($((i * 5))s / 300s)"
    fi
done

if [ ${CARLA_READY} -eq 0 ]; then
    echo "ERROR: CARLA did not start within 5 minutes"
    echo "=== CARLA startup log ==="
    head -50 /tmp/carla_demo_stdout.log 2>/dev/null || echo "No CARLA log"
    kill ${CARLA_PID} 2>/dev/null || true
    exit 1
fi

# ---- Pick GPU with most free memory for PyTorch ----
PYTORCH_GPU=$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader 2>/dev/null | \
    grep -E "^[0-9]" | sort -t',' -k2 -rn | head -1 | cut -d',' -f1 | tr -d ' ')
PYTORCH_GPU=${PYTORCH_GPU:-0}
echo "Using CUDA device ${PYTORCH_GPU} for PyTorch"

# ---- Run demo evaluation ----
echo "Running RES-004 demo on RouteScenario_21..."
cd "${PROJECT_DIR}/InterFuser"

CUDA_VISIBLE_DEVICES=${PYTORCH_GPU} python leaderboard/leaderboard/leaderboard_evaluator.py \
    --scenarios="${SCENARIO_FILE}" \
    --routes="${ROUTE_FILE}" \
    --repetitions=1 \
    --track=SENSORS \
    --checkpoint="${RESULTS_DIR}/res004_s21_result.json" \
    --agent=leaderboard/team_code/temporal_agent.py \
    --agent-config=leaderboard/team_code/temporal_reg_config.py \
    --debug=0 \
    --resume=False \
    --port=${PORT} \
    --trafficManagerPort=${TM_PORT}

echo "Evaluation complete."
kill ${CARLA_PID} 2>/dev/null || true
echo "CARLA stopped."

# ---- Stitch frames into video ----
FRAME_COUNT=$(ls "${DEMO_FRAMES_DIR}"/frame_*.png 2>/dev/null | wc -l)
echo "Frames saved: ${FRAME_COUNT}"

if [ "${FRAME_COUNT}" -gt 0 ]; then
    echo "Stitching frames into ${OUTPUT_VIDEO} at 20 fps..."
    ffmpeg -y \
        -framerate 20 \
        -i "${DEMO_FRAMES_DIR}/frame_%06d.png" \
        -c:v libx264 \
        -preset slow \
        -crf 18 \
        -pix_fmt yuv420p \
        "${OUTPUT_VIDEO}"
    echo "Video ready: ${OUTPUT_VIDEO}"
    ls -lh "${OUTPUT_VIDEO}"
else
    echo "WARNING: No frames found in ${DEMO_FRAMES_DIR} — video not created."
fi

echo "=========================================="
echo "Demo job complete."
echo "  Frames:  ${DEMO_FRAMES_DIR} (${FRAME_COUNT} PNGs)"
echo "  Video:   ${OUTPUT_VIDEO}"
echo "  Result:  ${RESULTS_DIR}/res004_s21_result.json"
echo "=========================================="
