#!/bin/bash
# =============================================================================
# carla_eval_temporal_gpu.sbatch — Closed-loop CARLA eval, temporal model (T=4)
# Array job: 5 batches × 2 routes each = 10 Town05 routes total
#
# Submit: sbatch scripts/slurm/carla_eval_temporal_gpu.sbatch
# =============================================================================

#SBATCH --job-name=carla_temp_gpu
#SBATCH --partition=gpu
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --array=0-4
#SBATCH --output=/scratch/nd967/CARLA_TEMPO/logs/carla_temp_gpu_%A_%a.out
#SBATCH --error=/scratch/nd967/CARLA_TEMPO/logs/carla_temp_gpu_%A_%a.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=nd967@rutgers.edu
#SBATCH --exclude=gpu018,gpu029,gpu030

set -e

BATCH_ID=${SLURM_ARRAY_TASK_ID}
PROJECT_DIR="/scratch/nd967/CARLA_TEMPO"
CARLA_ROOT="/scratch/nd967/carla"
CONDA_ENV="interfuser"
PORT=$((3000 + BATCH_ID * 2))
TM_PORT=$((3500 + BATCH_ID * 2))
RESULTS_DIR="${PROJECT_DIR}/results/carla_eval"
ROUTE_FILE="${PROJECT_DIR}/InterFuser/leaderboard/data/training_routes/routes_town05_batch${BATCH_ID}.xml"
SCENARIO_FILE="${PROJECT_DIR}/InterFuser/leaderboard/data/scenarios/town05_all_scenarios.json"

echo "=========================================="
echo "Job:      ${SLURM_JOB_ID}_${BATCH_ID}"
echo "Node:     $(hostname)"
echo "Batch:    ${BATCH_ID}, Port: ${PORT}"
echo "Route:    ${ROUTE_FILE}"
echo "=========================================="

module purge
module load cuda/12.1.0
source ~/miniconda3/bin/activate ${CONDA_ENV}
export LD_LIBRARY_PATH=~/miniconda3/envs/${CONDA_ENV}/lib:${LD_LIBRARY_PATH}

# ---- GPU status ----
echo "=== GPU Status ==="
nvidia-smi --query-gpu=index,name,memory.free,memory.total --format=csv 2>/dev/null || echo "nvidia-smi failed"

# ---- Find working GPUs (skip broken PCI devices) ----
WORKING_GPUS=""
NUM_GPUS=$(nvidia-smi --list-gpus 2>/dev/null | wc -l)
for gpu_idx in $(seq 0 $((NUM_GPUS - 1))); do
    if nvidia-smi -i ${gpu_idx} --query-gpu=index --format=csv,noheader 2>/dev/null | grep -q "[0-9]"; then
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

mkdir -p "${RESULTS_DIR}" "${PROJECT_DIR}/logs"

# ---- Set PYTHONPATH (includes PROJECT_DIR for temporal/ package) ----
export CARLA_ROOT="${CARLA_ROOT}"
export PYTHONPATH="${CARLA_ROOT}/PythonAPI:${CARLA_ROOT}/PythonAPI/carla:${PROJECT_DIR}/InterFuser/leaderboard:${PROJECT_DIR}/InterFuser/leaderboard/team_code:${PROJECT_DIR}/InterFuser/scenario_runner:${PROJECT_DIR}/InterFuser/interfuser:${PROJECT_DIR}"

# ---- Start CARLA server ----
echo "Starting CARLA on port ${PORT} with CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}..."
SDL_VIDEODRIVER=offscreen "${CARLA_ROOT}/CarlaUE4.sh" \
    --world-port=${PORT} \
    -RenderOffScreen \
    -nosound \
    -carla-streaming-port=0 \
    -quality-level=Low \
    > /tmp/carla_stdout_${BATCH_ID}.log 2>&1 &
CARLA_PID=$!
echo "CARLA PID: ${CARLA_PID}"

# ---- Wait for CARLA (up to 5 minutes = 60 × 5s) ----
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
    head -50 /tmp/carla_stdout_${BATCH_ID}.log 2>/dev/null || echo "No CARLA log"
    kill ${CARLA_PID} 2>/dev/null || true
    exit 1
fi

# ---- Pick GPU with most free memory for PyTorch ----
PYTORCH_GPU=$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader 2>/dev/null | \
    sort -t',' -k2 -rn | head -1 | cut -d',' -f1 | tr -d ' ')
PYTORCH_GPU=${PYTORCH_GPU:-0}
echo "Using CUDA device ${PYTORCH_GPU} for PyTorch"

# ---- Run evaluation ----
echo "Running temporal evaluation (batch ${BATCH_ID}, T=4, stride=5)..."
cd "${PROJECT_DIR}/InterFuser"

CUDA_VISIBLE_DEVICES=${PYTORCH_GPU} python leaderboard/leaderboard/leaderboard_evaluator.py \
    --scenarios="${SCENARIO_FILE}" \
    --routes="${ROUTE_FILE}" \
    --repetitions=1 \
    --track=SENSORS \
    --checkpoint="${RESULTS_DIR}/temporal_result_batch${BATCH_ID}.json" \
    --agent=leaderboard/team_code/temporal_agent.py \
    --agent-config=leaderboard/team_code/temporal_config.py \
    --debug=0 \
    --resume=False \
    --port=${PORT} \
    --trafficManagerPort=${TM_PORT}

echo "Evaluation complete. Results: ${RESULTS_DIR}/temporal_result_batch${BATCH_ID}.json"
kill ${CARLA_PID} 2>/dev/null || true
echo "CARLA stopped."
echo "=========================================="
