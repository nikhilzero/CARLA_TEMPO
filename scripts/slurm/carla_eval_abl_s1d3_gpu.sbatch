#!/bin/bash
# carla_eval_abl_s1d3_gpu.sbatch — Closed-loop CARLA eval, ablation stride=1 d03 (dropout=0.3)
#SBATCH --job-name=carla_s1d3
#SBATCH --partition=gpu
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --array=0-4
#SBATCH --output=/scratch/nd967/CARLA_TEMPO/logs/carla_abl_s1d3_%A_%a.out
#SBATCH --error=/scratch/nd967/CARLA_TEMPO/logs/carla_abl_s1d3_%A_%a.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=nd967@rutgers.edu
#SBATCH --exclude=gpu005,gpu006,gpu018,gpu029,gpu030

set -e
BATCH_ID=${SLURM_ARRAY_TASK_ID}
PROJECT_DIR="/scratch/nd967/CARLA_TEMPO"
CARLA_ROOT="/scratch/nd967/carla"
CONDA_ENV="interfuser"
PORT=$((9000 + BATCH_ID * 2))
TM_PORT=$((9500 + BATCH_ID * 2))
RESULTS_DIR="${PROJECT_DIR}/results/carla_eval"
ROUTE_FILE="${PROJECT_DIR}/InterFuser/leaderboard/data/training_routes/routes_town05_batch${BATCH_ID}.xml"
SCENARIO_FILE="${PROJECT_DIR}/InterFuser/leaderboard/data/scenarios/town05_all_scenarios.json"

echo "=========================================="
echo "Job: ${SLURM_JOB_ID}_${BATCH_ID} | Node: $(hostname) | Port: ${PORT}"
echo "=========================================="

module purge
module load cuda/12.1.0
source ~/miniconda3/bin/activate ${CONDA_ENV}
export LD_LIBRARY_PATH=~/miniconda3/envs/${CONDA_ENV}/lib:${LD_LIBRARY_PATH}
export SDL_VIDEODRIVER=offscreen
export PYTHONUNBUFFERED=1
export SDL_HINT_CUDA_DEVICE=0
export SAVE_PATH="/tmp/carla_eval_s1d3_${BATCH_ID}"

nvidia-smi --query-gpu=index,name,memory.free,memory.total --format=csv 2>/dev/null || echo "nvidia-smi failed"
WORKING_GPUS=""
NUM_GPUS=$(nvidia-smi --list-gpus 2>/dev/null | wc -l)
for gpu_idx in $(seq 0 $((NUM_GPUS - 1))); do
    if nvidia-smi -i ${gpu_idx} --query-gpu=index --format=csv,noheader 2>/dev/null | grep -qE "^[0-9]+$"; then
        WORKING_GPUS="${WORKING_GPUS}${gpu_idx},"
    fi
done
WORKING_GPUS=${WORKING_GPUS%,}
[ -z "${WORKING_GPUS}" ] && { echo "ERROR: No working GPUs"; exit 1; }
export CUDA_VISIBLE_DEVICES="${WORKING_GPUS}"

[ ! -f "${CARLA_ROOT}/CarlaUE4.sh" ] && { echo "ERROR: CARLA not found"; exit 1; }
mkdir -p "${RESULTS_DIR}" "${PROJECT_DIR}/logs"

export CARLA_ROOT="${CARLA_ROOT}"
export PYTHONPATH="${CARLA_ROOT}/PythonAPI:${CARLA_ROOT}/PythonAPI/carla:${PROJECT_DIR}/InterFuser/leaderboard:${PROJECT_DIR}/InterFuser/leaderboard/team_code:${PROJECT_DIR}/InterFuser/scenario_runner:${PROJECT_DIR}/InterFuser/interfuser:${PROJECT_DIR}"

SDL_VIDEODRIVER=offscreen "${CARLA_ROOT}/CarlaUE4.sh" --world-port=${PORT} -RenderOffScreen -nosound -carla-streaming-port=0 -quality-level=Low > /tmp/carla_stdout_s1d3_${BATCH_ID}.log 2>&1 &
CARLA_PID=$!
echo "CARLA PID: ${CARLA_PID}"

CARLA_READY=0
for i in $(seq 1 60); do
    sleep 5
    if python -c "import carla; c=carla.Client('localhost',${PORT}); c.set_timeout(5.0); c.get_server_version()" 2>/dev/null; then
        CARLA_READY=1; echo "CARLA ready after $((i*5))s"; break
    fi
    (( i % 6 == 0 )) && echo "  Waiting... $((i*5))s/300s"
done
[ ${CARLA_READY} -eq 0 ] && { echo "ERROR: CARLA timeout"; kill ${CARLA_PID} 2>/dev/null; exit 1; }

PYTORCH_GPU=$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader 2>/dev/null | grep -E "^[0-9]" | sort -t',' -k2 -rn | head -1 | cut -d',' -f1 | tr -d ' ')
PYTORCH_GPU=${PYTORCH_GPU:-0}

export ROUTES="${ROUTE_FILE}"
echo "Running stride=1 d03 evaluation (batch ${BATCH_ID})..."
cd "${PROJECT_DIR}/InterFuser"

CUDA_VISIBLE_DEVICES=${PYTORCH_GPU} python leaderboard/leaderboard/leaderboard_evaluator.py \
    --scenarios="${SCENARIO_FILE}" --routes="${ROUTE_FILE}" --repetitions=1 --track=SENSORS \
    --checkpoint="${RESULTS_DIR}/abl_s1d3_result_batch${BATCH_ID}.json" \
    --agent=leaderboard/team_code/temporal_agent.py \
    --agent-config=leaderboard/team_code/temporal_abl_s1_d03_config.py \
    --debug=0 --resume=False --port=${PORT} --trafficManagerPort=${TM_PORT}

echo "Done. Results: ${RESULTS_DIR}/abl_s1d3_result_batch${BATCH_ID}.json"
kill ${CARLA_PID} 2>/dev/null || true
echo "=========================================="
