#!/bin/bash
# =============================================================================
# carla_eval_temporal.sbatch — Closed-loop CARLA evaluation, temporal model
#
# Same structure as carla_eval_baseline.sbatch but uses temporal_agent.py.
# Prerequisites: bash scripts/install_carla.sh must have been run first.
#
# Submit: sbatch scripts/slurm/carla_eval_temporal.sbatch
# =============================================================================

#SBATCH --job-name=carla_eval_temporal
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=06:00:00
#SBATCH --output=/scratch/nd967/CARLA_TEMPO/logs/carla_temporal_%j.out
#SBATCH --error=/scratch/nd967/CARLA_TEMPO/logs/carla_temporal_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=nd967@rutgers.edu

set -e

PROJECT_DIR="/scratch/nd967/CARLA_TEMPO"
CARLA_ROOT="/scratch/nd967/carla"
CONDA_ENV="interfuser"
PORT=2000
TM_PORT=2500
RESULTS_DIR="${PROJECT_DIR}/results/carla_eval"

echo "=========================================="
echo "Job ID:   ${SLURM_JOB_ID}"
echo "Node:     $(hostname)"
echo "GPU:      $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo unknown)"
echo "=========================================="

module purge
module load cuda/12.1.0
source ~/miniconda3/bin/activate ${CONDA_ENV}
export LD_LIBRARY_PATH=~/miniconda3/envs/${CONDA_ENV}/lib:${LD_LIBRARY_PATH}

if [ ! -f "${CARLA_ROOT}/CarlaUE4.sh" ]; then
    echo "ERROR: CARLA not found at ${CARLA_ROOT}"
    echo "Run: bash scripts/install_carla.sh"
    exit 1
fi

mkdir -p "${RESULTS_DIR}" "${PROJECT_DIR}/logs"

# ---- Sync agent files from tracked location into leaderboard ----
cp "${PROJECT_DIR}/temporal/agents/temporal_agent.py"  "${PROJECT_DIR}/InterFuser/leaderboard/team_code/temporal_agent.py"
cp "${PROJECT_DIR}/temporal/agents/temporal_config.py" "${PROJECT_DIR}/InterFuser/leaderboard/team_code/temporal_config.py"
echo "Agent files synced."

export CARLA_ROOT="${CARLA_ROOT}"
export PYTHONPATH="${CARLA_ROOT}/PythonAPI:${CARLA_ROOT}/PythonAPI/carla:${PROJECT_DIR}/InterFuser/leaderboard:${PROJECT_DIR}/InterFuser/leaderboard/team_code:${PROJECT_DIR}/InterFuser/scenario_runner:${PROJECT_DIR}/InterFuser/interfuser:${PROJECT_DIR}"

# ---- Start CARLA server ----
echo "Starting CARLA server on port ${PORT}..."
SDL_VIDEODRIVER=offscreen "${CARLA_ROOT}/CarlaUE4.sh" \
    --world-port=${PORT} \
    -RenderOffScreen \
    -nosound \
    -carla-streaming-port=0 \
    &
CARLA_PID=$!
echo "CARLA PID: ${CARLA_PID}"

echo "Waiting for CARLA server to start..."
for i in $(seq 1 18); do
    sleep 5
    if python -c "import carla; c=carla.Client('localhost',${PORT}); c.set_timeout(5.0); c.get_server_version(); print('CARLA ready')" 2>/dev/null; then
        break
    fi
    echo "  Waiting... (${i}/18)"
done

# ---- Run evaluation ----
echo "Running leaderboard evaluation (temporal T=4, stride=5)..."
cd "${PROJECT_DIR}/InterFuser"

python leaderboard/leaderboard/leaderboard_evaluator.py \
    --scenarios=leaderboard/data/scenarios/town05_all_scenarios.json \
    --routes=leaderboard/data/training_routes/routes_town05_long.xml \
    --repetitions=1 \
    --track=SENSORS \
    --checkpoint="${RESULTS_DIR}/temporal_result.json" \
    --agent=leaderboard/team_code/temporal_agent.py \
    --agent-config=leaderboard/team_code/temporal_config.py \
    --debug=0 \
    --resume=False \
    --port=${PORT} \
    --trafficManagerPort=${TM_PORT}

echo "Evaluation complete. Results: ${RESULTS_DIR}/temporal_result.json"

kill ${CARLA_PID} 2>/dev/null || true
echo "CARLA server stopped."
echo "=========================================="
