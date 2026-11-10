#!/usr/bin/env bash
set -euo pipefail

ETH_USERNAME=eazevedo
PROJECT_NAME=semproj
SCRATCH=/cluster/scratch/${ETH_USERNAME}/${PROJECT_NAME}
CONDA_ENVIRONMENT=semproj
CONDA_ROOT=${HOME}/miniforge3
# Under sbatch the script runs from a copy in /var/spool/slurm, so BASH_SOURCE
# points there; resolve the repo from the submission directory instead.
if [[ -n "${SLURM_JOB_ID:-}" ]]; then
    PROJECT_ROOT="$(git -C "${SLURM_SUBMIT_DIR}" rev-parse --show-toplevel)"
else
    PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
fi

# eth_proxy gives the login node outbound internet (needed by conda/pip/wandb).
module load eth_proxy

if [ ! -f "${CONDA_ROOT}/etc/profile.d/conda.sh" ]; then
    echo "No conda at ${CONDA_ROOT}. Install Miniforge first:" >&2
    echo "  curl -L -o /tmp/miniforge.sh https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh" >&2
    echo "  bash /tmp/miniforge.sh -b -p ${CONDA_ROOT}" >&2
    exit 1
fi
source "${CONDA_ROOT}/etc/profile.d/conda.sh"

# conda's own activation hooks (e.g. MKL's) reference unset vars, so
# relax nounset around anything that touches conda activate/create.
set +u
if conda env list | grep -qE "^${CONDA_ENVIRONMENT}\s"; then
    conda activate "${CONDA_ENVIRONMENT}"
else
    echo "Conda environment '${CONDA_ENVIRONMENT}' not found, creating from environment.yaml"
    conda env create -f "${PROJECT_ROOT}/environment.yaml"
    conda activate "${CONDA_ENVIRONMENT}"
fi
set -u

# E0 room probe: which rooms still separate configs under the base config (colour off, random goal)?
# The pilot found two_room and four_room saturate (padded AUC >= 0.92 for every config), so E4 Stage 1
# needs rooms that don't. One seed of the pilot's best MLP and LSTM per candidate room: a room is usable
# for screening if neither saturates (AUC well below 0.9) and not both are stuck at 0.
GROUP_NAME="e0_room_probe"
SEEDS=(0)
GOAL_RADIUS=16
NOW=$(date +"%Y%m%d_%H%M%S")
STORAGE=${SCRATCH}/${GROUP_NAME}

join() { local IFS=","; echo "$*"; }

# One multirun per tier and backbone: the learning rate is the pilot's pick per backbone
run_tier() {
    local rooms=$1 episode_length=$2 gamma=$3 backbone=$4 lr=$5
    python scripts/train.py --multirun \
        hydra/launcher=euler \
        hydra.launcher.submitit_folder="${STORAGE}/logs/%j" \
        hydra.launcher.timeout_min=240 \
        env.room_path="${rooms}" \
        env.goal_radius="${GOAL_RADIUS}" \
        env.colour_mode=off \
        env.goal_mode=random \
        env.gamma="${gamma}" \
        algorithm.max_steps="${episode_length}" \
        algorithm.lr="${lr}" \
        backbone="${backbone}" \
        wandb.group="${GROUP_NAME}" \
        seed="$(join "${SEEDS[@]}")" \
        storage="${STORAGE}" \
        timestamp="${NOW}" \
        || echo "Some training jobs failed, plotting the seeds that finished" >&2
}

SHORT_ROOMS=("rooms/T_maze_room.json" "rooms/I_maze_room.json")
LONG_ROOMS=("rooms/cross_room.json" "rooms/C_maze.json" "rooms/hard_maze.json")

cd "${PROJECT_ROOT}"
run_tier "$(join "${SHORT_ROOMS[@]}")" 512 0.99 mlp_backbone 1e-4 &
run_tier "$(join "${SHORT_ROOMS[@]}")" 512 0.99 lstm_backbone 3e-4 &
run_tier "$(join "${LONG_ROOMS[@]}")" 4096 0.999 mlp_backbone 1e-4 &
run_tier "$(join "${LONG_ROOMS[@]}")" 4096 0.999 lstm_backbone 3e-4 &
wait

python scripts/evaluate_metrics.py \
    wandb.group="${GROUP_NAME}" \
    storage="${STORAGE}" \
    timestamp="${NOW}"
