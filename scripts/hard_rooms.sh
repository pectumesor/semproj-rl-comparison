#!/usr/bin/env bash
set -euo pipefail

ETH_USERNAME=eazevedo
PROJECT_NAME=semproj
SCRATCH=/cluster/scratch/${ETH_USERNAME}/${PROJECT_NAME}
CONDA_ENVIRONMENT=semproj
CONDA_ROOT=${HOME}/miniforge3
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

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

mkdir -p "${SCRATCH}/.submitit"

GROUP_NAME="hard_rooms"
ROOM_PATHS=("rooms/cross_room.json" "rooms/C_maze.json" "rooms/medium_maze.json" "rooms/hard_maze.json")
#SEEDS=(0 1 2 3 4)
SEEDS=(0 4)
OBSERVATION_NN=("mlp_observation" "cnn_observation")
BACKBONE_NN=("lstm_backbone" "mlp_backbone")
AUX_HEAD=True
GOAL_RADIUS=16
GAMMA=0.999
EPISODE_LENGTH=4096
# One timestamp for the whole sweep, so every SLURM job writes its seed under the same folder
NOW=$(date +"%Y%m%d_%H%M%S")
STORAGE=${SCRATCH}/${GROUP_NAME}

# Hydra sweeps over comma-separated values: (a b c) -> a,b,c
join() { local IFS=","; echo "$*"; }

cd "${PROJECT_ROOT}"
# Blocks until every SLURM job has finished. A failed job must not stop the
# seeds that did finish from being plotted below.
python scripts/train.py --multirun \
    hydra/launcher=euler \
    hydra.launcher.submitit_folder="${SCRATCH}/.submitit/%j" \
    env.room_path="$(join "${ROOM_PATHS[@]}")" \
    env.goal_radius="${GOAL_RADIUS}" \
    wandb.group="${GROUP_NAME}" \
    seed="$(join "${SEEDS[@]}")" \
    observation="$(join "${OBSERVATION_NN[@]}")" \
    backbone="$(join "${BACKBONE_NN[@]}")" \
    head.auxiliary_head.enabled="$(join "${AUX_HEAD[@]}")" \
    env.gamma="${GAMMA}" \
    algorithm.max_steps="${EPISODE_LENGTH}" \
    storage="${STORAGE}" \
    timestamp="${NOW}" \
    || echo "Some training jobs failed, plotting the seeds that finished" >&2

# Mean-of-means plot per room and architecture, uploaded to W&B from this node
python scripts/evaluate_metrics.py \
    wandb.group="${GROUP_NAME}" \
    storage="${STORAGE}" \
    timestamp="${NOW}"
