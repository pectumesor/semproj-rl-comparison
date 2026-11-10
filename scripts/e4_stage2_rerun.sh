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

# Reruns the 28 E4 Stage 2 runs lost when the home quota filled up on Nov 4 (15:24-15:40). Hydra caught the
# OSError, so sacct shows the 18 failed ones as COMPLETED; the MLP reference never got submitted at all:
#   CNN base                 10 of 10  (array 19704799, failed at wandb.init)
#   MLP reference            10 of 10  (multirun failed on its sweep folder)
#   Discrete actions          4 of 10  (19704798: medium seed 101 kept its auc.npy but has no metrics)
#   Masked LSTM               4 of 10  (19704800: medium seeds 100-103)
# Same overrides, storage and timestamp as Stage 2, so every run lands in its config's folder:
#   sbatch ... scripts/e4_stage2_rerun.sh
GROUP_NAME=e4_stage2
STORAGE=${SCRATCH}/${GROUP_NAME}
NOW=20261104133601
GOAL_RADIUS=16
MLP_LR=1e-4
LSTM_LR=3e-4
ROOMS=rooms/medium_maze.json,rooms/hard_maze.json
SEEDS=100,101,102,103,104

# W&B and Hydra wrote to the home folder, which is how a full quota could kill training. Keep both on scratch
export WANDB_DIR=${STORAGE}/wandb
export WANDB_CACHE_DIR=${STORAGE}/wandb/cache
export WANDB_DATA_DIR=${STORAGE}/wandb/data
mkdir -p "${WANDB_DIR}" "${WANDB_CACHE_DIR}" "${WANDB_DATA_DIR}"

# One multirun per lost block. The name keeps the parallel multiruns out of each other's sweep folder
run() {
    local name=$1 rooms=$2 seeds=$3; shift 3
    python scripts/train.py --multirun \
        hydra/launcher=euler \
        hydra.sweep.dir="${STORAGE}/multirun/rerun_${name}" \
        hydra.launcher.submitit_folder="${STORAGE}/logs/%j" \
        hydra.launcher.timeout_min=1440 \
        env.room_path="${rooms}" \
        env.goal_radius="${GOAL_RADIUS}" \
        env.colour_mode=off \
        env.goal_mode=random \
        algorithm.max_steps=4096 \
        algorithm.lr="${LR:-${MLP_LR}}" \
        wandb.group="${GROUP_NAME}" \
        seed="${seeds}" \
        storage="${STORAGE}" \
        timestamp="${NOW}" \
        env.gamma=0.999 \
        env.fov=180 \
        "$@" \
        || echo "Some training jobs failed, plotting the seeds that finished" >&2
}

cnn_base="observation=cnn_observation env.collision_penalty=0.01"
discrete="env.action_type=discrete env.discrete_actions=3"

cd "${PROJECT_ROOT}"
# shellcheck disable=SC2086
run base            "${ROOMS}"              "${SEEDS}"   ${cnn_base} &
run mlp_reference   "${ROOMS}"              "${SEEDS}"   observation=mlp_observation env.collision_penalty=0.0 &
# shellcheck disable=SC2086
run discrete_medium rooms/medium_maze.json  101          ${cnn_base} ${discrete} &
# shellcheck disable=SC2086
run discrete_hard   rooms/hard_maze.json    102,103,104  ${cnn_base} ${discrete} &
LR=${LSTM_LR} run masked_lstm rooms/medium_maze.json 100,101,102,103 observation=mlp_observation \
    env.collision_penalty=0.0 backbone=lstm_backbone 'observation.masked_channels=[3]' head.auxiliary_head.enabled=false &
wait

python scripts/evaluate_metrics.py \
    wandb.group="${GROUP_NAME}" \
    storage="${STORAGE}" \
    timestamp="${NOW}" \
    hydra.run.dir="${STORAGE}/hydra" \
    hydra.output_subdir=null
