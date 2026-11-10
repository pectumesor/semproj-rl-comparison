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

# Fair BPTT windows (reports/Recurrent PPO BPTT implementations.md). RecurrentPPO runs
# (num_envs / minibatch_size) x (num_steps / bptt_window) x n_epochs updates per iteration, so a fixed minibatch gives
# 128, 32 and 8 updates at windows 16, 64 and 256. A minibatch_size of 1024 / window keeps all three at 128 updates
# of 1024 samples, where the LSTM's 3e-4 was tuned.
#   Learning rate pilot: every window at three rates on the screening seeds, it picks the rates, it supports no claim
#   MLP arm: the MLP at the LSTM's 128 updates of 1024 samples, against the tuned MLP of Stage 2 (lr by the square
#   root rule of Hilton et al. 2022). If its AUC moves, the LSTM vs MLP claim carries that qualifier
#   sbatch ... scripts/e4_lstm_updates.sh
GROUP_NAME=e4_lstm_updates
STORAGE=${SCRATCH}/${GROUP_NAME}
NOW=$(date +"%Y%m%d_%H%M%S")
GOAL_RADIUS=16
LRS=1e-4,3e-4,1e-3
SCREENING_SEEDS=0,1,2
SEEDS=100,101,102,103,104
ROOMS=rooms/medium_maze.json,rooms/hard_maze.json

# Keep W&B and Hydra off the home folder's file quota
export WANDB_DIR=${STORAGE}/wandb
export WANDB_CACHE_DIR=${STORAGE}/wandb/cache
export WANDB_DATA_DIR=${STORAGE}/wandb/data
mkdir -p "${WANDB_DIR}" "${WANDB_CACHE_DIR}" "${WANDB_DATA_DIR}"

run() {
    local name=$1 rooms=$2 lrs=$3 seeds=$4; shift 4
    python scripts/train.py --multirun \
        hydra/launcher=euler \
        hydra.sweep.dir="${STORAGE}/multirun/${NOW}/${name}" \
        hydra.launcher.submitit_folder="${STORAGE}/logs/%j" \
        hydra.launcher.timeout_min="${TIMEOUT_MIN:-1440}" \
        env.room_path="${rooms}" \
        env.goal_radius="${GOAL_RADIUS}" \
        env.colour_mode=off \
        env.goal_mode=random \
        env.gamma=0.999 \
        env.fov=180 \
        env.collision_penalty=0.0 \
        algorithm.max_steps=4096 \
        algorithm.lr="${lrs}" \
        observation=mlp_observation \
        wandb.group="${GROUP_NAME}" \
        seed="${seeds}" \
        storage="${STORAGE}" \
        timestamp="${NOW}" \
        "$@" \
        || echo "Some training jobs failed, plotting the seeds that finished" >&2
}

lstm="backbone=lstm_backbone head.auxiliary_head.enabled=false"

cd "${PROJECT_ROOT}"
# shellcheck disable=SC2086
run pilot_16 rooms/medium_maze.json "${LRS}" "${SCREENING_SEEDS}" ${lstm} backbone.bptt_window_size=16 algorithm.minibatch_size=64 &
# shellcheck disable=SC2086
run pilot_64 rooms/medium_maze.json "${LRS}" "${SCREENING_SEEDS}" ${lstm} backbone.bptt_window_size=64 algorithm.minibatch_size=16 &
# Window 256 with 4 envs per minibatch costs about 7x the update time of window 16, on a slow node 24 h can be too short,
# and a killed run saves no auc.npy
# shellcheck disable=SC2086
TIMEOUT_MIN=2880 run pilot_256 rooms/medium_maze.json "${LRS}" "${SCREENING_SEEDS}" ${lstm} backbone.bptt_window_size=256 algorithm.minibatch_size=4 &
run mlp_arm "${ROOMS}" 2e-4 "${SEEDS}" backbone=mlp_backbone algorithm.mini_batch_size=1024 &
wait

python scripts/evaluate_metrics.py \
    wandb.group="${GROUP_NAME}" \
    storage="${STORAGE}" \
    timestamp="${NOW}" \
    hydra.run.dir="${STORAGE}/hydra" \
    hydra.output_subdir=null
