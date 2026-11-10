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

# BPTT windows 64 and 256 at equal updates on Stage 2's seeds (plan: "BPTT windows at equal updates"). The pilot
# (scripts/e4_lstm_updates.sh) kept the LSTM's 3e-4 at every window, and a minibatch of 1024 / window envs gives
# 128 updates of 1024 samples, as Stage 2's window 16 (64 envs) already has. Same settings as Stage 2's unmasked
# LSTM (Stage 1's MLP encoder, no collision penalty), storage and timestamp, so e4_promotion.py ranks the mb levels
# next to Stage 2's fixed-minibatch windows:
#   sbatch ... scripts/e4_bptt_confirm.sh
GROUP_NAME=e4_stage2
STORAGE=${SCRATCH}/${GROUP_NAME}
NOW=20261104133601
GOAL_RADIUS=16
LSTM_LR=3e-4

# Keep W&B and Hydra off the home folder's file quota
export WANDB_DIR=${STORAGE}/wandb
export WANDB_CACHE_DIR=${STORAGE}/wandb/cache
export WANDB_DATA_DIR=${STORAGE}/wandb/data
mkdir -p "${WANDB_DIR}" "${WANDB_CACHE_DIR}" "${WANDB_DATA_DIR}"

run() {
    local window=$1 minibatch=$2 timeout=$3
    python scripts/train.py --multirun \
        hydra/launcher=euler \
        hydra.sweep.dir="${STORAGE}/multirun/bptt_confirm_${window}" \
        hydra.launcher.submitit_folder="${STORAGE}/logs/%j" \
        hydra.launcher.timeout_min="${timeout}" \
        env.room_path=rooms/medium_maze.json,rooms/hard_maze.json \
        env.goal_radius="${GOAL_RADIUS}" \
        env.colour_mode=off \
        env.goal_mode=random \
        algorithm.max_steps=4096 \
        algorithm.lr="${LSTM_LR}" \
        wandb.group="${GROUP_NAME}" \
        seed=100,101,102,103,104 \
        storage="${STORAGE}" \
        timestamp="${NOW}" \
        env.gamma=0.999 \
        observation=mlp_observation \
        env.fov=180 \
        env.collision_penalty=0.0 \
        backbone=lstm_backbone \
        head.auxiliary_head.enabled=false \
        backbone.bptt_window_size="${window}" \
        algorithm.minibatch_size="${minibatch}" \
        || echo "Some training jobs failed, plotting the seeds that finished" >&2
}

cd "${PROJECT_ROOT}"
run 64 16 1440 &
# Window 256 with 4 envs per minibatch costs about 7x the update time of window 16, on a slow node 24 h can be too short,
# and a killed run saves no auc.npy
run 256 4 2880 &
wait

python scripts/evaluate_metrics.py \
    wandb.group="${GROUP_NAME}" \
    storage="${STORAGE}" \
    timestamp="${NOW}" \
    hydra.run.dir="${STORAGE}/hydra" \
    hydra.output_subdir=null
