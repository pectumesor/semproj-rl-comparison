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

# E0 pilot: one seed per room tier, MLP and LSTM, three learning rates.
# Gives the real hours per run that every later budget scales with, and the learning rate
# per backbone (Andrychowicz et al. tune the learning rate in every theme).
# Base config of the plan: colour off, goal random per episode.
GROUP_NAME="e0_pilot"
SEEDS=(0)
BACKBONE_NN=("mlp_backbone" "lstm_backbone")
LEARNING_RATES=(1e-4 3e-4 1e-3)
COLOUR_MODE=off
GOAL_MODE=random
GOAL_RADIUS=16
# One timestamp for the whole sweep, so every SLURM job writes its seed under the same folder
NOW=$(date +"%Y%m%d_%H%M%S")
STORAGE=${SCRATCH}/${GROUP_NAME}

# Hydra sweeps over comma-separated values: (a b c) -> a,b,c
join() { local IFS=","; echo "$*"; }

# One multirun per room tier: the episode length and gamma are tier properties
run_tier() {
    local rooms=$1 episode_length=$2 gamma=$3
    python scripts/train.py --multirun \
        hydra/launcher=euler \
        hydra.launcher.submitit_folder="${STORAGE}/logs/%j" \
        hydra.launcher.timeout_min=720 \
        env.room_path="${rooms}" \
        env.goal_radius="${GOAL_RADIUS}" \
        env.colour_mode="${COLOUR_MODE}" \
        env.goal_mode="${GOAL_MODE}" \
        env.gamma="${gamma}" \
        algorithm.max_steps="${episode_length}" \
        algorithm.lr="$(join "${LEARNING_RATES[@]}")" \
        backbone="$(join "${BACKBONE_NN[@]}")" \
        wandb.group="${GROUP_NAME}" \
        seed="$(join "${SEEDS[@]}")" \
        storage="${STORAGE}" \
        timestamp="${NOW}" \
        || echo "Some training jobs failed, plotting the seeds that finished" >&2
}

cd "${PROJECT_ROOT}"
# Each launcher blocks until its SLURM jobs have finished, so the tiers run side by side.
# A failed job must not stop the seeds that did finish from being plotted below.
run_tier "rooms/two_room.json" 512 0.99 &
run_tier "rooms/four_room.json" 512 0.99 &
run_tier "rooms/medium_maze.json" 4096 0.999 &
wait

# Mean-of-means plot per room and architecture, uploaded to W&B from this node.
# Read the per-run wall clock from the SLURM logs in ${STORAGE}/logs to fill the budget table.
python scripts/evaluate_metrics.py \
    wandb.group="${GROUP_NAME}" \
    storage="${STORAGE}" \
    timestamp="${NOW}"
