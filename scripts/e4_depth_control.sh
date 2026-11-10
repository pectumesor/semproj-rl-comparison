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

# Control of scripts/e4_depth_screen.sh: Stage 1's baseline at the usual 2 hidden layers per module with the
# ReLU between modules. The 3-layer level changes depth and this activation together, this level changes only
# the activation. Same rooms, seeds, storage and timestamp as Stage 1:
#   sbatch ... scripts/e4_depth_control.sh
GROUP_NAME=e4_stage1
STORAGE=${SCRATCH}/${GROUP_NAME}
NOW=20261102095921
GOAL_RADIUS=16
MLP_LR=1e-4

cd "${PROJECT_ROOT}"
python scripts/train.py --multirun \
    hydra/launcher=euler \
    hydra.launcher.submitit_folder="${STORAGE}/logs/%j" \
    hydra.launcher.timeout_min=720 \
    env.room_path=rooms/C_maze.json,rooms/medium_maze.json \
    env.goal_radius="${GOAL_RADIUS}" \
    env.colour_mode=off \
    env.goal_mode=random \
    algorithm.max_steps=4096 \
    algorithm.lr="${MLP_LR}" \
    wandb.group="${GROUP_NAME}" \
    seed=0,1,2 \
    storage="${STORAGE}" \
    timestamp="${NOW}" \
    env.gamma=0.999 \
    module_output_activation=true \
    || echo "Some training jobs failed, plotting the seeds that finished" >&2

python scripts/evaluate_metrics.py \
    wandb.group="${GROUP_NAME}" \
    storage="${STORAGE}" \
    timestamp="${NOW}"
