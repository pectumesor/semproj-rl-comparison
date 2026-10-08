#!/usr/bin/env bash
set -euo pipefail

ETH_USERNAME=eazevedo
PROJECT_NAME=semproj
SCRATCH=/cluster/scratch/${ETH_USERNAME}/${PROJECT_NAME}
CONDA_ENVIRONMENT=semproj
CONDA_ROOT=${HOME}/miniforge3
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Checkpoints are loaded from ${LOG_ROOT}/<obs>_<backbone>_<algo>_seed_<s>/${RUN_NAME}/best.pt
# Override via: ./test_scatter_plot.sh [LOG_ROOT] [RUN_NAME] [TIMEOUT_MIN]
#   RUN_NAME=latest picks the newest checkpoint per seed
#   TIMEOUT_MIN is the SLURM time limit; every architecture x seed runs sequentially in one job
LOG_ROOT="${1:-${PROJECT_ROOT}/logs}"
RUN_NAME="${2:-26_09_20_bptt_model}"
TIMEOUT_MIN="${3:-720}"


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

cd "${PROJECT_ROOT}"


# Single SLURM job and single wandb run evaluating every architecture on all seeds.
# --multirun is only needed so the submitit launcher submits to SLURM; no override below
# is a comma sweep (the [..] lists are single values), so it expands to exactly one job.
python scripts/evaluate_metrics.py --multirun \
    hydra/launcher=euler \
    hydra.launcher.submitit_folder="${SCRATCH}/.submitit/%j" \
    hydra.launcher.timeout_min="${TIMEOUT_MIN}" \
    env.room_path="rooms/four_room.json" \
    env.goal_radius=16 \
    wandb.group=scatter_plot_test \
    "+eval.log_root=${LOG_ROOT}" \
    "+eval.run_name=${RUN_NAME}" \
    "+eval.seeds=[0,1,2,3,4]" \
    "+eval.observations=[mlp_observation,cnn_observation]" \
    "+eval.backbones=[mlp_backbone,lstm_backbone]"
