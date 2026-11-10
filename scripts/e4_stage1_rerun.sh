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

# Reruns the 22 E4 Stage 1 runs that hit the 4 h time limit (CNN encoder, width 256 and the LSTM aux
# levels, at 530-940 of 1000 iterations), with a 12 h limit. Same overrides, storage and timestamp as
# the Stage 1 sweep, so every run lands in its config's folder next to the seeds that finished:
#   sbatch ... scripts/e4_stage1_rerun.sh
GROUP_NAME=e4_stage1
STORAGE=${SCRATCH}/${GROUP_NAME}
NOW=20261102095921
GOAL_RADIUS=16
MLP_LR=1e-4
LSTM_LR=3e-4

# One multirun per room and config block, with only the seeds that timed out
run() {
    local room=$1 seeds=$2; shift 2
    python scripts/train.py --multirun \
        hydra/launcher=euler \
        hydra.launcher.submitit_folder="${STORAGE}/logs/%j" \
        hydra.launcher.timeout_min=720 \
        env.room_path="rooms/${room}.json" \
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
        "$@" \
        || echo "Some training jobs failed, plotting the seeds that finished" >&2
}

lstm_aux() {
    local room=$1 seeds=$2 depth_coeff=$3 loop_coeff=$4
    LR=${LSTM_LR} run "${room}" "${seeds}" backbone=lstm_backbone \
        'observation.masked_channels=[3]' head.auxiliary_head.enabled=true \
        head.auxiliary_head.depth_coeff="${depth_coeff}" head.auxiliary_head.loop_closure_coeff="${loop_coeff}"
}

cd "${PROJECT_ROOT}"
lstm_aux C_maze      0,1,2 1.0 0.0 &     # depth only
lstm_aux medium_maze 0,2   1.0 0.0 &
lstm_aux C_maze      0,1   0.0 1.0 &     # loop closure only
lstm_aux medium_maze 0,1,2 0.0 1.0 &
lstm_aux C_maze      0,1,2 1.0 1.0 &     # both
lstm_aux medium_maze 0,1,2 1.0 1.0 &
run medium_maze 2   observation=cnn_observation env.fov=180 &
run medium_maze 0,1 observation=cnn_observation env.fov=270 &
run medium_maze 0,1,2 observation.obs_embed_hidden_sizes=[256,256] backbone.backbone_hidden_sizes=[256,256] \
    head.policy_hidden_sizes=[256,256] head.value_hidden_sizes=[256,256] &
wait

python scripts/evaluate_metrics.py \
    wandb.group="${GROUP_NAME}" \
    storage="${STORAGE}" \
    timestamp="${NOW}"
