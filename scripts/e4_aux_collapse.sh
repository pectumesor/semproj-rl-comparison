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

# Why CNN + LSTM + loop closure stops moving (plan: "CNN + LSTM + loop closure collapse"). Stage 2 ran it at collision
# penalty 0.01 and coefficient 1.0, and 9 of 10 agents stood still. Each arm changes one of the two:
#   no_penalty: penalty 0, coefficient 1.0. A collision costs as much as 41 steps of time penalty, standing still avoids it
#   low_coeff:  penalty 0.01, coefficient 0.1. The aux gradient is 7-9x the task gradient in the LSTM and the encoder
# It's a diagnostic on Stage 2's seeds, not a level of the ablation: read the speed, value loss and success on W&B
# around iteration 200, the runs can be cancelled once the answer is clear. Stage 2's own runs at seeds 100-102 on
# medium_maze are the reference, so nothing here reruns them:
#   sbatch ... scripts/e4_aux_collapse.sh
GROUP_NAME=e4_aux_collapse
STORAGE=${SCRATCH}/${GROUP_NAME}
NOW=$(date +"%Y%m%d_%H%M%S")
GOAL_RADIUS=16
LSTM_LR=3e-4

# Keep W&B and Hydra off the home folder's file quota
export WANDB_DIR=${STORAGE}/wandb
export WANDB_CACHE_DIR=${STORAGE}/wandb/cache
export WANDB_DATA_DIR=${STORAGE}/wandb/data
mkdir -p "${WANDB_DIR}" "${WANDB_CACHE_DIR}" "${WANDB_DATA_DIR}"

run() {
    local name=$1 penalty=$2 coeff=$3
    python scripts/train.py --multirun \
        hydra/launcher=euler \
        hydra.sweep.dir="${STORAGE}/multirun/${NOW}/${name}" \
        hydra.launcher.submitit_folder="${STORAGE}/logs/%j" \
        hydra.launcher.timeout_min=1440 \
        env.room_path=rooms/medium_maze.json \
        env.goal_radius="${GOAL_RADIUS}" \
        env.colour_mode=off \
        env.goal_mode=random \
        algorithm.max_steps=4096 \
        algorithm.lr="${LSTM_LR}" \
        wandb.group="${GROUP_NAME}" \
        seed=100,101,102 \
        storage="${STORAGE}" \
        timestamp="${NOW}" \
        env.gamma=0.999 \
        observation=cnn_observation \
        env.fov=180 \
        env.collision_penalty="${penalty}" \
        backbone=lstm_backbone \
        head.auxiliary_head.enabled=true \
        head.auxiliary_head.depth_coeff=0.0 \
        head.auxiliary_head.loop_closure_coeff="${coeff}" \
        || echo "Some training jobs failed, plotting the seeds that finished" >&2
}

cd "${PROJECT_ROOT}"
run no_penalty 0.0 1.0 &
run low_coeff 0.01 0.1 &
wait

python scripts/evaluate_metrics.py \
    wandb.group="${GROUP_NAME}" \
    storage="${STORAGE}" \
    timestamp="${NOW}" \
    hydra.run.dir="${STORAGE}/hydra" \
    hydra.output_subdir=null
