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

# E1b field of view as a perception dial: narrowing the view should increase the memory gain.
# env.max_turn_deg stays at 90, so the FOV no longer changes the action space.
# LSTM is the recurrent backbone; swap in GRU only if E2 shows it clearly better.
GROUP_NAME="e1b_fov"
SEEDS=(100 101 102 103 104)       # stage-2 seed set: screening seeds 0-2 never appear in reported results
# Learning rate per backbone from the E0 pilot (medium_maze AUC), the same as E4
BACKBONES=("mlp_backbone 1e-4" "lstm_backbone 3e-4")
FOVS=(90 180 270 360)
ROOM_PATHS=("rooms/C_maze.json")
COLOUR_MODE=off
GOAL_MODE=random
GOAL_RADIUS=16
GAMMA=0.999
EPISODE_LENGTH=4096
NOW=$(date +"%Y%m%d_%H%M%S")
STORAGE=${SCRATCH}/${GROUP_NAME}

# Keep W&B and Hydra off the home folder's file quota
export WANDB_DIR=${STORAGE}/wandb
export WANDB_CACHE_DIR=${STORAGE}/wandb/cache
export WANDB_DATA_DIR=${STORAGE}/wandb/data
mkdir -p "${WANDB_DIR}" "${WANDB_CACHE_DIR}" "${WANDB_DATA_DIR}"

join() { local IFS=","; echo "$*"; }

cd "${PROJECT_ROOT}"
for entry in "${BACKBONES[@]}"; do
    read -r backbone lr <<< "${entry}"
    python scripts/train.py --multirun \
        hydra/launcher=euler \
        hydra.sweep.dir="${STORAGE}/multirun/${NOW}/${backbone}" \
        hydra.launcher.submitit_folder="${STORAGE}/logs/%j" \
        hydra.launcher.timeout_min=1440 \
        env.room_path="$(join "${ROOM_PATHS[@]}")" \
        env.goal_radius="${GOAL_RADIUS}" \
        env.colour_mode="${COLOUR_MODE}" \
        env.goal_mode="${GOAL_MODE}" \
        env.fov="$(join "${FOVS[@]}")" \
        env.gamma="${GAMMA}" \
        algorithm.max_steps="${EPISODE_LENGTH}" \
        backbone="${backbone}" \
        algorithm.lr="${lr}" \
        wandb.group="${GROUP_NAME}" \
        seed="$(join "${SEEDS[@]}")" \
        storage="${STORAGE}" \
        timestamp="${NOW}" \
        || echo "Some training jobs failed, plotting the seeds that finished" >&2 &
done
wait

python scripts/aliasing_score.py --multirun \
    env.room_path="$(join "${ROOM_PATHS[@]}")" \
    env.colour_mode="${COLOUR_MODE}" \
    env.goal_mode="${GOAL_MODE}" \
    env.goal_radius="${GOAL_RADIUS}" \
    env.fov="$(join "${FOVS[@]}")" \
    storage="${STORAGE}" \
    hydra.sweep.dir="${STORAGE}/multirun/${NOW}/aliasing"

python scripts/evaluate_metrics.py \
    wandb.group="${GROUP_NAME}" \
    storage="${STORAGE}" \
    timestamp="${NOW}" \
    hydra.run.dir="${STORAGE}/hydra" \
    hydra.output_subdir=null

# Seed-level statistics for the paper: MLP vs LSTM at every FOV
python scripts/seed_stats.py \
    wandb.group="${GROUP_NAME}" \
    storage="${STORAGE}" \
    timestamp="${NOW}" \
    hydra.run.dir="${STORAGE}/hydra" \
    hydra.output_subdir=null \
    +stats.vary=backbone \
    +stats.reference=mlp

# Crossover plot: AUC vs field of view per backbone
python scripts/paper_figures.py \
    wandb.group="${GROUP_NAME}" \
    storage="${STORAGE}" \
    timestamp="${NOW}" \
    hydra.run.dir="${STORAGE}/hydra" \
    hydra.output_subdir=null \
    +figure.name=e1b
