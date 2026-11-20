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

# E4 Stage 3, final: the best combined config, every leave-one-theme-out and Stage 1's baseline, on the final seed
# set (plan: E4 "Stages" and "Stage 2 result"). Only the action theme's leader (discrete 3) differs from Stage 2's
# base; the encoder (CNN), reward (collision penalty 0.01), capacity (width 64), FOV (180) and memory (MLP backbone)
# leaders are the base itself, and no aux level beat its reference. So leaving a theme out reverts it to Stage 1's
# value, and the capacity, FOV, memory and aux themes have nothing to leave out:
#   best      CNN, discrete 3, collision penalty 0.01      (= Stage 2's discrete 3 config)
#   -action   CNN, continuous, collision penalty 0.01      (= Stage 2's base)
#   -encoder  MLP encoder, discrete 3, collision penalty 0.01
#   -reward   CNN, discrete 3, no collision penalty
#   baseline  MLP encoder, continuous, no collision penalty (Stage 1's base)
# 5 configs x 2 rooms x 10 seeds = 100 runs. Compare best against each other config with Welch's t, alpha 0.01, Holm
#   sbatch ... scripts/e4_stage3.sh
GROUP_NAME=e4_stage3
STORAGE=${SCRATCH}/${GROUP_NAME}
SEEDS=(1000 1001 1002 1003 1004 1005 1006 1007 1008 1009)     # final-claim seed set, never used before
ROOMS=("rooms/medium_maze.json" "rooms/hard_maze.json")
GOAL_RADIUS=16
MLP_LR=1e-4
NOW=$(date +"%Y%m%d_%H%M%S")

# Keep W&B and Hydra off the home folder's file quota
export WANDB_DIR=${STORAGE}/wandb
export WANDB_CACHE_DIR=${STORAGE}/wandb/cache
export WANDB_DATA_DIR=${STORAGE}/wandb/data
mkdir -p "${WANDB_DIR}" "${WANDB_CACHE_DIR}" "${WANDB_DATA_DIR}"

join() { local IFS=","; echo "$*"; }

DISCRETE="env.action_type=discrete env.discrete_actions=3"
CONFIGS=(
    "best|observation=cnn_observation env.collision_penalty=0.01 ${DISCRETE}"
    "no_action|observation=cnn_observation env.collision_penalty=0.01"
    "no_encoder|observation=mlp_observation env.collision_penalty=0.01 ${DISCRETE}"
    "no_reward|observation=cnn_observation env.collision_penalty=0.0 ${DISCRETE}"
    "baseline|observation=mlp_observation env.collision_penalty=0.0"
)

# Time limit 24 h: the CNN encoder runs 17-49 s/it depending on the node, so a seed that never stops early takes
# up to ~14 h, and submitit can't requeue on Euler
run() {
    local name=$1; shift
    python scripts/train.py --multirun \
        hydra/launcher=euler \
        hydra.sweep.dir="${STORAGE}/multirun/${NOW}/${name}" \
        hydra.launcher.submitit_folder="${STORAGE}/logs/%j" \
        hydra.launcher.timeout_min=1440 \
        env.room_path="$(join "${ROOMS[@]}")" \
        env.goal_radius="${GOAL_RADIUS}" \
        env.colour_mode=off \
        env.goal_mode=random \
        algorithm.max_steps=4096 \
        algorithm.lr="${MLP_LR}" \
        wandb.group="${GROUP_NAME}" \
        seed="$(join "${SEEDS[@]}")" \
        storage="${STORAGE}" \
        timestamp="${NOW}" \
        env.gamma=0.999 \
        env.fov=180 \
        "$@" \
        || echo "Some training jobs failed, plotting the seeds that finished" >&2
}

cd "${PROJECT_ROOT}"
for config in "${CONFIGS[@]}"; do
    # shellcheck disable=SC2086
    run "${config%%|*}" ${config#*|} &
done
wait

python scripts/evaluate_metrics.py \
    wandb.group="${GROUP_NAME}" \
    storage="${STORAGE}" \
    timestamp="${NOW}" \
    hydra.run.dir="${STORAGE}/hydra" \
    hydra.output_subdir=null
