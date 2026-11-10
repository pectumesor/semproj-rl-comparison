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

# E1 information-leak audit: does the MLP's lead come from the global Perlin place code?
# Conditions: (global colour, fixed goal) = current setup, (colour off, fixed goal), (colour off, random goal).
# GRU only runs where LSTM's IQM is below MLP's with non-overlapping CIs (pre-registered trigger rule).
GROUP_NAME="e1_place_code_audit"
SEEDS=(100 101 102 103 104)       # stage-2 seed set: screening seeds 0-2 never appear in reported results
# Learning rate per backbone from the E0 pilot (medium_maze AUC), the same as E4
BACKBONES=("mlp_backbone 1e-4" "lstm_backbone 3e-4")
SHORT_ROOMS=("rooms/two_room.json" "rooms/four_room.json")
LONG_ROOMS=("rooms/C_maze.json" "rooms/medium_maze.json")
GOAL_RADIUS=16
NOW=$(date +"%Y%m%d_%H%M%S")
STORAGE=${SCRATCH}/${GROUP_NAME}

# Keep W&B and Hydra off the home folder's file quota
export WANDB_DIR=${STORAGE}/wandb
export WANDB_CACHE_DIR=${STORAGE}/wandb/cache
export WANDB_DATA_DIR=${STORAGE}/wandb/data
mkdir -p "${WANDB_DIR}" "${WANDB_CACHE_DIR}" "${WANDB_DATA_DIR}"

# Every multirun is a Hydra process waiting on the launcher's 4G: at most this many at once
MAX_MULTIRUNS=6
throttle() { while (( $(jobs -rp | wc -l) >= MAX_MULTIRUNS )); do wait -n || true; done; }

join() { local IFS=","; echo "$*"; }

run() {
    local rooms=$1 episode_length=$2 gamma=$3 colour_modes=$4 goal_mode=$5 backbone=$6 lr=$7
    python scripts/train.py --multirun \
        hydra/launcher=euler \
        hydra.sweep.dir="${STORAGE}/multirun/${NOW}/${episode_length}_${goal_mode}_${backbone}" \
        hydra.launcher.submitit_folder="${STORAGE}/logs/%j" \
        hydra.launcher.timeout_min=1440 \
        env.room_path="${rooms}" \
        env.goal_radius="${GOAL_RADIUS}" \
        env.colour_mode="${colour_modes}" \
        env.goal_mode="${goal_mode}" \
        env.gamma="${gamma}" \
        algorithm.max_steps="${episode_length}" \
        backbone="${backbone}" \
        algorithm.lr="${lr}" \
        wandb.group="${GROUP_NAME}" \
        seed="$(join "${SEEDS[@]}")" \
        storage="${STORAGE}" \
        timestamp="${NOW}" \
        || echo "Some training jobs failed, plotting the seeds that finished" >&2
}

cd "${PROJECT_ROOT}"
for entry in "${BACKBONES[@]}"; do
    read -r backbone lr <<< "${entry}"
    throttle; run "$(join "${SHORT_ROOMS[@]}")" 512 0.99 global,off fixed "${backbone}" "${lr}" &
    throttle; run "$(join "${SHORT_ROOMS[@]}")" 512 0.99 off random "${backbone}" "${lr}" &
    throttle; run "$(join "${LONG_ROOMS[@]}")" 4096 0.999 global,off fixed "${backbone}" "${lr}" &
    throttle; run "$(join "${LONG_ROOMS[@]}")" 4096 0.999 off random "${backbone}" "${lr}" &
done
wait

# Aliasing score (E1a) of every room and colour mode, no training: x-axis of the memory-gain plot
python scripts/aliasing_score.py --multirun \
    env.room_path="$(join "${SHORT_ROOMS[@]}" "${LONG_ROOMS[@]}")" \
    env.colour_mode=global,off \
    env.goal_radius="${GOAL_RADIUS}" \
    storage="${STORAGE}" \
    hydra.sweep.dir="${STORAGE}/multirun/${NOW}/aliasing"

python scripts/evaluate_metrics.py \
    wandb.group="${GROUP_NAME}" \
    storage="${STORAGE}" \
    timestamp="${NOW}" \
    hydra.run.dir="${STORAGE}/hydra" \
    hydra.output_subdir=null

# Seed-level statistics for the paper: MLP vs LSTM within every (colour, goal) condition
python scripts/seed_stats.py \
    wandb.group="${GROUP_NAME}" \
    storage="${STORAGE}" \
    timestamp="${NOW}" \
    hydra.run.dir="${STORAGE}/hydra" \
    hydra.output_subdir=null \
    +stats.vary=backbone \
    +stats.reference=mlp

# Figure 2 (left): AUC per condition and room, memory gain vs aliasing score
python scripts/paper_figures.py \
    wandb.group="${GROUP_NAME}" \
    storage="${STORAGE}" \
    timestamp="${NOW}" \
    hydra.run.dir="${STORAGE}/hydra" \
    hydra.output_subdir=null \
    +figure.name=e1
