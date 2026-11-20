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

# E2 optional: K = 4 twin rooms (plan, E2: "K = 4 rooms at L in {32, 128} for the 3 best backbones"). The cue picks
# one of four identical rooms behind a two-level junction (scripts/generate_twin_rooms.py --rooms 4), so a memoryless
# policy sits at 0.25 and the recurrent state has to keep two bits instead of one. Start, corridor, first junction,
# shaping and episode length are E2's, so E2's K = 2 seeds at the same L are the reference. The 3 best backbones of E2 by solved seeds over L = 8-256: LSTM-BPTT256 (19/25), GRU-BPTT64 (18/25),
# LSTM-BPTT64 (14/25, ahead of BPTT16 at 13/25). No memoryless backbone: E2 has them at chance from L = 32 on.
# To rerun only the diagnostics on a finished sweep: bash scripts/e2_k4.sh <timestamp>
GROUP_NAME="e2_twin_rooms_k4"
SEEDS=(100 101 102 103 104)       # stage-2 seed set, the same as E2
MEMORY_HORIZONS=(32 128)
JUNCTION_SHAPING=1.0
RECURRENT_LR=3e-4
GOAL_RADIUS=16
GAMMA=0.999
# Per condition and seed: at 1000 the 95% interval of a rate near 0.25 is about +-0.03
DIAGNOSTIC_EPISODES=1000
# A timestamp argument skips training and only runs the diagnostics on that sweep
DIAGNOSE_ONLY=${1:-}
NOW=${DIAGNOSE_ONLY:-$(date +"%Y%m%d_%H%M%S")}
STORAGE=${SCRATCH}/${GROUP_NAME}

# Keep W&B and Hydra off the home folder's file quota
export WANDB_DIR=${STORAGE}/wandb
export WANDB_CACHE_DIR=${STORAGE}/wandb/cache
export WANDB_DATA_DIR=${STORAGE}/wandb/data
mkdir -p "${WANDB_DIR}" "${WANDB_CACHE_DIR}" "${WANDB_DATA_DIR}"

join() { local IFS=","; echo "$*"; }

# The E2 setting shared by training and diagnostics, in the K = 4 room
twin_rooms() {
    local L=$1
    echo env.room_path="rooms/twin_rooms_K4_L${L}.json" env.goal_mode=regions env.colour_mode=off \
        env.goal_radius="${GOAL_RADIUS}" env.gamma="${GAMMA}" env.junction_shaping="${JUNCTION_SHAPING}" \
        algorithm.max_steps="$(( 2 * L + 256 ))"
}

run() {
    local L=$1 name=$2 timeout=$3; shift 3
    python scripts/train.py --multirun \
        hydra/launcher=euler \
        hydra.sweep.dir="${STORAGE}/multirun/${NOW}/L${L}_${name}" \
        hydra.launcher.submitit_folder="${STORAGE}/logs/%j" \
        hydra.launcher.timeout_min="${timeout}" \
        $(twin_rooms "${L}") \
        wandb.group="${GROUP_NAME}" \
        seed="$(join "${SEEDS[@]}")" \
        storage="${STORAGE}" \
        timestamp="${NOW}" \
        "$@" \
        || echo "Some training jobs failed, diagnosing the seeds that finished" >&2
}

# One diagnostics job per config, it loops over the config's seeds itself
diagnose() {
    local L=$1 name=$2 timeout=$3; shift 3
    python scripts/e2_diagnostics.py --multirun \
        hydra/launcher=euler \
        hydra.sweep.dir="${STORAGE}/multirun/${NOW}/diagnose_L${L}_${name}" \
        hydra.launcher.submitit_folder="${STORAGE}/logs/%j" \
        hydra.launcher.timeout_min=120 \
        $(twin_rooms "${L}") \
        +diagnostics.n_episodes="${DIAGNOSTIC_EPISODES}" \
        storage="${STORAGE}" \
        timestamp="${NOW}" \
        "$@" \
        || echo "Some diagnostics jobs failed" >&2
}

# backbone name|training timeout (min)|overrides. Window and envs per minibatch as in E2 (128 updates of 1024
# samples per iteration); window 256 is about 7x slower per update, and an unsolved E2 run took ~11 h at L = 128
BACKBONES=(
    "lstm256|2880|backbone=lstm_backbone algorithm.lr=${RECURRENT_LR} backbone.bptt_window_size=256 algorithm.minibatch_size=4"
    "gru64|1440|backbone=gru_backbone algorithm.lr=${RECURRENT_LR} backbone.bptt_window_size=64 algorithm.minibatch_size=16"
    "lstm64|1440|backbone=lstm_backbone algorithm.lr=${RECURRENT_LR} backbone.bptt_window_size=64 algorithm.minibatch_size=16"
)

# step runs or diagnoses every L x backbone
for_all() {
    local step=$1
    for L in "${MEMORY_HORIZONS[@]}"; do
        for backbone in "${BACKBONES[@]}"; do
            IFS="|" read -r name timeout overrides <<< "${backbone}"
            "${step}" "${L}" "${name}" "${timeout}" ${overrides} &
        done
    done
    wait
}

cd "${PROJECT_ROOT}"
python scripts/generate_twin_rooms.py --rooms 4 --L "${MEMORY_HORIZONS[@]}"

if [[ -z "${DIAGNOSE_ONLY}" ]]; then
    for_all run
fi
for_all diagnose

python scripts/evaluate_metrics.py \
    wandb.group="${GROUP_NAME}" \
    storage="${STORAGE}" \
    timestamp="${NOW}" \
    hydra.run.dir="${STORAGE}/hydra" \
    hydra.output_subdir=null

# Seed-level statistics: every pair of backbones per L (one room per L)
python scripts/seed_stats.py \
    wandb.group="${GROUP_NAME}" \
    storage="${STORAGE}" \
    timestamp="${NOW}" \
    hydra.run.dir="${STORAGE}/hydra" \
    hydra.output_subdir=null
