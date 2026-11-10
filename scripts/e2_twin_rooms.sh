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

# E2 twin rooms with a cue (headline experiment): success vs memory horizon L per backbone.
# The cue colour in the start alcove says which of two identical rooms holds the goal; entering the
# wrong room ends the episode, so a memoryless policy sits at 0.5. Rooms: scripts/generate_twin_rooms.py.
# GRU always runs here: it is the head-to-head LSTM vs GRU check of the trigger rule.
# After training, scripts/e2_diagnostics.py runs on every config (resets at the junction and along the corridor,
# confusion matrix, linear probe, mirrored pose, state/pose swaps). To rerun only the diagnostics on a finished
# sweep: bash scripts/e2_twin_rooms.sh <timestamp>
GROUP_NAME="e2_twin_rooms"
SEEDS=(100 101 102 103 104)       # stage-2 seed set: screening seeds 0-2 never appear in reported results
# The pilot and the full sweep share a timestamp, so the pilot's L = 32 seeds count for the sweep:
#   E2_HORIZONS=32 sbatch ... scripts/e2_twin_rooms.sh                                 (pilot)
#   E2_HORIZONS="8 64 128 256" E2_TIMESTAMP=<pilot timestamp> sbatch ... scripts/e2_twin_rooms.sh
# The unshaped pilot (timestamp 20261107000000, E2_SHAPING=0) stays as the reference: most of its seeds never
# got out of the alcove and corridor, whatever the backbone
read -r -a MEMORY_HORIZONS <<< "${E2_HORIZONS:-8 32 64 128 256}"
# Potential-based shaping toward the junction, cue-free (env.junction_shaping in configs/env/base_env.yaml)
JUNCTION_SHAPING=${E2_SHAPING:-1.0}
# Window and envs per minibatch go together: 1024 / window keeps every recurrent config at 128 updates of 1024
# samples per iteration, at a fixed minibatch the update count would fall with the window (128, 32, 8)
LSTM_WINDOWS=("16 64" "64 16" "256 4")
FRAME_STACK_SIZES=(4 16)
GRU_WINDOW="64 16"
# Learning rate per backbone from the E0 pilot, the same as E4. The GRU and the frame stacks had no pilot:
# the GRU takes the LSTM's and the frame stacks the MLP's
MLP_LR=1e-4
RECURRENT_LR=3e-4
GOAL_RADIUS=16
GAMMA=0.999
# Per condition and seed: at 1000 the 95% interval of a rate near 0.5 is about +-0.03 (+-0.07 at 200)
DIAGNOSTIC_EPISODES=1000
# A timestamp argument skips training and only runs the diagnostics on that sweep
DIAGNOSE_ONLY=${1:-}
NOW=${DIAGNOSE_ONLY:-${E2_TIMESTAMP:-$(date +"%Y%m%d_%H%M%S")}}
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

# One diagnostics job per config, it loops over the config's seeds itself (a few minutes per seed)
diagnose() {
    local L=$1 name=$2; shift 2
    local episode_length=$(( 2 * L + 256 ))
    python scripts/e2_diagnostics.py --multirun \
        hydra/launcher=euler \
        hydra.sweep.dir="${STORAGE}/multirun/${NOW}/diagnose_L${L}_${name}" \
        hydra.launcher.submitit_folder="${STORAGE}/logs/%j" \
        hydra.launcher.timeout_min=120 \
        env.room_path="rooms/twin_rooms_L${L}.json" \
        env.goal_mode=regions \
        env.colour_mode=off \
        env.goal_radius="${GOAL_RADIUS}" \
        env.gamma="${GAMMA}" \
        env.junction_shaping="${JUNCTION_SHAPING}" \
        algorithm.max_steps="${episode_length}" \
        +diagnostics.n_episodes="${DIAGNOSTIC_EPISODES}" \
        storage="${STORAGE}" \
        timestamp="${NOW}" \
        "$@" \
        || echo "Some diagnostics jobs failed" >&2
}

# Twin rooms: all walls look the same (colour off), only the cue walls are coloured
run() {
    local L=$1 name=$2; shift 2
    # Room to cross: L corridor steps plus the alcove, shaft, arm and room. Leave time to explore both ways
    local episode_length=$(( 2 * L + 256 ))
    python scripts/train.py --multirun \
        hydra/launcher=euler \
        hydra.sweep.dir="${STORAGE}/multirun/${NOW}/L${L}_${name}" \
        hydra.launcher.submitit_folder="${STORAGE}/logs/%j" \
        hydra.launcher.timeout_min="${TIMEOUT_MIN:-1440}" \
        env.room_path="rooms/twin_rooms_L${L}.json" \
        env.goal_mode=regions \
        env.colour_mode=off \
        env.goal_radius="${GOAL_RADIUS}" \
        env.gamma="${GAMMA}" \
        env.junction_shaping="${JUNCTION_SHAPING}" \
        algorithm.max_steps="${episode_length}" \
        wandb.group="${GROUP_NAME}" \
        seed="$(join "${SEEDS[@]}")" \
        storage="${STORAGE}" \
        timestamp="${NOW}" \
        "$@" \
        || echo "Some training jobs failed, plotting the seeds that finished" >&2
}

cd "${PROJECT_ROOT}"
python scripts/generate_twin_rooms.py --L "${MEMORY_HORIZONS[@]}"

if [[ -z "${DIAGNOSE_ONLY}" ]]; then
    for L in "${MEMORY_HORIZONS[@]}"; do
        throttle; run "${L}" mlp backbone=mlp_backbone algorithm.lr="${MLP_LR}" &
        # 16 stacked frames need about 1.1 GB of rollout buffer alone, more than the default 2G can hold
        throttle; run "${L}" frame_stack backbone=mlp_backbone algorithm.lr="${MLP_LR}" observation.frame_stack.enabled=true \
            observation.frame_stack.size="$(join "${FRAME_STACK_SIZES[@]}")" hydra.launcher.mem_per_cpu=4G &
        for entry in "${LSTM_WINDOWS[@]}"; do
            read -r window minibatch <<< "${entry}"
            # About 7x the update time of window 16: on a slow node 24 h can be too short, and a killed run saves no auc.npy
            throttle; TIMEOUT_MIN=$(( window == 256 ? 2880 : 1440 )) run "${L}" "lstm_${window}" backbone=lstm_backbone algorithm.lr="${RECURRENT_LR}" \
                backbone.bptt_window_size="${window}" algorithm.minibatch_size="${minibatch}" &
        done
        read -r window minibatch <<< "${GRU_WINDOW}"
        throttle; run "${L}" gru backbone=gru_backbone algorithm.lr="${RECURRENT_LR}" \
            backbone.bptt_window_size="${window}" algorithm.minibatch_size="${minibatch}" &
    done
    wait
fi

# Same configs as above: memoryless ones only get success + confusion matrix, recurrent ones also
# the reset at the junction and the linear probe
for L in "${MEMORY_HORIZONS[@]}"; do
    throttle; diagnose "${L}" mlp backbone=mlp_backbone algorithm.lr="${MLP_LR}" &
    throttle; diagnose "${L}" frame_stack backbone=mlp_backbone algorithm.lr="${MLP_LR}" observation.frame_stack.enabled=true \
        observation.frame_stack.size="$(join "${FRAME_STACK_SIZES[@]}")" &
    for entry in "${LSTM_WINDOWS[@]}"; do
        read -r window minibatch <<< "${entry}"
        throttle; diagnose "${L}" "lstm_${window}" backbone=lstm_backbone algorithm.lr="${RECURRENT_LR}" \
            backbone.bptt_window_size="${window}" algorithm.minibatch_size="${minibatch}" &
    done
    read -r window minibatch <<< "${GRU_WINDOW}"
    throttle; diagnose "${L}" gru backbone=gru_backbone algorithm.lr="${RECURRENT_LR}" \
        backbone.bptt_window_size="${window}" algorithm.minibatch_size="${minibatch}" &
done
wait

python scripts/evaluate_metrics.py \
    wandb.group="${GROUP_NAME}" \
    storage="${STORAGE}" \
    timestamp="${NOW}" \
    hydra.run.dir="${STORAGE}/hydra" \
    hydra.output_subdir=null

# Seed-level statistics for the paper: every backbone vs the plain MLP, per memory horizon L (one room per L)
python scripts/seed_stats.py \
    wandb.group="${GROUP_NAME}" \
    storage="${STORAGE}" \
    timestamp="${NOW}" \
    hydra.run.dir="${STORAGE}/hydra" \
    hydra.output_subdir=null \
    +stats.reference=backbone:mlp_frame_stack:False

# Figure 3 and the diagnostics table
python scripts/paper_figures.py \
    wandb.group="${GROUP_NAME}" \
    storage="${STORAGE}" \
    timestamp="${NOW}" \
    hydra.run.dir="${STORAGE}/hydra" \
    hydra.output_subdir=null \
    +figure.name=e2
