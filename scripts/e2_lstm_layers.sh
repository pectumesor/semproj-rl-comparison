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

# E2 follow-up: does the number of stacked LSTM layers change memory? Every recurrent run so far has 2 layers
# (configs/backbone/lstm_backbone.yaml). Layers stack per step and don't lengthen the gradient path in time,
# so the BPTT window, not depth, should decide whether the cue is learned. Runs 1 and 3 layers of LSTM-BPTT16
# at L = 64, where 2 layers solve 2 of 5 seeds; E2's 2-layer seeds (timestamp 20261110145407) are the reference.
# Own timestamp, so E2's sweep and its figure keep their rows. To rerun only the diagnostics on a finished
# sweep: bash scripts/e2_lstm_layers.sh <timestamp>
GROUP_NAME="e2_twin_rooms"
SEEDS=(100 101 102 103 104)       # stage-2 seed set, the same as E2's 2-layer reference
L=${E2_LAYERS_L:-64}
LSTM_LAYERS=(1 3)
# E2's settings for LSTM-BPTT16
JUNCTION_SHAPING=1.0
WINDOW=16
MINIBATCH=64
RECURRENT_LR=3e-4
GOAL_RADIUS=16
GAMMA=0.999
DIAGNOSTIC_EPISODES=1000
# A timestamp argument skips training and only runs the diagnostics on that sweep
DIAGNOSE_ONLY=${1:-}
NOW=${DIAGNOSE_ONLY:-$(date +"%Y%m%d_%H%M%S")}
STORAGE=${SCRATCH}/${GROUP_NAME}
EPISODE_LENGTH=$(( 2 * L + 256 ))

# Keep W&B and Hydra off the home folder's file quota
export WANDB_DIR=${STORAGE}/wandb
export WANDB_CACHE_DIR=${STORAGE}/wandb/cache
export WANDB_DATA_DIR=${STORAGE}/wandb/data
mkdir -p "${WANDB_DIR}" "${WANDB_CACHE_DIR}" "${WANDB_DATA_DIR}"

join() { local IFS=","; echo "$*"; }

# The E2 setting and the LSTM, shared by training and diagnostics
lstm_twin_rooms() {
    echo env.room_path="rooms/twin_rooms_L${L}.json" env.goal_mode=regions env.colour_mode=off \
        env.goal_radius="${GOAL_RADIUS}" env.gamma="${GAMMA}" env.junction_shaping="${JUNCTION_SHAPING}" \
        algorithm.max_steps="${EPISODE_LENGTH}" backbone=lstm_backbone algorithm.lr="${RECURRENT_LR}" \
        backbone.bptt_window_size="${WINDOW}" algorithm.minibatch_size="${MINIBATCH}" \
        backbone.lstm_num_layers="$(join "${LSTM_LAYERS[@]}")"
}

cd "${PROJECT_ROOT}"
python scripts/generate_twin_rooms.py --L "${L}"

if [[ -z "${DIAGNOSE_ONLY}" ]]; then
    python scripts/train.py --multirun \
        hydra/launcher=euler \
        hydra.sweep.dir="${STORAGE}/multirun/${NOW}/L${L}_lstm_layers" \
        hydra.launcher.submitit_folder="${STORAGE}/logs/%j" \
        hydra.launcher.timeout_min=1440 \
        $(lstm_twin_rooms) \
        wandb.group="${GROUP_NAME}" \
        seed="$(join "${SEEDS[@]}")" \
        storage="${STORAGE}" \
        timestamp="${NOW}" \
        || echo "Some training jobs failed, diagnosing the seeds that finished" >&2
fi

# One diagnostics job per layer count, it loops over the seeds itself
python scripts/e2_diagnostics.py --multirun \
    hydra/launcher=euler \
    hydra.sweep.dir="${STORAGE}/multirun/${NOW}/diagnose_L${L}_lstm_layers" \
    hydra.launcher.submitit_folder="${STORAGE}/logs/%j" \
    hydra.launcher.timeout_min=120 \
    $(lstm_twin_rooms) \
    +diagnostics.n_episodes="${DIAGNOSTIC_EPISODES}" \
    storage="${STORAGE}" \
    timestamp="${NOW}" \
    || echo "Some diagnostics jobs failed" >&2

# No seed_stats or figure here: the 2-layer reference lives under E2's timestamp, compare against it by hand
python scripts/evaluate_metrics.py \
    wandb.group="${GROUP_NAME}" \
    storage="${STORAGE}" \
    timestamp="${NOW}" \
    hydra.run.dir="${STORAGE}/hydra" \
    hydra.output_subdir=null
