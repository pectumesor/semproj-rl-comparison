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

# E3T rerun of C64 with a random telling phase (plan: "C64 timing check"). With the tellings on steps 0, 64, 128, ...
# the agents reach the junction 50-79 steps into an episode at L = 32, so the second telling lands at the decision
# and C64 doesn't test memory. Here each episode's first telling falls at a uniform step in 0..63, then every 64
# steps, so the last telling before the choice is anywhere from 0 to 63 steps old. Same room, settings, backbones
# and seeds as E3T; the junction latch keeps E3T's version, so the 90 other E3T runs stay the reference.
# To rerun only the diagnostics on a finished sweep: bash scripts/e3t_random_phase.sh <timestamp>
GROUP_NAME="e3_twin_rooms"
SEEDS=(100 101 102 103 104)       # stage-2 seed set, the same as E3T
L=${E3T_L:-32}
CUE_INTERVAL=64
JUNCTION_SHAPING=1.0
MLP_LR=1e-4
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

# E3T's setting with the C64 telling at a random phase, shared by training and diagnostics
twin_rooms() {
    echo env.room_path="rooms/twin_rooms_L${L}.json" env.goal_mode=regions env.colour_mode=off \
        env.goal_radius="${GOAL_RADIUS}" env.gamma="${GAMMA}" env.junction_shaping="${JUNCTION_SHAPING}" \
        algorithm.max_steps="${EPISODE_LENGTH}" env.information.reserve_info_slots=true \
        env.information.cue_info=every env.information.cue_interval="${CUE_INTERVAL}" \
        env.information.cue_phase=random
}

BACKBONES=(
    "mlp|backbone=mlp_backbone algorithm.lr=${MLP_LR}"
    "fs16|backbone=mlp_backbone algorithm.lr=${MLP_LR} observation.frame_stack.enabled=true observation.frame_stack.size=16 hydra.launcher.mem_per_cpu=4G"
    "lstm16|backbone=lstm_backbone algorithm.lr=${RECURRENT_LR} backbone.bptt_window_size=16 algorithm.minibatch_size=64"
)

run() {
    local name=$1; shift
    python scripts/train.py --multirun \
        hydra/launcher=euler \
        hydra.sweep.dir="${STORAGE}/multirun/${NOW}/C${CUE_INTERVAL}r_${name}" \
        hydra.launcher.submitit_folder="${STORAGE}/logs/%j" \
        hydra.launcher.timeout_min=1440 \
        $(twin_rooms) \
        wandb.group="${GROUP_NAME}" \
        seed="$(join "${SEEDS[@]}")" \
        storage="${STORAGE}" \
        timestamp="${NOW}" \
        "$@" \
        || echo "Some training jobs failed, diagnosing the seeds that finished" >&2
}

diagnose() {
    local name=$1; shift
    python scripts/e2_diagnostics.py --multirun \
        hydra/launcher=euler \
        hydra.sweep.dir="${STORAGE}/multirun/${NOW}/diagnose_C${CUE_INTERVAL}r_${name}" \
        hydra.launcher.submitit_folder="${STORAGE}/logs/%j" \
        hydra.launcher.timeout_min=120 \
        $(twin_rooms) \
        +diagnostics.n_episodes="${DIAGNOSTIC_EPISODES}" \
        storage="${STORAGE}" \
        timestamp="${NOW}" \
        "$@" \
        || echo "Some diagnostics jobs failed" >&2
}

cd "${PROJECT_ROOT}"
python scripts/generate_twin_rooms.py --L "${L}"

if [[ -z "${DIAGNOSE_ONLY}" ]]; then
    for backbone in "${BACKBONES[@]}"; do
        run "${backbone%%|*}" ${backbone#*|} &
    done
    wait
fi
for backbone in "${BACKBONES[@]}"; do
    diagnose "${backbone%%|*}" ${backbone#*|} &
done
wait

# No seed_stats or figure here: the other E3T conditions live under E3T's timestamp, compare against them by hand
python scripts/evaluate_metrics.py \
    wandb.group="${GROUP_NAME}" \
    storage="${STORAGE}" \
    timestamp="${NOW}" \
    hydra.run.dir="${STORAGE}/hydra" \
    hydra.output_subdir=null
