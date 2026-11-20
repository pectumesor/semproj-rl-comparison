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

# E3T at L = 64 with a random telling phase (plan: "E3T at L = 64"). At L = 32 FS16 solves C64r by body memory
# (pose probe 0.96-1.00) and the fixed phase put a telling at the junction, so E3T there can't tell remembering
# from being told. At L = 64 E2's FS16 never solves (pose probe at chance) and LSTM-BPTT256 solves 5/5, so the
# tellings every 4, 16 or 64 steps (first one at a uniform step in 0..N-1) separate the backbones cleanly.
# BPTT256 instead of E3T's BPTT16, which solves only 3/5 at L = 64. C1, G1 and P1 are left out: at L = 32 they
# were at the ceiling (C1, G1) or answered (P1). The training junction latch counts the corner cut (fixed Nov 17).
# To rerun only the diagnostics on a finished sweep: bash scripts/e3t_L64.sh <timestamp>
GROUP_NAME="e3_twin_rooms"
SEEDS=(100 101 102 103 104)       # stage-2 seed set, the same as E2 and E3T
L=64
CUE_INTERVALS=(4 16 64)
JUNCTION_SHAPING=1.0
# Learning rates as in E2: the frame stack takes the MLP's
MLP_LR=1e-4
RECURRENT_LR=3e-4
GOAL_RADIUS=16
GAMMA=0.999
# Per condition and seed: at 1000 the 95% interval of a rate near 0.5 is about +-0.03
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

# The E2 setting shared by training and diagnostics
twin_rooms() {
    echo env.room_path="rooms/twin_rooms_L${L}.json" env.goal_mode=regions env.colour_mode=off \
        env.goal_radius="${GOAL_RADIUS}" env.gamma="${GAMMA}" env.junction_shaping="${JUNCTION_SHAPING}" \
        algorithm.max_steps="${EPISODE_LENGTH}" env.information.reserve_info_slots=true
}

run() {
    local name=$1; shift
    python scripts/train.py --multirun \
        hydra/launcher=euler \
        hydra.sweep.dir="${STORAGE}/multirun/${NOW}/${name}" \
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

# One diagnostics job per config, it loops over the config's seeds itself
diagnose() {
    local name=$1; shift
    python scripts/e2_diagnostics.py --multirun \
        hydra/launcher=euler \
        hydra.sweep.dir="${STORAGE}/multirun/${NOW}/diagnose_${name}" \
        hydra.launcher.submitit_folder="${STORAGE}/logs/%j" \
        hydra.launcher.timeout_min=120 \
        $(twin_rooms) \
        +diagnostics.n_episodes="${DIAGNOSTIC_EPISODES}" \
        storage="${STORAGE}" \
        timestamp="${NOW}" \
        "$@" \
        || echo "Some diagnostics jobs failed" >&2
}

# backbone name -> its overrides; 16 stacked frames need about 1.1 GB of rollout buffer, more than the default 2G.
# Window 256 with 4 envs per minibatch keeps E2's 128 updates of 1024 samples per iteration
BACKBONES=(
    "mlp|backbone=mlp_backbone algorithm.lr=${MLP_LR}"
    "fs16|backbone=mlp_backbone algorithm.lr=${MLP_LR} observation.frame_stack.enabled=true observation.frame_stack.size=16 hydra.launcher.mem_per_cpu=4G"
    "lstm256|backbone=lstm_backbone algorithm.lr=${RECURRENT_LR} backbone.bptt_window_size=256 algorithm.minibatch_size=4"
)
# condition name -> its information overrides (the C<N>r share one multirun over the intervals)
CONDITIONS=(
    "cue|env.information.cue_info=every env.information.cue_interval=$(join "${CUE_INTERVALS[@]}") env.information.cue_phase=random"
    "I0|env.information.cue_info=reserved"
)

# step runs or diagnoses every condition x backbone
for_all() {
    local step=$1
    for backbone in "${BACKBONES[@]}"; do
        for condition in "${CONDITIONS[@]}"; do
            "${step}" "${condition%%|*}_${backbone%%|*}" ${condition#*|} ${backbone#*|} &
        done
    done
    wait
}

cd "${PROJECT_ROOT}"
python scripts/generate_twin_rooms.py --L "${L}"

if [[ -z "${DIAGNOSE_ONLY}" ]]; then
    for_all run
fi
# The diagnostics jobs don't need the training memory: the frame stack's 4G override is harmless there
for_all diagnose

python scripts/evaluate_metrics.py \
    wandb.group="${GROUP_NAME}" \
    storage="${STORAGE}" \
    timestamp="${NOW}" \
    hydra.run.dir="${STORAGE}/hydra" \
    hydra.output_subdir=null

# Seed-level statistics: the LSTM and FS16 against the MLP in every condition
python scripts/seed_stats.py \
    wandb.group="${GROUP_NAME}" \
    storage="${STORAGE}" \
    timestamp="${NOW}" \
    hydra.run.dir="${STORAGE}/hydra" \
    hydra.output_subdir=null \
    +stats.vary=backbone \
    +stats.reference=mlp
