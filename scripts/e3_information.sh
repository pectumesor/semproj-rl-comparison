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

# E3 information sources ("be told"): goal and position information vs memory.
# Goal random per episode in every condition, colour off. reserve_info_slots keeps the input size
# the same in every condition (unused slots are zeros), so only the information changes.
# GRU only runs where LSTM's IQM is below the leader's with non-overlapping CIs (trigger rule).
GROUP_NAME="e3_information"
SEEDS=(100 101 102 103 104)       # stage-2 seed set: screening seeds 0-2 never appear in reported results
# Learning rate per backbone from the E0 pilot (medium_maze AUC), the same as E4
BACKBONES=("mlp_backbone 1e-4" "lstm_backbone 3e-4")
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

# Conditions as "goal_info position_info goal_grid"
CONDITIONS=(
    "none none false"                 # I0  nothing extra
    "oracle none false"               # G1  oracle goal ray every step (through walls)
    "last_seen_updated none false"    # G2a last sighting, updated by odometry
    "last_seen_frozen none false"     # G2b last sighting, frozen as it looked then
    "none xy false"                   # P1  raw (x, y)
    "none rff false"                  # P2  random Fourier features (same size as the grid code)
    "none grid false"                 # P3  hand-crafted hexagonal grid code
    "none noisy_grid false"           # P4  grid code of a noisy path-integrated position
    "none grid true"                  # P5  grid code of agent and goal (Banino-style vector navigation)
)

join() { local IFS=","; echo "$*"; }

run() {
    local room=$1 episode_length=$2 gamma=$3 goal_info=$4 position_info=$5 goal_grid=$6 backbone=$7 lr=$8
    python scripts/train.py --multirun \
        hydra/launcher=euler \
        hydra.sweep.dir="${STORAGE}/multirun/${NOW}/$(basename "${room}" .json)_${goal_info}_${position_info}_${goal_grid}_${backbone}" \
        hydra.launcher.submitit_folder="${STORAGE}/logs/%j" \
        hydra.launcher.timeout_min=1440 \
        env.room_path="${room}" \
        env.goal_radius="${GOAL_RADIUS}" \
        env.colour_mode=off \
        env.goal_mode=random \
        env.information.reserve_info_slots=true \
        env.information.goal_info="${goal_info}" \
        env.information.position_info="${position_info}" \
        env.information.goal_grid="${goal_grid}" \
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
for condition in "${CONDITIONS[@]}"; do
    read -r goal_info position_info goal_grid <<< "${condition}"
    for entry in "${BACKBONES[@]}"; do
        read -r backbone lr <<< "${entry}"
        throttle; run "rooms/four_room.json" 512 0.99 "${goal_info}" "${position_info}" "${goal_grid}" "${backbone}" "${lr}" &   # open layout
        throttle; run "rooms/C_maze.json" 4096 0.999 "${goal_info}" "${position_info}" "${goal_grid}" "${backbone}" "${lr}" &    # dead ends
    done
done
wait

python scripts/evaluate_metrics.py \
    wandb.group="${GROUP_NAME}" \
    storage="${STORAGE}" \
    timestamp="${NOW}" \
    hydra.run.dir="${STORAGE}/hydra" \
    hydra.output_subdir=null

# Seed-level statistics for the paper: MLP vs LSTM within every information condition (the memory gain of the condition)
python scripts/seed_stats.py \
    wandb.group="${GROUP_NAME}" \
    storage="${STORAGE}" \
    timestamp="${NOW}" \
    hydra.run.dir="${STORAGE}/hydra" \
    hydra.output_subdir=null \
    +stats.vary=backbone \
    +stats.reference=mlp

# Figure 4: AUC per information condition and backbone, memory gain row
python scripts/paper_figures.py \
    wandb.group="${GROUP_NAME}" \
    storage="${STORAGE}" \
    timestamp="${NOW}" \
    hydra.run.dir="${STORAGE}/hydra" \
    hydra.output_subdir=null \
    +figure.name=e3
