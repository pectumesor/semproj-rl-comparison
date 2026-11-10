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

# E4 staged ablation of the remaining design choices: every theme is varied around the base config
# (Stage 1: MLP encoder + MLP backbone, colour off, goal random; Stage 2: CNN encoder + collision penalty),
# the other themes stay fixed.
#   STAGE=1  screen    C_maze, medium_maze              seeds 0-2
#   STAGE=2  confirm   medium_maze, hard_maze           seeds 100-104
#   STAGE=3  final     medium_maze, hard_maze           seeds 1000-1009
# Every stage uses the hard tier: under the base config two_room, T_maze, I_maze, four_room and cross_room
# saturate (E0 room probe, padded AUC >= 0.96 for MLP and LSTM), so screening on them can't rank configs.
# C_maze still ranks by learning speed (AUC 0.93-0.96) at ~0.4 h per run. 8192-step episodes did worse than
# 4096 on C_maze and hard_maze (E0 episode-length probe), so every room keeps 4096.
# Between stages, prune each theme's arrays below to its survivors (leader + every config whose IQM CI
# overlaps it, at most ceil(n/3), plus one random wildcard; always keep the baseline) and set the
# leaders as the new base config. Run the probe panel on every tier for the rank-consistency check.
# The arrays hold the Stage 1 survivors; Stage 1 itself ran every level (see the plan's theme table).
#   bash scripts/e4_staged_ablation.sh 1
STAGE=${1:-1}
# The baseline runs once ("baseline"); every theme below only runs its levels that differ from it,
# otherwise e.g. action=continuous, collision_penalty=0.0 and width=64 would each rerun the baseline
# into the same storage folder
THEMES=("baseline" "observation" "action" "reward" "aux" "capacity")
GOAL_RADIUS=16
# Learning rate per backbone from the E0 pilot (medium_maze AUC): MLP 1e-4, LSTM 3e-4
MLP_LR=1e-4
LSTM_LR=3e-4
NOW=$(date +"%Y%m%d_%H%M%S")

# MATCHED_REFERENCES adds the aux-off configs with the same masking as the aux-on levels (masked MLP at
# the aux widths), the unmasked LSTM with and without loop closure, and the unmasked LSTM's BPTT windows,
# so e4_promotion.py can rank every aux and BPTT level against its matched reference. Stage 1 ran without them: its masked LSTM (AUC ~0.11 on both
# rooms, vs ~0.8 unmasked in the E0 probe) showed that hiding distances alone cripples the agent
case "${STAGE}" in
    1) SEEDS=(0 1 2);                        SHORT_ROOMS=(); LONG_ROOMS=("rooms/C_maze.json" "rooms/medium_maze.json"); MATCHED_REFERENCES=false ;;
    2) SEEDS=(100 101 102 103 104);          SHORT_ROOMS=(); LONG_ROOMS=("rooms/medium_maze.json" "rooms/hard_maze.json"); MATCHED_REFERENCES=true ;;
    3) SEEDS=(1000 1001 1002 1003 1004 1005 1006 1007 1008 1009); SHORT_ROOMS=(); LONG_ROOMS=("rooms/medium_maze.json" "rooms/hard_maze.json"); MATCHED_REFERENCES=true ;;
    *) echo "STAGE must be 1, 2 or 3" >&2; exit 1 ;;
esac
GROUP_NAME="e4_stage${STAGE}"
STORAGE=${SCRATCH}/${GROUP_NAME}

# -- Theme levels (Stage 1 survivors, prune to the survivors between stages) --
MLP_FOVS=(90 180)                                # A. observation: encoder x FOV (MLP 90 = wildcard)
CNN_FOVS=(180 270)
DISCRETE_ACTIONS=(3)                             # B. action space: continuous + discrete 3 (wildcard)
COLLISION_PENALTIES=(0.0 0.01)                   # C. reward: collision penalty, then gamma (0.99 = wildcard)
GAMMAS=(0.99 0.999)
AUX_LEVELS=("0.0 1.0" "1.0 1.0")                 # D. aux on the LSTM: depth coeff, loop closure coeff (loop = wildcard)
WIDTHS=(16 64 256)                               # E. capacity: width of every MLP (16 = wildcard)
AUX_WIDTHS=(16 64)                               #    depth aux (masked) at matched widths
DEPTH_LEVELS=("[64]")                            #    hidden layers per module: 1 (AUC IQM 0.87 vs 0.73, Stage 1 depth screen)
# Stage 1's leaders are the base config, except width 256 (AUC IQM 0.81 vs 0.73): on the CNN encoder it costs
# 1.8x per iteration, and it was never run with the CNN, so it's a capacity level that tests the combination
BASE_ENCODER=cnn_observation
BASE_FOV=180
BASE_WIDTH=64
BASE_COLLISION_PENALTY=0.01
# Stage 1's baseline and the aux levels stay on Stage 1's encoder and reward: aux on/off are ranked against
# each other anyway, this keeps them comparable with Stage 1, and a CNN + LSTM + aux run takes twice as long
STAGE1_ENCODER=mlp_observation
STAGE1_COLLISION_PENALTY=0.0
# BPTT windows of the unmasked LSTM (with MATCHED_REFERENCES). 16 is the default; gradients through 16 steps
# are short next to 4096-step episodes. The only earlier sweep (W&B group lstm_tbptt, Sep 2026) ran
# four_room with a fixed goal and a global colour place code, where memory isn't needed: every window
# converged, 128/256 reached CR 100% vs 92% and shorter paths, but learned slower
LSTM_BPTT_WINDOWS=(16 64 256)

join() { local IFS=","; echo "$*"; }

# Levels of a theme without the baseline's value: without 64 16 64 256 -> 16 256
without() {
    local skip=$1; shift
    local levels=()
    for level in "$@"; do
        [ "${level}" != "${skip}" ] && levels+=("${level}")
    done
    echo "${levels[@]}"
}

# FOV levels of an encoder, without the baseline cell
encoder_fovs() {
    local encoder=$1; shift
    if [ "${encoder}" = "${BASE_ENCODER}" ]; then without "${BASE_FOV}" "$@"; else echo "$@"; fi
}

# One multirun per tier and theme block, on top of the base config (LR=<lr>, ENCODER=<encoder>, FOV=<fov> or
# COLLISION_PENALTY=<penalty> run ... overrides it; hydra rejects the same key given twice).
# Time limit 24 h: the same config runs up to 2x slower on a busy node (CNN encoder 17-49 s/it, CNN + width 256
# 30 s/it at best, so up to ~17 h), and submitit can't requeue on Euler, so a killed run saves no auc.npy
run() {
    local rooms=$1 episode_length=$2 gamma=$3; shift 3
    python scripts/train.py --multirun \
        hydra/launcher=euler \
        hydra.launcher.submitit_folder="${STORAGE}/logs/%j" \
        hydra.launcher.timeout_min=1440 \
        env.room_path="${rooms}" \
        env.goal_radius="${GOAL_RADIUS}" \
        env.colour_mode=off \
        env.goal_mode=random \
        algorithm.max_steps="${episode_length}" \
        algorithm.lr="${LR:-${MLP_LR}}" \
        wandb.group="${GROUP_NAME}" \
        seed="$(join "${SEEDS[@]}")" \
        storage="${STORAGE}" \
        timestamp="${NOW}" \
        env.gamma="${gamma}" \
        observation="${ENCODER:-${BASE_ENCODER}}" \
        env.fov="${FOV:-${BASE_FOV}}" \
        env.collision_penalty="${COLLISION_PENALTY:-${BASE_COLLISION_PENALTY}}" \
        "$@" \
        || echo "Some training jobs failed, plotting the seeds that finished" >&2
}

# run on Stage 1's encoder and reward
stage1_run() {
    ENCODER=${STAGE1_ENCODER} COLLISION_PENALTY=${STAGE1_COLLISION_PENALTY} run "$@"
}

width_overrides() {
    local w=$1
    echo "observation.obs_embed_hidden_sizes=[${w},${w}] backbone.backbone_hidden_sizes=[${w},${w}] head.policy_hidden_sizes=[${w},${w}] head.value_hidden_sizes=[${w},${w}]"
}

depth_overrides() {
    local sizes=$1
    echo "observation.obs_embed_hidden_sizes=${sizes} backbone.backbone_hidden_sizes=${sizes} head.policy_hidden_sizes=${sizes} head.value_hidden_sizes=${sizes}"
}

run_theme() {
    local rooms=$1 episode_length=$2 gamma=$3 theme=$4
    case "${theme}" in
        baseline)
            run "${rooms}" "${episode_length}" "${gamma}"
            # Stage 1's baseline is always kept: it links the stages and is the MLP side of MLP vs LSTM
            stage1_run "${rooms}" "${episode_length}" "${gamma}" ;;
        observation)
            # Encoder x FOV without the baseline cell
            # shellcheck disable=SC2046
            ENCODER=mlp_observation FOV="$(join $(encoder_fovs mlp_observation "${MLP_FOVS[@]}"))" \
                run "${rooms}" "${episode_length}" "${gamma}"
            # shellcheck disable=SC2046
            ENCODER=cnn_observation FOV="$(join $(encoder_fovs cnn_observation "${CNN_FOVS[@]}"))" \
                run "${rooms}" "${episode_length}" "${gamma}" ;;
        action)
            run "${rooms}" "${episode_length}" "${gamma}" \
                env.action_type=discrete env.discrete_actions="$(join "${DISCRETE_ACTIONS[@]}")" ;;
        reward)
            # shellcheck disable=SC2046
            COLLISION_PENALTY="$(join $(without "${BASE_COLLISION_PENALTY}" "${COLLISION_PENALTIES[@]}"))" \
                run "${rooms}" "${episode_length}" "${gamma}"
            # The gamma levels replace the tier's default gamma
            # shellcheck disable=SC2046
            run "${rooms}" "${episode_length}" "$(join $(without "${gamma}" "${GAMMAS[@]}"))" ;;
        aux)
            # Loop closure needs a recurrent state, so this theme runs on the LSTM. Distances are masked
            # in every level so aux on/off see the same input
            LR=${LSTM_LR} stage1_run "${rooms}" "${episode_length}" "${gamma}" backbone=lstm_backbone \
                'observation.masked_channels=[3]' head.auxiliary_head.enabled=false
            for coeffs in "${AUX_LEVELS[@]}"; do
                read -r depth_coeff loop_coeff <<< "${coeffs}"
                LR=${LSTM_LR} stage1_run "${rooms}" "${episode_length}" "${gamma}" backbone=lstm_backbone \
                    'observation.masked_channels=[3]' head.auxiliary_head.enabled=true \
                    head.auxiliary_head.depth_coeff="${depth_coeff}" head.auxiliary_head.loop_closure_coeff="${loop_coeff}"
            done
            if [ "${MATCHED_REFERENCES}" = true ]; then
                # Loop closure doesn't need a hidden target, so it also runs on the unmasked LSTM, against
                # the unmasked LSTM without aux. That one runs at every BPTT window (the bptt theme)
                LR=${LSTM_LR} stage1_run "${rooms}" "${episode_length}" "${gamma}" backbone=lstm_backbone \
                    head.auxiliary_head.enabled=false backbone.bptt_window_size="$(join "${LSTM_BPTT_WINDOWS[@]}")"
                LR=${LSTM_LR} stage1_run "${rooms}" "${episode_length}" "${gamma}" backbone=lstm_backbone \
                    head.auxiliary_head.enabled=true head.auxiliary_head.depth_coeff=0.0 head.auxiliary_head.loop_closure_coeff=1.0
            fi ;;
        capacity)
            for w in $(without "${BASE_WIDTH}" "${WIDTHS[@]}"); do
                # shellcheck disable=SC2046
                run "${rooms}" "${episode_length}" "${gamma}" $(width_overrides "${w}")
            done
            for sizes in "${DEPTH_LEVELS[@]}"; do
                # shellcheck disable=SC2046
                run "${rooms}" "${episode_length}" "${gamma}" $(depth_overrides "${sizes}")
            done
            for w in "${AUX_WIDTHS[@]}"; do
                # Depth aux only: the MLP backbone has no state to predict a loop closure from. Stage 1 left
                # loop_closure_coeff at its default 1.0, so its aux points are depth + loop closure (aux:1.0-1.0)
                # shellcheck disable=SC2046
                stage1_run "${rooms}" "${episode_length}" "${gamma}" $(width_overrides "${w}") \
                    'observation.masked_channels=[3]' head.auxiliary_head.enabled=true \
                    head.auxiliary_head.depth_coeff=1.0 head.auxiliary_head.loop_closure_coeff=0.0
                if [ "${MATCHED_REFERENCES}" = true ]; then
                    # The same masked MLP without aux, its matched reference
                    # shellcheck disable=SC2046
                    stage1_run "${rooms}" "${episode_length}" "${gamma}" $(width_overrides "${w}") \
                        'observation.masked_channels=[3]' head.auxiliary_head.enabled=false
                fi
            done ;;
    esac
}

cd "${PROJECT_ROOT}"
for theme in "${THEMES[@]}"; do
    if [ ${#SHORT_ROOMS[@]} -gt 0 ]; then
        run_theme "$(join "${SHORT_ROOMS[@]}")" 512 0.99 "${theme}" &
    fi
    if [ ${#LONG_ROOMS[@]} -gt 0 ]; then
        run_theme "$(join "${LONG_ROOMS[@]}")" 4096 0.999 "${theme}" &
    fi
done
wait

python scripts/evaluate_metrics.py \
    wandb.group="${GROUP_NAME}" \
    storage="${STORAGE}" \
    timestamp="${NOW}"
