#!/usr/bin/env bash
set -euo pipefail

# Under sbatch the script runs from a copy in /var/spool/slurm, so BASH_SOURCE
# points there; resolve the repo from the submission directory instead.
if [[ -n "${SLURM_JOB_ID:-}" ]]; then
    PROJECT_ROOT="$(git -C "${SLURM_SUBMIT_DIR}" rev-parse --show-toplevel)"
else
    PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
fi

GROUP_NAME="four_room_test"
# One timestamp for the whole sweep, so every job writes its seed under the same folder
NOW=$(date +"%Y%m%d_%H%M%S")
STORAGE=${PROJECT_ROOT}/logs/${GROUP_NAME}

cd "${PROJECT_ROOT}"
# A failed job must not stop the seeds that did finish from being plotted below
python scripts/train.py --multirun \
    env.room_path="rooms/four_room.json" \
    seed=0,1,2,3,4 \
    wandb.group="${GROUP_NAME}" \
    storage="${STORAGE}" \
    timestamp="${NOW}" \
    || echo "Some training jobs failed, plotting the seeds that finished" >&2

# Mean-of-means plot per room and architecture, uploaded to W&B
python scripts/evaluate_metrics.py \
    wandb.group="${GROUP_NAME}" \
    storage="${STORAGE}" \
    timestamp="${NOW}"
