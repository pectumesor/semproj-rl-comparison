#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

GROUP_NAME="two_room_run"
# One timestamp for the whole sweep, so every job writes its seed under the same folder
NOW=$(date +"%Y%m%d_%H%M%S")
STORAGE=${PROJECT_ROOT}/logs/${GROUP_NAME}

cd "${PROJECT_ROOT}"
# A failed job must not stop the seeds that did finish from being plotted below
python scripts/train.py --multirun \
    env.room_path="rooms/two_room.json" \
    env.goal_radius=16 \
    env.goal_pos.x=40 \
    env.goal_pos.y=40 \
    observation=mlp_observation,cnn_observation \
    backbone=lstm_backbone,mlp_backbone \
    wandb.group="${GROUP_NAME}" \
    storage="${STORAGE}" \
    timestamp="${NOW}" \
    || echo "Some training jobs failed, plotting the seeds that finished" >&2

# Mean-of-means plot per room and architecture, uploaded to W&B
python scripts/evaluate_metrics.py \
    wandb.group="${GROUP_NAME}" \
    storage="${STORAGE}" \
    timestamp="${NOW}"
