#!/usr/bin/env bash
# Tiny end-to-end training run to check that a code change didn't break the pipeline.
# Not a learning test: 2 PPO iterations, few envs, W&B disabled, output in a temp dir.
#
# Usage: smoke_test.sh <python> [extra hydra overrides...]
#   smoke_test.sh ./venv/bin/python backbone=lstm_backbone
#   smoke_test.sh ./venv/bin/python observation=cnn_observation observation.frame_stack.enabled=true
set -euo pipefail

PYTHON=${1:?"usage: smoke_test.sh <python> [hydra overrides...]"}
shift

REPO_ROOT="$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)"
OUT_DIR="$(mktemp -d "${TMPDIR:-/tmp}/semproj_smoke.XXXXXX")"

export WANDB_MODE=disabled
export WANDB_SILENT=true
export SDL_VIDEODRIVER=dummy   # pygame surfaces for the rollout video, no display needed

cd "${REPO_ROOT}"
# num_envs % minibatch_size == 0 keeps RecurrentPPO happy; max_steps small so eval and metrics finish fast
"${PYTHON}" scripts/train.py \
    env.room_path="rooms/one_room.json" \
    env.num_envs=8 \
    env.num_eval_envs=4 \
    env.completion_rate_eps=2 \
    env.mean_of_means_runs=1 \
    algorithm.n_iterations=2 \
    algorithm.num_steps=16 \
    algorithm.max_steps=32 \
    algorithm.eval_interval=1 \
    algorithm.minibatch_size=4 \
    algorithm.mini_batch_size=32 \
    wandb.mode=disabled \
    storage="${OUT_DIR}" \
    hydra.run.dir="${OUT_DIR}/hydra" \
    "$@"

echo "Smoke test passed, outputs in ${OUT_DIR}"
