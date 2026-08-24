#!/usr/bin/env bash
ETH_USERNAME=eazevedo
PROJECT_NAME=semproj
SCRATCH=/cluster/scratch/${ETH_USERNAME}/${PROJECT_NAME}
CONDA_ENVIRONMENT=semproj

module load stack/2025-06
eval "$(conda shell.bash hook)"

if conda env list | grep -qE "^${CONDA_ENVIRONMENT}\s"; then
    conda activate ${CONDA_ENVIRONMENT}
else
    echo "Conda environment '${CONDA_ENVIRONMENT}' not found, creating from environment.yaml"
    conda env create -f environment.yaml
    conda activate ${CONDA_ENVIRONMENT}
fi

python scripts/train.py --multirun \
    hydra/launcher=euler \
    hydra.launcher.submitit_folder="${SCRATCH}/.submitit/%j" \
    env.room_path="rooms/four_room.json" \
    env.goal_radius=16 \
    env.goal_pos.x=40 \
    env.goal_pos.y=40 \
    wandb.group=euler_test \
    observation=mlp_observation,cnn_observation \
    backbone=lstm_backbone,mlp_backbone \
    seed=0,1,2,3,4

