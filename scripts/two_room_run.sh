#!/usr/bin/env bash

python scripts/train.py --multirun \
    env.room_path="rooms/two_room.json" \
    env.goal_radius=16 \
    env.goal_pos.x=40 \
    env.goal_pos.y=40 \
    observation=mlp_observation,cnn_observation \
    backbone=lstm_backbone,mlp_backbone \

