#!/usr/bin/env bash

python scripts/train.py --multirun \
    env.room_path="rooms/one_room.json" \
    observation=mlp_observation,cnn_observation \
    backbone=mlp_backbone,lstm_backbone \
    env.range_type=diagonal,horizontal,inf

