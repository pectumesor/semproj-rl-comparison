#!/usr/bin/env bash

python scripts/train.py --multirun \
    env.room_path="rooms/four_room.json" \
    wandb.group=four_room_test \
    seed=0,1,2,3,4

