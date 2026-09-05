#!/usr/bin/env bash

python tests/test_metrics.py --multirun \
    env.room_path="rooms/four_room.json" \
    env.rand_pos="False" \
    wandb.group=metrics_test \
    seed=0,1,2,3,4

