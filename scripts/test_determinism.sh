#!/usr/bin/env bash

#Script to train the reproduceability of agents training by fixing the start position, end position and seed.
#Run with env.rand_pos="False" flag to toggle fixed initial position

python tests/test_determinism.py \
    wandb.group=test_determinism \
    env.rand_pos="False" \
    algorithm.n_iterations=50

