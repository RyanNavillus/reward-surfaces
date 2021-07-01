#!/usr/bin/env bash
env_name=$1
checkpoint_name=$2
echo $env_name
echo $checkpoint_name

export PYTHONPATH=/home/ryan_sullivan/reward-surfaces
python scripts/train_agent.py "./runs/${checkpoint_name}_checkpoints" 10000000 SB3_ON "$env_name" cpu '{"ALGO": "PPO", "num_envs": 8, "n_epochs": 4, "clip_range": "lin_0.1", "ent_coef": 0.001, "vf_coef": 0.5, "learning_rate": "lin_0.00025", "batch_size": 256}' --save_freq=10000
