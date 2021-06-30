#!/usr/bin/env bash
env_name=$1
checkpoint_name=$2
echo $env_name
echo $checkpoint_name

export PYTHONPATH=/home/ryan_sullivan/reward-surfaces
python scripts/train_agent.py "./runs/${checkpoint_name}_checkpoints" 1000000 SB3_ON "$env_name" cuda '{"ALGO": "PPO", "num_envs": 1, "n_epochs": 1, "gamma": 0.99, "gae_lambda": 0.95, "clip_range": 0.2, "ent_coef": 0.0, "vf_coef": 0.5, "max_grad_norm": 0.5, "learning_rate": 0.0003, "batch_size": 64}' --save_freq=10000
