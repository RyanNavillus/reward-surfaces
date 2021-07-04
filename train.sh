#!/usr/bin/env bash
env_name=$1
checkpoint_name=$2
echo $env_name
echo $checkpoint_name

export PYTHONPATH=/home/ryansullivan/Documents/SwarmLabs/reward-surfaces
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/ryan/.mujoco/mujoco200/bin
python3 scripts/train_agent.py "./runs/${checkpoint_name}_checkpoints" SB3_ON "$env_name" cuda '{"ALGO": "PPO", "num_envs": 8, "n_timesteps": 755}' --save_freq=10000
