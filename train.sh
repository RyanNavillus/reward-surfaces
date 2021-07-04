#!/usr/bin/env bash
env_name=$1
checkpoint_name=$2
echo $env_name
echo $checkpoint_name

export PYTHONPATH=/home/ryan_sullivan/reward-surfaces
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/ryan/.mujoco/mujoco200/bin
python3 scripts/train_agent.py "./runs/${checkpoint_name}_checkpoints" 10000000 SB3_ON "$env_name" cuda '{"ALGO": "PPO",
"num_envs": 8}' --save_freq=1000000
