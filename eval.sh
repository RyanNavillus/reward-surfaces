#!/usr/bin/env bash
env_name=$1
checkpoint_name=$2
echo $env_name
echo $checkpoint_name

export PYTHONPATH=/home/ryan_sullivan/reward-surfaces
python scripts/generate_eval_jobs.py --num-steps=1000000 --checkpoint="best" "./runs/${checkpoint_name}_checkpoints" "./runs/${checkpoint_name}_eig_vecs/"
python scripts/run_jobs_multiproc.py "./runs/${checkpoint_name}_eig_vecs/jobs.sh"
