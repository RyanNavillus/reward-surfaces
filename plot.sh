#!/usr/bin/env bash
env_name=$1
checkpoint_name=$2
echo $env_name
echo $checkpoint_name

export PYTHONPATH=/home/ryan_sullivan/reward-surfaces
python scripts/train_agent.py "./runs/${checkpoint_name}_checkpoints" 1000000 SB3_ON "$env_name" cuda '{"ALGO": "PPO", "num_envs": 1, "n_epochs": 1, "gamma": 0.99, "gae_lambda": 0.95, "clip_range": 0.2, "ent_coef": 0.0, "vf_coef": 0.5, "max_grad_norm": 0.5, "learning_rate": 0.0003, "batch_size": 64}' --save_freq=10000
python scripts/generate_eval_jobs.py --num-steps=1000000 --checkpoint=1000000 "./runs/${checkpoint_name}_checkpoints" "./runs/${checkpoint_name}_eig_vecs/"
python scripts/run_jobs_multiproc.py --num-cpus 1 "./runs/${checkpoint_name}_eig_vecs/jobs.sh"
python scripts/generate_plane_jobs.py  --grid-size=31 --magnitude=1.0 --num-steps=200000 "./runs/${checkpoint_name}_checkpoints/1000000" "./runs/${checkpoint_name}_eig_vecs_plane/"
python scripts/run_jobs_multiproc.py --num-cpus 2 "./runs/${checkpoint_name}_eig_vecs_plane/jobs.sh"
python scripts/job_results_to_csv.py "./runs/${checkpoint_name}_eig_vecs_plane/"
python scripts/plot_plane.py "./runs/${checkpoint_name}_eig_vecs_plane/results.csv" --outname="${checkpoint_name}_curvature_plot.png"
