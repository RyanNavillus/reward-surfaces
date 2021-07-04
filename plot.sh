#!/usr/bin/env bash
env_name=$1
checkpoint_name=$2
echo $env_name
echo $checkpoint_name

export PYTHONPATH=/home/ryan_sullivan/reward-surfaces
python scripts/generate_plane_jobs.py --grid-size=31 --magnitude=1.0 --num-steps=200000 "./runs/${checkpoint_name}_checkpoints/best" "./runs/${checkpoint_name}_eig_vecs_plane/"
python scripts/run_jobs_multiproc.py "./runs/${checkpoint_name}_eig_vecs_plane/jobs.sh"
python scripts/job_results_to_csv.py "./runs/${checkpoint_name}_eig_vecs_plane/"
python scripts/plot_plane.py "./runs/${checkpoint_name}_eig_vecs_plane/results.csv" --outname="./runs/${checkpoint_name}_curvature_plot.png"
