import argparse
import json
import os
import torch
from reward_surfaces.utils.surface_utils import readz
from pathlib import Path
from reward_surfaces.algorithms import search
from reward_surfaces.runners.run_jobs_multiproc import run_job_list_list
from reward_surfaces.utils.job_results_to_csv import job_results_to_csv


bigint = 1000000000000
def main():
    parser = argparse.ArgumentParser(description='run a particular evaluation job')
    parser.add_argument('train_dir', type=str)
    parser.add_argument('grads_dir', type=str, help="direction to search for threshold")
    parser.add_argument('output_dir', type=str)
    parser.add_argument('--num-steps', type=int, default=bigint)
    parser.add_argument('--num-episodes', type=int, default=bigint)
    parser.add_argument('--device', type=str, default="cpu")
    parser.add_argument('--key', type=str, default="episode_rewards")
    parser.add_argument('--tolerance', type=float, default=0.05)

    args = parser.parse_args()

    out_path = Path(args.output_dir)
    train_path = Path(args.train_dir)
    grad_path = Path(args.grads_dir)
    os.mkdir(out_path)
    os.mkdir(out_path/"results")

    checkpoints = sorted([checkpoint for checkpoint in os.listdir(train_path) if os.path.isdir(train_path/checkpoint)])
    commands = []
    for checkpoint in checkpoints:
        params = train_path/checkpoint/"parameters.th"
        grad = grad_path/checkpoint/"grad.npz"
        out_fname = str(out_path/"results"/checkpoint)+".json"
        command = f"python3 -m reward_surfaces.bin.search_equiv {params} {grad} {out_fname} --num-steps={args.num_steps} --num-episodes={args.num_episodes} --device {args.device} --key {args.key} --tolerance {args.tolerance}"
        commands.append(command)

    info = json.load(open(train_path/"info.json"))
    info['search_steps'] = args.num_steps
    info['search_episodes'] = args.num_episodes
    info['search_key'] = args.key
    info['search_tolerance'] = args.tolerance

    json.dump(info, open(out_path/"info.json",'w'))

    with open(out_path/"jobs.sh", 'w') as file:
        file.write("\n".join(commands))
    # run_job_list_list(commands)
    # job_results_to_csv(out_path)

if __name__ == "__main__":
    main()
