import argparse
from agents.make_agent import make_agent
import torch
import json
import os
import shutil
from pathlib import Path
import numpy as np

def strip_lagging_slash(f):
    if f[-1] == '/':
        return f[:-1]
    else:
        return f

def main():
    parser = argparse.ArgumentParser(description='generate jobs for plane')
    parser.add_argument('train_dir', type=str)
    parser.add_argument('out_dir', type=str)
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--est-hesh', action='store_true')
    parser.add_argument('--calc-hesh', action='store_true')
    parser.add_argument('--num-steps', type=int)
    parser.add_argument('--num-episodes', type=int)

    args = parser.parse_args()

    assert not (args.est_hesh and args.calc_hesh), "calculating and estimating hessian cannot happen at the same time"
    assert args.num_steps is not None or args.num_episodes is not None, "one of num_step or num_episodes must be specified"
    if args.num_steps is None:
        args.num_steps = 10000000000000
    if args.num_episodes is None:
        args.num_episodes = 10000000000000

    train_dir = Path(args.train_dir)
    out_dir = Path(args.out_dir)
    os.makedirs(out_dir / "results")

    info_fname = "info.json"
    info = json.load(open((train_dir / info_fname)))

    device = args.device
    info['eval_num_episodes'] = args.num_episodes
    info['eval_num_steps'] = args.num_steps
    info['est_hesh'] = args.est_hesh
    info['calc_hesh'] = args.calc_hesh

    json.dump(info, open((out_dir / info_fname),'w'), indent=4)

    checkpoints = [dir for dir in os.listdir(train_dir) if os.path.isdir(train_dir / dir) and dir.isdigit()]
    all_jobs = []
    for checkpoint in checkpoints:
        job = f"python eval_tradj.py {args.train_dir} {checkpoint} {out_dir} --device={args.device}"
        all_jobs.append(job)

    jobs = "\n".join(all_jobs)+"\n"
    open((out_dir / "jobs.sh"),'w').write(jobs)

if __name__ == "__main__":
    main()
