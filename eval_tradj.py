import argparse
from agents.make_agent import make_agent
import torch
import json
import os
import shutil
import pandas
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
    parser.add_argument('--num-steps', type=int)
    parser.add_argument('--num-episodes', type=int)
    parser.add_argument('--device', type=str, default='cuda')

    args = parser.parse_args()

    assert args.num_steps is not None or args.num_episodes is not None, "one of num_step or num_episodes must be specified"
    if args.num_steps is None:
        args.num_steps = 10000000000000
    if args.num_episodes is None:
        args.num_episodes = 10000000000000

    train_dir = Path(args.train_dir)

    checkpoints = [dir for dir in os.listdir(train_dir) if os.path.isdir(train_dir / dir)]
    checkpoint_paths = [train_dir / dir for dir in checkpoints]
    checkpoint_fnames = [checkpoint_dir / next(fname for fname in os.listdir(checkpoint_dir) if "checkpoint" in fname) for checkpoint_dir in checkpoint_paths]
    print(checkpoint_fnames)

    info_fname = "info.json"
    info = json.load(open((train_dir / info_fname)))

    device = args.device
    agent = make_agent(info['agent_name'], info['env'], device, info['hyperparameters'])
    info['eval_num_episodes'] = args.num_episodes
    info['eval_num_steps'] = args.num_steps

    json.dump(info, open((train_dir / info_fname),'w'), indent=4)

    all_results = []
    for checkpoint, path in zip(checkpoints, checkpoint_fnames):
        try:
            agent.load_weights(path)
            print("evaluating ",checkpoint)
            results = agent.evaluate(args.num_episodes, args.num_steps)
            results['checkpoint'] = checkpoint
            all_results.append(results)
        except Exception:
            print("failed to evaluate checkpoint",checkpoint)
        

    df = pandas.read_json(json.dumps(all_results))
    df.to_csv(train_dir/"eval_tradj.csv",index=False)

if __name__ == "__main__":
    main()
