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
    parser.add_argument('checkpoint', type=int)
    parser.add_argument('out_dir', type=str)
    parser.add_argument('--device', type=str, default='cuda')

    args = parser.parse_args()

    train_dir = Path(args.train_dir)
    out_dir = Path(args.out_dir)

    info_fname = "info.json"
    info = json.load(open((out_dir / info_fname)))

    device = args.device
    agent = make_agent(info['agent_name'], info['env'], device, info['hyperparameters'])

    checkpoint = args.checkpoint
    path = train_dir / str(checkpoint)
    checkpoint_fname = next(fname for fname in os.listdir(path) if "checkpoint" in fname)
    checkpoint_path = path / checkpoint_fname


    agent.load_weights(checkpoint_path)
    print("evaluating ",checkpoint)

    if info['est_hesh']:
        print(f"estimating hesh with {info['eval_num_steps']} steps")
        assert info['eval_num_episodes'] > 100000000, "hesh calculation only takes in steps, not episodes"
        results = agent.calculate_eigenvalues(info['eval_num_steps'])
    else:
        results = agent.evaluate(info['eval_num_steps'], info['eval_num_episodes'])
    results['checkpoint'] = checkpoint

    json.dump(results,open(out_dir/f"results/{checkpoint}.json",'w'))

if __name__ == "__main__":
    main()
