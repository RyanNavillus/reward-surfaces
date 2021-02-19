import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
import argparse
from agents.make_agent import make_agent
import torch
torch.set_num_threads(1)
import json
import shutil
from pathlib import Path
import numpy as np
from reward_surfaces.utils.compute_results import save_results

def main():
    parser = argparse.ArgumentParser(description='generate jobs for plane')
    parser.add_argument('train_dir', type=str)
    parser.add_argument('checkpoint', type=str)
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

    job_name = checkpoint
    cur_results = {
        "checkpoint": checkpoint,
    }
    save_results(agent, info, out_dir, cur_results, job_name)


if __name__ == "__main__":
    main()
