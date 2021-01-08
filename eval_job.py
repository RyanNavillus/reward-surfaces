import torch
import argparse
from agents.make_agent import make_agent
import torch
import json
import os
import shutil
import numpy as np


def main():
    parser = argparse.ArgumentParser(description='generate jobs for plane')
    parser.add_argument('job_dir', type=str)
    parser.add_argument('--offset1', type=float, help="if specified, looks for dir1.npz for parameter offset and multiplies it by offset, adds to parameter for evaluation")
    parser.add_argument('--offset2', type=float, help="if specified, looks for dir2.npz for parameter offset and multiplies it by offset, adds to parameter for evaluation")
    #parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--use_offset_critic', action='store_true')

    args = parser.parse_args()

    torch.set_num_threads(1)

    base_source_path = args.job_dir
    checkpoint_fname = next(fname for fname in os.listdir(args.job_dir) if "checkpoint" in fname)
    checkpoint_path = os.path.join(args.job_dir, checkpoint_fname)
    info_fname = "info.json"
    params_fname = "parameters.th"

    info = json.load(open(os.path.join(args.job_dir, info_fname)))

    agent = make_agent(info['agent_name'], info['env'], info['eval_device'], info['hyperparameters'])
    agent.load_weights(checkpoint_path)

    eval_agent = None
    if args.use_offset_critic:
        eval_agent = make_agent(info['agent_name'], info['env'], info['eval_device'], info['hyperparameters'])
        eval_agent.load_weights(checkpoint_path)

    agent_weights = agent.get_weights()
    if args.offset1 is not None:
        offset1_data = np.load(os.path.join(args.job_dir, "dir1.npz"))
        agent_weights += [off * args.offset1 / (info['grid_size']//2) for off in offset1_data.values()]
    if args.offset2 is not None:
        offset2_data = np.load(os.path.join(args.job_dir, "dir2.npz"))
        agent_weights += [off * args.offset2 / (info['grid_size']//2) for off in offset2_data.values()]

    agent.set_weights(agent_weights)

    results = agent.evaluate(info['num_episodes'], info['num_steps'], eval_trainer=eval_agent)

    json.dump(results, open(os.path.join(args.job_dir,'results',f"{args.offset1:03},{args.offset2:03}.json"),'w'))



if __name__ == "__main__":
    main()
