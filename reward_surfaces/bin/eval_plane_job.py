import os
import torch
import argparse
import json
import numpy as np
import pathlib
from reward_surfaces.agents import make_agent
from reward_surfaces.utils.compute_results import save_results
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
torch.set_num_threads(1)


def main():
    parser = argparse.ArgumentParser(description='generate jobs for plane')
    parser.add_argument('job_dir', type=str)
    parser.add_argument('--offset1', type=float, help="if specified, looks for dir1.npz for parameter offset and multiplies it by offset, adds to parameter for evaluation")
    parser.add_argument('--offset2', type=float, help="if specified, looks for dir2.npz for parameter offset and multiplies it by offset, adds to parameter for evaluation")
    #parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--use_offset_critic', action='store_true')

    args = parser.parse_args()

    torch.set_num_threads(1)

    base_source_path = pathlib.Path(args.job_dir)
    checkpoint_fname = next(fname for fname in os.listdir(args.job_dir) if "checkpoint" in fname)
    checkpoint_path = base_source_path / checkpoint_fname
    info_fname = "info.json"
    params_fname = "parameters.th"

    info = json.load(open(base_source_path / info_fname))

    agent = make_agent(info['agent_name'], info['env'], info['device'], info['hyperparameters'])
    agent.load_weights(checkpoint_path)

    eval_agent = None
    if args.use_offset_critic:
        eval_agent = make_agent(info['agent_name'], info['env'], info['device'], info['hyperparameters'])
        eval_agent.load_weights(checkpoint_path)

    agent_weights = agent.get_weights()
    if args.offset1 is not None:
        offset1_data = np.load(base_source_path / "dir1.npz")
        offset1_scalar = args.offset1 / (info['grid_size']//2)
        for a_weight, off in zip(agent_weights, offset1_data.values()):
            a_weight += off * offset1_scalar
    if args.offset2 is not None:
        offset2_data = np.load(base_source_path / "dir2.npz")
        offset2_scalar = args.offset2 / (info['grid_size']//2)
        # agent_weights += [off * args.offset2 / (info['grid_size']//2) for off in offset2_data.values()]
        for a_weight, off in zip(agent_weights, offset2_data.values()):
            a_weight += off * offset2_scalar

    agent.set_weights(agent_weights)

    job_name = f"{args.offset1:03},{args.offset2:03}"
    cur_results = {
        "dim0": offset1_scalar,
        "dim1": offset2_scalar,
    }
    save_results(agent, info, base_source_path, cur_results, job_name)


if __name__ == "__main__":
    main()
