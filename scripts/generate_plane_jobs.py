import argparse
from reward_surfaces.agents.make_agent import make_agent
import torch
import json
import os
import shutil
from pathlib import Path
import numpy as np
from reward_surfaces.utils.surface_utils import readz, filter_normalized_params
from reward_surfaces.utils.path_utils import strip_lagging_slash
from reward_surfaces.experiments import generate_plane_data

def find_unscaled_alts(agent, method):
    if method == "filter":
        return filter_normalized_params(agent)
    elif method == "fischer":
        raise NotImplemenetedError()
    else:
        raise ValueError(f"bad directions argument {method}, must be 'filter' or `fischer`")


def main():
    parser = argparse.ArgumentParser(description='generate jobs for plane')
    parser.add_argument('checkpoint_dir', type=str)
    parser.add_argument('output_path', type=str)
    parser.add_argument('--directions', type=str, default="filter", help="'filter' is only option right now")
    parser.add_argument('--copy-directions', type=str, help="overrides directions with directions from specified folder. Does not copy any other data. ")
    parser.add_argument('--scale-vec', type=str, help="A .npz file of same shape as directions, which indicates how much each dimention should be scaled by.")
    parser.add_argument('--dir1', type=str, help="overrides dir1 with vector from specified path.")
    parser.add_argument('--dir2', type=str, help="overrides dir2 with vector from specified path.")
    parser.add_argument('--magnitude', type=float, default=1., help="scales directions by given amount")
    parser.add_argument('--grid-size', type=int, default=5)
    parser.add_argument('--num-steps', type=int)
    parser.add_argument('--num-episodes', type=int)
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--use_offset_critic', action='store_true', help="use critic at center or at offset for value estimation")
    parser.add_argument('--est-hesh', action='store_true')
    parser.add_argument('--calc-hesh', action='store_true')
    parser.add_argument('--calc-grad', action='store_true')

    args = parser.parse_args()

    assert args.copy_directions is None or args.directions == "copy", "if --copy-directions is None, --directions=copy must be set"
    output_path = Path(args.output_path)
    checkpoint_dir = Path(args.checkpoint_dir)
    folder_argname = Path(os.path.dirname(strip_lagging_slash(args.checkpoint_dir)))
    checkpoint_fname = next(fname for fname in os.listdir(checkpoint_dir) if "checkpoint" in fname)
    checkpoint_path = checkpoint_dir / checkpoint_fname

    info_fname = "info.json"
    info = json.load(open((folder_argname / info_fname)))

    device = "cpu"
    agent = make_agent(info['agent_name'], info['env'], device, info['hyperparameters'])
    agent.load_weights(checkpoint_path)

    # generate directions normally
    dir1_vec, dir2_vec = find_unscaled_alts(agent, args.directions)

    # copy directions
    if args.copy_directions is not None:
        dir_path = Path(args.copy_directions)
        dir1_vec = readz(dir_path / "dir1.npz")
        dir2_vec = readz(dir_path / "dir2.npz")
    if args.dir1 is not None:
        dir1_vec = readz(args.dir1)
        info['dir1'] = args.dir1
    if args.dir2 is not None:
        dir2_vec = readz(args.dir2)
        info['dir2'] = args.dir2

    if args.scale_vec is not None:
        scale_vec = readz(args.scale_vec)
        dir1_vec = scale_dir(dir1_vec, scale_vec)
        dir2_vec = scale_dir(dir2_vec, scale_vec)

    if args.magnitude is not None:
        info['magnitude'] = m = args.magnitude
        dir1_vec = [m*v for v in dir1_vec]
        dir2_vec = [m*v for v in dir2_vec]

    info['directions'] = args.directions if args.copy_directions is None else "copy"


    generate_plane_data(args.checkpoint_dir, args.output_path, dir1_vec, dir2_vec, info,
        grid_size=args.grid_size,
        num_steps=args.num_steps,
        num_episodes=args.num_episodes,
        device=args.device,
        use_offset_critic=args.use_offset_critic,
        est_hesh=args.est_hesh,
        calc_hesh=args.calc_hesh,
        calc_grad=args.calc_grad,
    )


if __name__ == "__main__":
    main()
