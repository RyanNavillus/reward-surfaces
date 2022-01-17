import argparse
import json
import torch
import numpy as np
from pathlib import Path
from reward_surfaces.experiments import generate_plane_data
from reward_surfaces.utils.surface_utils import readz, filter_normalize


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='generate jobs for heatmap')
    parser.add_argument('checkpoint', type=str, help="Path to checkpoint")
    parser.add_argument('grad_path', type=str, help="Path to gradient direction .npz file")
    parser.add_argument('output_path', type=str, help="Path to output folder")
    parser.add_argument('--grad-magnitude', type=float, default=0.000000003,
                        help="Magnitude of the step size in the gradient direction")
    parser.add_argument('--rand-magnitude', type=float, default=0.3,
                        help="Magnitude of the step size in the random direction")
    parser.add_argument('--num-episodes', type=int, default=1000000, help="Number of episodes to evaluate each point")
    parser.add_argument('--num-steps', type=int, default=200000, help="Number of steps to evaluate each point")
    parser.add_argument('--grid-size', type=int, default=31, help="Width and height of the heatmap grid")
    args = parser.parse_args()

    trained_checkpoint = Path(args.checkpoint)
    output_path = Path(args.output_path)

    # Load gradient direction
    grad_dir_fname = Path(args.grad_path) / "grad.npz"
    grad_dir = readz(grad_dir_fname)

    # Load random filter normalized direction
    rand_dir_fname = trained_checkpoint / "parameters.th"
    param_values = torch.load(rand_dir_fname, map_location=torch.device('cpu')).values()
    rand_dir = [filter_normalize(v.cpu().detach().numpy()) for v in param_values]

    train_info = json.load(open(trained_checkpoint.parent / "info.json"))
    # TODO: Make option for grad normalization
    flat_grad = [d.flatten() for d in grad_dir]
    grad_cat = np.concatenate(flat_grad, axis=0)
    grad_magnitude = np.linalg.norm(grad_cat)
    dir1 = [d / grad_magnitude for d in grad_dir]
    dir2 = [d * args.rand_magnitude for d in rand_dir]

    train_info['dir1_mag'] = args.grad_magnitude
    train_info['dir2_mag'] = args.rand_magnitude

    generate_plane_data(trained_checkpoint, output_path, dir1, dir2, 1.0, train_info,
                        dir1_scale=args.grad_magnitude,
                        dir2_scale=args.rand_magnitude,
                        grid_size=args.grid_size,
                        num_episodes=args.num_episodes,
                        num_steps=args.num_steps)
