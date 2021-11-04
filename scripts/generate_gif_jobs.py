import torch
import numpy as np
from reward_surfaces.experiments import generate_plane_data
import json
import subprocess
import argparse
from reward_surfaces.utils.surface_utils import filter_normalize
from pathlib import Path
import random
import os

def main():
    parser = argparse.ArgumentParser(description='generate jobs for plane')
    parser.add_argument('train_dir', type=str)
    parser.add_argument('out_dir', type=str)
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--num-steps', type=int)
    parser.add_argument('--num-episodes', type=int)
    parser.add_argument('--seed', type=int)
    parser.add_argument('--magnitude', type=float, default=0.2, help="scales directions by given amount")
    parser.add_argument('--grid-size', type=int, default=5)

    args = parser.parse_args()

    generate_gif_jobs(
        args.train_dir,
        args.out_dir,
        num_steps=args.num_steps,
        num_episodes=args.num_episodes,
        device=args.device,
        magnitude=args.magnitude,
        grid_size=args.grid_size,
        seed=args.seed
    )


def generate_gif_jobs(train_dir, out_dir, num_steps=None, num_episodes=None, device="cpu", seed=None, grid_size=5, magnitude=0.2):
    train_dir = Path(train_dir)
    out_dir = Path(out_dir)

    if seed is None:
        seed = random.randint(0,2**31)

    os.mkdir(out_dir)

    train_info = json.load(open(train_dir / "info.json"))
    train_info['magnitude'] = magnitude
    train_info['grid_size'] = grid_size
    train_info['num_steps'] = num_steps
    train_info['num_episodes'] = num_episodes
    train_info['seed'] = seed
    open(out_dir/"info.json",'w').write(json.dumps(train_info))

    all_gen_dirs = []
    for p_str in sorted(os.listdir(train_dir)):
        checkpoint_param_fname = train_dir / p_str / "parameters.th"
        if os.path.exists(checkpoint_param_fname):
            trained_checkpoint = train_dir / p_str
            generated_dir = out_dir / p_str

            train_info = json.load(open(train_dir / "info.json"))

            # reseeds so that random generated directions are the same every time
            np.random.seed(seed)

            checkpoint_params = [v.cpu().detach().numpy() for v in torch.load(checkpoint_param_fname,map_location=torch.device('cpu')).values()]
            dir1 = [filter_normalize(v)*magnitude for v in checkpoint_params]
            dir2 = [filter_normalize(v)*magnitude for v in checkpoint_params]

            generate_plane_data(trained_checkpoint, generated_dir, dir1, dir2, train_info, num_steps=num_steps, num_episodes=num_episodes, device=device,grid_size=grid_size)

            all_gen_dirs.append(generated_dir)

    # concatenate all jobs
    all_job_paths = "".join([open(dir/"jobs.sh").read() for dir in all_gen_dirs])
    multiproc_job_paths = "\n".join([f"python3 scripts/run_jobs_multiproc.py {dir/'jobs.sh'}" for dir in all_gen_dirs])
    open(out_dir/"all_jobs.sh",'w').write(all_job_paths)
    open(out_dir/"multiproc_jobs.sh",'w').write(multiproc_job_paths)


if __name__ == "__main__":
    main()
