import argparse
from reward_surfaces.plotting import plot_plane
import os
from pathlib import Path
from reward_surfaces.utils.job_results_to_csv import job_results_to_csv
import shutil
import numpy as np
import pandas

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='generate jobs for plane')
    parser.add_argument('gif_data_folder', type=str)
    parser.add_argument('--outname', type=str, help="if specified, outputs file with this name (extension added onto name)")
    parser.add_argument('--key', type=str, default="episode_rewards", help="key in csv file to plot")
    parser.add_argument('--type', type=str, default="mesh", help="plot type. Possible types are: [all, mesh, vtp, heat, contour, contourf]")

    args = parser.parse_args()

    frames_dir = "_frames/"
    if os.path.exists(frames_dir):
        shutil.rmtree(frames_dir)
    os.mkdir(frames_dir)

    gif_data_folder = Path(args.gif_data_folder)

    checkpoints = []
    for checkpoint in sorted(os.listdir(gif_data_folder)):
        check_path = gif_data_folder / checkpoint
        if os.path.isdir(check_path) and len(os.listdir(check_path/"results/")) > 0:
            checkpoints.append(checkpoint)

    for checkpoint in checkpoints:
        job_results_to_csv(gif_data_folder / checkpoint)

    base_mag = None
    factor = 2
    frame_idx = 0
    for checkpoint in checkpoints:
        csv_path = gif_data_folder / checkpoint / "results.csv"
        m = np.max(pandas.read_csv(csv_path)[args.key])
        if base_mag is None:
            base_mag = m
        else:
            while m > base_mag * factor:
                base_mag = base_mag * factor

            fname = plot_plane(str(csv_path), str(frames_dir+checkpoint), args.key, args.type, vmin=0, vmax=base_mag, show=False)
            if not fname:
                continue
            os.rename(fname, f"{frames_dir}{frame_idx:05}.png")
            frame_idx += 1
