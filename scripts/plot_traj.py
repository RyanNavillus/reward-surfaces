import pandas
import sys
import matplotlib.pyplot as plt
import numpy as np
import re
import os
import argparse
from reward_surfaces.plotting import plot_traj
from reward_surfaces.utils import job_results_to_csv


def main():
    parser = argparse.ArgumentParser(description='plot csv data via matplotlib')
    parser.add_argument('csv_file', type=str)
    parser.add_argument('--key', type=str, default='episode_rewards')
    parser.add_argument('--log-plot', action="store_true")

    args = parser.parse_args()

    if os.path.isdir(args.csv_file):
        job_results_to_csv(args.csv_file)
        csv_path = os.path.join(args.csv_file,"results.csv")
    else:
        csv_path = args.csv_file

    plot_traj(csv_path, key=args.key, log_plot=args.log_plot)


if __name__ == "__main__":
    main()
