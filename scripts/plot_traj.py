import pandas
import sys
import matplotlib.pyplot as plt
import numpy as np
import re
import argparse
from reward_surfaces.plotting import plot_traj


def main():
    parser = argparse.ArgumentParser(description='plot csv data via matplotlib')
    parser.add_argument('csv_file', type=str)
    parser.add_argument('--key', type=str, default='episode_rewards')

    args = parser.parse_args()

    plot_traj(args.csv_file, key=args.key)

if __name__ == "__main__":
    main()
