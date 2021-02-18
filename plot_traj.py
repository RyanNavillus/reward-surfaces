import pandas
import sys
import matplotlib.pyplot as plt
import numpy as np
import re
import argparse

def plot_traj(csv_file, key="episode_rewards"):
    df = pandas.read_csv(csv_file)
    idxs = np.argsort(df['checkpoint'])
    fig = plt.figure()
    ax = plt.axes()
    ax.plot(df['checkpoint'][idxs], df[args.key][idxs])
    out_fname = "vis/"+"".join([c for c in csv_file if re.match(r'\w', c)]) + args.key
    plt.savefig(out_fname)


def main():
    parser = argparse.ArgumentParser(description='plot csv data via matplotlib')
    parser.add_argument('csv_file', type=str)
    parser.add_argument('--key', type=str, default='episode_rewards')

    args = parser.parse_args()

    plot_traj(args.csv_file, key=args.key)

if __name__ == "__main__":
    main()
