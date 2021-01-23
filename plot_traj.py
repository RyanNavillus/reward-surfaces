import pandas
import sys
import matplotlib.pyplot as plt
import numpy as np
import re
import argparse

def main():
    parser = argparse.ArgumentParser(description='plot csv data via matplotlib')
    parser.add_argument('csv_file', type=str)
    parser.add_argument('--key', type=str, default='episode_rewards')

    args = parser.parse_args()
    csv_file = args.csv_file
    df = pandas.read_csv(csv_file)
    idxs = np.argsort(df['checkpoint'])
    fig = plt.figure()
    ax = plt.axes()
    ax.plot(df['checkpoint'][idxs], df[args.key][idxs])
    out_fname = "vis/"+"".join([c for c in csv_file if re.match(r'\w', c)])
    plt.savefig(out_fname)

if __name__ == "__main__":
    main()
