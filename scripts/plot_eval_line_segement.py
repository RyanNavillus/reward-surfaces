import argparse
import pandas
import os
import math
import matplotlib.pyplot as plt
import re
import numpy as np
from reward_surfaces.utils import job_results_to_csv


def plot_grad_search(csv_fname, outname=None, key_name="episode_rewards", logscale=False):
    if os.path.isdir(csv_fname):
        job_results_to_csv(csv_fname)
        csv_fname = csv_fname+"results.csv"

    title = outname.split("_")[0]

    default_outname = "vis/" + "".join([c for c in csv_fname if re.match(r'\w', c)]) + key_name
    outname = outname if outname is not None else default_outname
    datafname = csv_fname

    data = pandas.read_csv(datafname)
    xvals = (data['dim0'].values)
    yvals = (data['offset'].values)
    zvals = (data[key_name].values)

    Z = zvals
    real_Z = zvals.copy()
    # Take numerically stable log of data
    if logscale:
        Z_neg = Z[Z < 0]
        Z_pos = Z[Z >= 0]
        Z_neg = -np.log10(1-Z_neg)
        Z_pos = np.log10(1+Z_pos)
        Z[Z < 0] = Z_neg
        Z[Z >= 0] = Z_pos
    zvals = Z

    x_size = len(np.unique(xvals))
    x_dims = np.sort(np.unique(xvals))

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    # Plot a basic wireframe.
    for i in range(x_size):
        boolmask = np.where(x_dims[i] == xvals)
        indexes = np.argsort(yvals[boolmask])
        ax.plot(xvals[boolmask][indexes], yvals[boolmask][indexes], zvals[boolmask][indexes])

    if logscale:
        # Find the highest order of magnitude
        max_Z = np.max(real_Z)
        if max_Z < 0:
            max_magnitude = -math.floor(math.log10(-max_Z))
        else:
            max_magnitude = math.floor(math.log10(max_Z))

        # Find the lowest order of magnitude
        min_Z = np.min(real_Z)
        if min_Z < 0:
            min_magnitude = -math.floor(math.log10(-min_Z))
        else:
            min_magnitude = math.floor(math.log10(min_Z))

        # Manually set colorbar and z axis tick text
        continuous_labels = np.round(np.linspace(min_magnitude, max_magnitude, 8, endpoint=True))
        zticks = []
        ztick_labels = []
        for label in continuous_labels:
            # Format label
            zticks.append(label)
            if label > 2 or label < -2:
                label = "$-10^{" + str(int(-label)) + "}$" if label < 0 else "$10^{" + str(int(label)) + "}$"
            else:
                label = "{}".format(-10.0**(-label)) if label < 0 else "{}".format(10.0**label)
            ztick_labels.append("    " + label)
        ax.set_zticks(zticks)
        ax.set_zticklabels(ztick_labels)

    ax.set_title(title)
    ax.set_xlabel('Training time')
    ax.set_ylabel('Gradient step size')
    ax.set_zlabel('Rewards', labelpad=12)
    fig.savefig(outname, dpi=300,
                bbox_inches='tight', format='png')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='generate jobs for plane')
    parser.add_argument('results', type=str)
    parser.add_argument('--outname', type=str, help="if specified, outputs file with this name (extension added onto name)")
    parser.add_argument('--key', type=str, default="episode_rewards", help="key in csv file to plot")
    parser.add_argument('--logscale', action="store_true", help="key in csv file to plot")

    args = parser.parse_args()

    plot_grad_search(args.results, args.outname, args.key, logscale=args.logscale)
