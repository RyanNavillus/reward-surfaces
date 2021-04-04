import argparse
from reward_surfaces.plotting import plot_plane
import pandas
import os
import matplotlib.pyplot as plt
from reward_surfaces.utils import job_results_to_csv
import re
import numpy as np

def plot_grad_search(csv_fname, outname=None, key_name="episode_rewards"):
    if os.path.isdir(csv_fname):
        job_results_to_csv(csv_fname)
        csv_fname = csv_fname+"results.csv"

    default_outname = "vis/" + "".join([c for c in csv_fname if re.match(r'\w', c)]) + key_name
    outname = outname if outname is not None else default_outname
    datafname = csv_fname

    data = pandas.read_csv(datafname)
    xvals = (data['dim0'].values)
    yvals = (data['offset'].values)
    zvals = (data[key_name].values)#.reshape(dsize,dsize)

    x_size = len(np.unique(xvals))
    y_size = len(np.unique(yvals))

    x_dims = np.sort(np.unique(xvals))
    # y_dims = np.sort(np.unique(yvals))

    # x_arr = np.zeros((y_size, x_size))
    # y_arr = np.zeros((y_size, x_size))
    # z_arr = np.zeros((y_size, x_size))

    idxs = np.argsort(xvals + yvals*1000000*x_size)
    #print("\n".join(str(x) for x in (zip(xvals[idxs],yvals[idxs]))))
    # xvals = xvals[idxs].reshape(y_size,x_size).transpose()
    # yvals = yvals[idxs].reshape(y_size,x_size).transpose()
    # zvals = zvals[idxs].reshape(y_size,x_size).transpose()

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    # Plot a basic wireframe.
    for i in range(x_size):
        boolmask = np.where(x_dims[i] == xvals)
        indexes = np.argsort(yvals[boolmask])
        ax.plot(xvals[boolmask][indexes], yvals[boolmask][indexes], zvals[boolmask][indexes])#, rstride=1, cstride=1)

    ax.set_xlabel('Training time')
    ax.set_ylabel('Gradient step size')
    ax.set_zlabel('Rewards')
    fig.savefig(outname, dpi=300,
                bbox_inches='tight', format='png')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='generate jobs for plane')
    parser.add_argument('results', type=str)
    parser.add_argument('--outname', type=str, help="if specified, outputs file with this name (extension added onto name)")
    parser.add_argument('--key', type=str, default="episode_rewards", help="key in csv file to plot")

    args = parser.parse_args()

    plot_grad_search(args.results, args.outname, args.key)
