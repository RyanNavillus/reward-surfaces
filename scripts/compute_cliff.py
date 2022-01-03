import glob
import os
import re
import pandas
import argparse
import numpy as np
import matplotlib.pyplot as plt


def isqrt(n):
    x = n
    y = (x + 1) // 2
    while y < x:
        x = y
        y = (x + n // x) // 2
    return x


def choose_color(is_cliff, index):
    cliff_colors = ["maroon", "red", "orange", "gold", "saddlebrown"]
    standard_colors = ["lawngreen", "green", "cyan", "dodgerblue", "navy"]
    if is_cliff:
        return cliff_colors[index % 5]
    else:
        return standard_colors[index % 5]


def print_cliff_values(csv_fname, outname=None, key_name="episode_rewards", title=None, cliff_ratio=0.5,
                       global_ratio=0.25):
    default_outname = "vis/" + "".join([c for c in csv_fname if re.match(r'\w', c)]) + key_name
    outname = outname if outname is not None else default_outname
    outname = outname + "_cliff"
    datafname = csv_fname

    data = pandas.read_csv(datafname)
    xvals = (data['dim0'].values)
    yvals = (data['offset'].values)
    zvals = (data[key_name].values)#.reshape(dsize,dsize)
    global_min = min(zvals)
    global_max = max(zvals)
    global_range = global_max - global_min

    x_size = len(np.unique(xvals))
    y_size = len(np.unique(yvals))

    x_dims = np.sort(np.unique(xvals))

    idxs = np.argsort(xvals + yvals*1000000*x_size)

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    # Plot a basic wireframe.
    for i in range(x_size):
        boolmask = np.where(x_dims[i] == xvals)
        indexes = np.argsort(yvals[boolmask])
        sorted_z = zvals[boolmask][indexes]

        # Compute metric local percentage with global threshold
        is_cliff = False
        # First 10 points are in 1 high-res step
        if sorted_z[9] <= cliff_ratio * sorted_z[0] and (sorted_z[0] - sorted_z[9]) / global_range > global_ratio:
            is_cliff = True
        second_last_val = sorted_z[0]
        last_val = sorted_z[9]
        for z in sorted_z[10:]:
            if ((z <= cliff_ratio * last_val and (last_val - z) / global_range > global_ratio)
               or (z <= cliff_ratio * second_last_val and (second_last_val - z) / global_range > global_ratio)):
                is_cliff = True
            second_last_val = last_val
            last_val = z

        # Compute global percentage metric
        #is_cliff = False
        ## First 10 points are in 1 high-res step
        #if (sorted_z[0] - sorted_z[9]) / global_range > cliff_ratio:
        #    is_cliff = True
        #second_last_val = sorted_z[8]
        #last_val = sorted_z[9]
        #for z in sorted_z[10:]:
        #    short_drop = last_val - z
        #    long_drop = second_last_val - z
        #    if short_drop / global_range > cliff_ratio or long_drop / global_range > cliff_ratio:
        #        is_cliff = True
        #    second_last_val = last_val
        #    last_val = z

        #max_cliff = 1
        ## First 10 points are in 1 high-res step
        #percent_remaining = sorted_z[9] / sorted_z[0]
        #if percent_remaining < 1:
        #    max_cliff = min(max_cliff, percent_remaining)
        #second_last_val = sorted_z[8]
        #last_val = sorted_z[9]
        #for z in sorted_z[10:]:
        #    percent_remaining = z / last_val
        #    if percent_remaining < 1:
        #        max_cliff = min(max_cliff, percent_remaining)

        #    percent_remaining = z / second_last_val
        #    if percent_remaining < 1:
        #        max_cliff = min(max_cliff, percent_remaining)
        #    second_last_val = last_val
        #    last_val = z


        #print(sorted_z)
        print("\t", x_dims[i], is_cliff)
        color = choose_color(is_cliff, i)
        ax.plot(xvals[boolmask][indexes], yvals[boolmask][indexes], zvals[boolmask][indexes], color=color)#, rstride=1, cstride=1)

    ax.set_title(title)
    ax.set_xlabel('Training time')
    ax.set_ylabel('Gradient step size')
    ax.set_zlabel('Rewards')
    fig.savefig(outname, dpi=300, bbox_inches='tight', format='png')
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate flatness metrics for existing plots')
    parser.add_argument('folder', type=str, help="Folder containing results csv files from plot_plane script.")
    parser.add_argument('--key', type=str, default="episode_rewards", help="key in csv file to plot")
    parser.add_argument('--ratio', type=float, default=0.5, help="Percent drop in reward that identifies a cliff")
    parser.add_argument('--globalratio', type=float, default=0.25, help="Percent drop in global reward that identifies a cliff")
    args = parser.parse_args()

    files = glob.glob(args.folder + "**/*results.csv", recursive=True)
    files = sorted(files)
    print(files)
    for csv_file in files:
        name = csv_file.split("/")[-2]
        print(name)
        path = "/".join(csv_file.split("/")[:-2])
        print_cliff_values(csv_file, outname=path + "/" + name + "_cliffs", title=name,
                           cliff_ratio=args.ratio, global_ratio=args.globalratio)
        print()


