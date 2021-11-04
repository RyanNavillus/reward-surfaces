import glob
import argparse
import pandas
import numpy as np


def isqrt(n):
    x = n
    y = (x + 1) // 2
    while y < x:
        x = y
        y = (x + n // x) // 2
    return x


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate flatness metrics for existing plots')
    parser.add_argument('folder', type=str, help="Folder containing results csv files from plot_plane script.")
    parser.add_argument('--key', type=str, default="episode_rewards", help="key in csv file to plot")
    args = parser.parse_args()

    files = glob.glob(args.folder + "*results.csv", recursive=False)
    files = sorted(files)
    for csv_file in files:
        name = csv_file.split("/")[-1].split("@")[0]

        # Check that data is complete and extract x,y values
        data = pandas.read_csv(csv_file)
        dsize = isqrt(len(data['dim0']))
        if dsize <= 1 or dsize**2 != len(data['dim0']):
            print(csv_file, "is not complete!")
            continue

        xvals = (data['dim0'].values)
        yvals = (data['dim1'].values)
        zvals = (data[args.key].values)

        # Sort x, y, z values according to x + 1000000(dsize^2)(y)
        idxs = np.argsort(xvals + yvals*1000000*len(data['dim0']))
        xvals = xvals[idxs].reshape(dsize, dsize)
        yvals = yvals[idxs].reshape(dsize, dsize)
        zvals = zvals[idxs].reshape(dsize, dsize)

        # Print flatness metric (stddev)
        flat_data = zvals
        flat_data = flat_data / np.ptp(flat_data)
        flat_data = flat_data + abs(np.min(flat_data))
        print(name)

        square_size = dsize
        matrix = flat_data

        global_std = np.std(matrix)
        print("Global std is: ", global_std)

        WINDOW_SIZE = 10  # assumes square kernel
        kernel_length = square_size - WINDOW_SIZE + 1

        local_stds = []
        for i in range(kernel_length):
            for j in range(kernel_length):
                local_stds.append(np.std(matrix[j:j+WINDOW_SIZE, i:i+WINDOW_SIZE]))

        print("Max local std is: ", max(local_stds))
        print()
