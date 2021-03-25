import pandas
import sys
import matplotlib.pyplot as plt
import numpy as np
import re

def plot_traj(csv_file, key="episode_rewards"):
    df = pandas.read_csv(csv_file)
    idxs = np.argsort(df['dim0'])
    fig = plt.figure()
    ax = plt.axes()
    ax.plot(df['dim0'][idxs], df[key][idxs])
    out_fname = "vis/"+"".join([c for c in csv_file if re.match(r'\w', c)]) + key
    plt.savefig(out_fname)
