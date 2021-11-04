import glob
import re
import argparse
import json
from reward_surfaces.plotting import plot_plane


def replot(csv_fname, outname=None, envname=None, key_name="episode_rewards", title=None, plot_type="mesh",
           file_type="png", logscale=False):
    default_outname = "vis/" + "".join([c for c in csv_fname if re.match(r'\w', c)]) + key_name
    outname = outname if outname is not None else default_outname
    datafname = csv_fname
    print(env_name)
    plot_plane(datafname, outname, key_name=args.key, plot_type=plot_type, file_type=file_type, envname=envname)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Replot data from csv files')
    parser.add_argument('folder', type=str, help="Folder containing results csv files from plot_plane script.")
    parser.add_argument('--key', type=str, default="episode_rewards", help="key in csv file to plot")
    parser.add_argument('--type', type=str, default="mesh", help="type of plot")
    parser.add_argument('--filetype', type=str, default="png", help="file extension of plot")
    parser.add_argument('--logscale', action="store_true", help="whether to use log scale for rewards")
    args = parser.parse_args()

    files = glob.glob(args.folder + "**/*results.csv", recursive=True)
    files = sorted(files)
    print(files)
    for csv_file in files:
        # Open info file
        info_filepath = "/".join(csv_file.split("/")[:-1]) + "/info.json"
        info_file = open(info_filepath)
        info = json.load(info_file)

        env_name = info["env"]
        # Get checkpoint name from info
        checkpoint = info["checkpoint_dir"]
        end_index = checkpoint.find("_checkpoints")
        start_index = checkpoint[::-1].find("/", 0, len(checkpoint) - end_index) + 1
        name = checkpoint[start_index:end_index]

        path = "/".join(csv_file.split("/")[:-2])
        replot(csv_file, outname=name, envname=env_name, plot_type=args.type, file_type=args.filetype,
               logscale=args.logscale)
