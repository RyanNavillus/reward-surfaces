import glob
import re
import argparse
import json
from reward_surfaces.plotting import plot_plane

ENVNAME = {
    "breakout": "Breakout-v0",
    "pong": "Pong-v0",
    "spaceinvaders": "SpaceInvaders-v0",
    "mspacman": "MsPacman-v0",
    "qbert": "Qbert-v0",
    "bankheist": "BankHeist-v0",
    "montezuma": "MontezumaRevenge-v0",
    "pitfall": "Pitfall-v0",
    "venture": "Venture-v0",
    "freeway": "Freeway-v0",
    "privateeye": "PrivateEye-v0",
    "solaris": "Solaris-v0",
    "acrobot": "Acrobot-v1",
    "cartpole": "Cartpole-v1",
    "mountaincar": "MountainCar-v0",
    "mountaincarcontinuous": "MountainCarContinuous-v0",
    "pendulum": "Pendulum-v0",
    "ant": "Ant-v2",
    "halfcheetah": "HalfCheetah-v2",
    "hopper": "Hopper-v2",
    "humanoid": "Humanoid-v2",
    "humanoidstandup": "HumanoidStandup-v2",
    "inverteddoublependulum": "InvertedDoublePendulum-v2",
    "invertedpendulum": "InvertedPendulum-v2",
    "reacher": "Reacher-v2",
    "swimmer": "Swimmer-v2",
    "walker2d": "Walker2d-v2",
}


def replot(csv_fname, outname=None, env_name=None, key_name="episode_rewards", title=None, plot_type="mesh",
           file_type="png", logscale=False):
    default_outname = "vis/" + "".join([c for c in csv_fname if re.match(r'\w', c)]) + key_name
    outname = outname if outname is not None else default_outname
    datafname = csv_fname
    plot_plane(datafname, outname, key_name=args.key, plot_type=plot_type, file_type=file_type, env_name=env_name, logscale=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Replot data from csv files')
    parser.add_argument('folder', type=str, help="Folder containing results csv files from plot_plane script.")
    parser.add_argument('--file', type=str, help="Specific csv to replot")
    parser.add_argument('--key', type=str, default="episode_rewards", help="key in csv file to plot")
    parser.add_argument('--type', type=str, default="mesh", help="type of plot")
    parser.add_argument('--filetype', type=str, default="png", help="file extension of plot")
    parser.add_argument('--logscale', action="store_true", help="whether to use log scale for rewards")
    args = parser.parse_args()

    if args.file:
        files = [args.file]
    else:
        files = glob.glob(args.folder + "**/results.csv", recursive=True)
        files = sorted(files)
    print(files)
    for csv_file in files:
        # Open info file
        info_filepath = "/".join(csv_file.split("/")[:-1]) + "/info.json"
        try:
            open("fakefile")
            info_file = open(info_filepath)
            info = json.load(info_file)
            env_name = info["env"]

            # Get checkpoint name from info
            checkpoint = info["checkpoint_dir"]
            end_index = checkpoint.find("_checkpoints")
            start_index = checkpoint[::-1].find("/", 0, len(checkpoint) - end_index) + 1
            name = checkpoint[start_index:end_index]
        except FileNotFoundError:
            print(csv_file)
            name = csv_file.split("/")[-1].split("_")[0].split("@")[0]
            name = csv_file.split("/")[3].split("_")[0].split("@")[0]
            print(name)
            env_name = ENVNAME[name]
            path = csv_file.split("/")[-1].split("@")[0]
            print(path)

        replot(csv_file, outname=name, env_name=env_name, plot_type=args.type, file_type=args.filetype,
               logscale=False)
