import argparse
from reward_surfaces.plotting import plot_plane


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='generate jobs for plane')
    parser.add_argument('datafname', type=str)
    parser.add_argument('--outname', type=str, help="if specified, outputs file with this name (extension added onto name)")
    parser.add_argument('--key', type=str, default="episode_rewards", help="key in csv file to plot")
    parser.add_argument('--type', type=str, default="mesh", help="plot type. Possible types are: [all, mesh, vtp, heat, contour, contourf]")
    parser.add_argument('--show', action='store_true', help="If true, shows plot instead of saving it (does not work for vtp output)")

    args = parser.parse_args()

    plot_plane(args.datafname, args.outname, args.key, args.type, args.show)
