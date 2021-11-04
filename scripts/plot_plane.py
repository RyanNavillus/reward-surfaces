import argparse
from reward_surfaces.plotting import plot_plane


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='generate jobs for plane')
    parser.add_argument('datafname', type=str)
    parser.add_argument('--outname',
                        type=str,
                        help="if specified, outputs file with this name (extension added onto name)")
    parser.add_argument('--key', type=str, default="episode_rewards", help="key in csv file to plot")
    parser.add_argument('--type',
                        type=str,
                        default="mesh",
                        help="plot type. Possible types are: [all, mesh, vtp, heat, contour, contourf]")
    parser.add_argument('--dir1_scale', type=float, help="Scale of the x axis in the plot")
    parser.add_argument('--dir2_scale', type=float, help="Scale of the y axis in the plot")
    parser.add_argument('--dir1_name', type=str, help="Name of the x axis in the plot")
    parser.add_argument('--dir2_name', type=str, help="Name of the y axis in the plot")
    parser.add_argument('--show',
                        action='store_true',
                        help="If true, shows plot instead of saving it (does not work for vtp output)")
    parser.add_argument('--logscale',
                        action='store_true',
                        help="If true, plot in log scale rather than the default linear scale")

    args = parser.parse_args()

    plot_plane(args.datafname, args.outname, args.key, args.type,
               dir1_name=args.dir1_name,
               dir2_name=args.dir2_name,
               dir1_scale=args.dir1_scale,
               dir2_scale=args.dir2_scale,
               show=args.show,
               logscale=args.logscale)
