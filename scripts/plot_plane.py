import argparse
from reward_surfaces.plotting import plot_plane


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='generate jobs for plane')
    parser.add_argument('datafname', type=str, "Path to results csv file for reward surface.")
    parser.add_argument('--outname',
                        type=str,
                        help="Outputs file name for reward surface (extension added onto name)")
    parser.add_argument('--env_name', type=str, help="Name of environment in plot")
    parser.add_argument('--key', type=str, default="episode_rewards", help="Key in csv file to plot")
    parser.add_argument('--type',
                        type=str,
                        default="mesh",
                        help="Plot type, must be one of all, mesh, vtp, heat, contour, or contourf")
    parser.add_argument('--dir1_scale', type=float, help="Scale of the x axis in the plot")
    parser.add_argument('--dir2_scale', type=float, help="Scale of the y axis in the plot")
    parser.add_argument('--dir1_name', type=str, help="Name of the x axis in the plot")
    parser.add_argument('--dir2_name', type=str, help="Name of the y axis in the plot")
    parser.add_argument('--show',
                        action='store_true',
                        help="Shows plot instead of saving it (does not work for vtp output)")
    parser.add_argument('--logscale',
                        action='store_true',
                        help="Plot in log scale rather than the default linear scale")

    args = parser.parse_args()

    plot_plane(args.datafname, args.outname,
               env_name=args.env_name,
               key_name=args.key,
               plot_type=args.type,
               dir1_name=args.dir1_name,
               dir2_name=args.dir2_name,
               dir1_scale=args.dir1_scale,
               dir2_scale=args.dir2_scale,
               show=args.show,
               logscale=args.logscale)
