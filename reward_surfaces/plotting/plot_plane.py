import numpy as np

from mpl_toolkits.mplot3d import Axes3D
from matplotlib import pyplot as plt
from matplotlib import cm
import re
import seaborn as sns

import math
import pandas
import warnings
from scipy import interpolate
from reward_surfaces.utils import REWARDCLASSES, ENVCLASSES


def plot_2d_contour(x_coords, y_coords, z_values, magnitude, base_name, vmin=0.1, vmax=10, vlevel=0.5, show=False,
                    plot_type='mesh', file_type="png", dir1_scale=1., dir2_scale=1., dir1_name="dim1", dir2_name="dim2",
                    env_name=None, key_name=None, logscale=False, autologscale=True):
    """Plot 2D contour map and 3D surface."""
    X = x_coords
    Y = y_coords
    Z = z_values

    title = env_name

    # Remove Atari name options
    env_name = env_name.replace("NoFrameskip", "")
    env_name = env_name.replace("-v0", "")
    env_name = env_name.replace("-v1", "")
    env_name = env_name.replace("-v2", "")
    env_name = env_name.replace("-v3", "")
    env_name = env_name.replace("-v4", "")
    env_name = env_name.replace("-v5", "")
    env_name = env_name.replace("Deterministic", "")

    # Construct Title
    if env_name in ENVCLASSES:
        title += " | " + ENVCLASSES[env_name]
    else:
        warnings.warn("Environment is not listed in reward_surfaces/utils/plot_utils.py, plots titles may be missing information.")
    if env_name in REWARDCLASSES:
        title += " | " + REWARDCLASSES[env_name]

    # Label max value
    if key_name == "episode_rewards":
        title += " | " + f"Max Reward: {np.max(Z):.02f}"
    elif key_name == "episode_std_rewards":
        title += " | " + f"Max Standard Deviation: {np.max(Z):.02f}"
    else:
        title += " | " + f"Max Value: {np.max(Z):.02f}"

    # Automatically choose logscale
    if not logscale and autologscale:
        if np.max(Z) - np.min(Z) > 10000:
            logscale = True

    # --------------------------------------------------------------------
    # Plot 2D contours
    # --------------------------------------------------------------------
    if plot_type == 'all' or plot_type == 'contour':
        fig = plt.figure()
        CS = plt.contour(X, Y, Z, cmap='summer', levels=np.arange(vmin, vmax, vlevel))
        plt.clabel(CS, inline=1, fontsize=8)
        out_fname = base_name + '_2dcontour.' + file_type
        fig.savefig(out_fname, dpi=300,
                    bbox_inches='tight', format=file_type)

    if plot_type == 'all' or plot_type == 'contourf':
        fig = plt.figure()
        print(base_name + '_2dcontourf' + '.png')
        CS = plt.contourf(X, Y, Z, cmap='summer', levels=np.arange(vmin, vmax, vlevel))
        out_fname = base_name + '_2dcontourf.' + file_type
        fig.savefig(out_fname, dpi=300,
                    bbox_inches='tight', format=file_type)

    # --------------------------------------------------------------------
    # Plot 2D heatmaps
    # --------------------------------------------------------------------
    if plot_type == 'all' or plot_type == 'heat':
        if file_type != "pgf":
            sns.set_theme(font="Serif")
        dir1_name = "Grad Direction"
        dir2_name = "Random Direction"
        size = len(X[0])
        dir1_magnitude = math.floor(math.log10(abs(dir1_scale)))
        dir2_magnitude = math.floor(math.log10(abs(dir2_scale)))

        labels_d1 = [f"{x:0.1f}" for x in (np.arange(size)-size//2)/(size/2)*dir1_scale*(10**-dir1_magnitude)]
        labels_d2 = [f"{x:0.1f}" for x in (np.arange(size)-size//2)/(size/2)*dir2_scale*(10**-dir2_magnitude)]
        ax = plt.axes()
        sns_plot = sns.heatmap(Z, cmap='viridis', cbar=True, vmin=vmin, vmax=vmax,
                               xticklabels=labels_d1, yticklabels=labels_d2)
        sns_plot.invert_yaxis()
        xlabel = dir1_name + r" ($10^{{{}}}$)".format(dir1_magnitude) if dir1_magnitude != 0 else dir1_name
        ylabel = dir2_name + r" ($10^{{{}}}$)".format(dir2_magnitude) if dir2_magnitude != 0 else dir2_name
        sns_plot.set(xlabel=xlabel, ylabel=ylabel)
        out_fname = base_name + '_2dheat.' + file_type
        ax.set_title(title)
        sns_plot.get_figure().savefig(out_fname, dpi=300, bbox_inches='tight', format=file_type)

    # --------------------------------------------------------------------
    # Plot 3D surface
    # --------------------------------------------------------------------
    if plot_type == 'all' or plot_type == 'mesh':
        fig = plt.figure()
        tnrfont = {'fontname': 'Sans'}
        ax = Axes3D(fig)

        fig.suptitle(title)
        if file_type != "pgf":
            fig.suptitle(title, **tnrfont)

        if np.min(Z) < -1e9 and not logscale:
            print("Warning: Data includes extremely large negative rewards ({:3E}).\
                  Consider setting logscale=True".format(np.min(Z)))

        # Scale X and Y values by the step size magnitude
        X = magnitude * X
        Y = magnitude * Y

        real_Z = Z.copy()
        # Take numerically stable log of data
        if logscale:
            Z_neg = Z[Z < 0]
            Z_pos = Z[Z >= 0]
            Z_neg = -np.log10(1-Z_neg)
            Z_pos = np.log10(1+Z_pos)
            Z[Z < 0] = Z_neg
            Z[Z >= 0] = Z_pos

        # Plot surface
        surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm, linewidth=0, antialiased=False, zorder=5)

        # Add max text
        # center = len(Z) // 2
        # ax.text(0.05, 0.05, np.max(Z), f"{real_Z[center][center]:.2f}", color='black')

        # Plot center line above surface
        Z_range = abs(np.max(Z) - np.min(Z))
        zline_above = np.linspace(Z[len(Z) // 2][len(Z[0]) // 2], np.max(Z) + (Z_range * 0.1), 4)
        xline_above = 0 * zline_above
        yline_above = 0 * zline_above
        ax.plot3D(xline_above, yline_above, zline_above, 'black', zorder=10)

        # Plot center line below surface
        zline_below = np.linspace(Z[len(Z) // 2][len(Z[0]) // 2], np.min(Z) - (Z_range * 0.1), 4)
        xline_below = 0 * zline_below
        yline_below = 0 * zline_below
        ax.plot3D(xline_below, yline_below, zline_below, 'black', zorder=0)

        # Fix colorbar labeling for log scale
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

            # Create colorbar
            continuous_labels = np.round(np.linspace(min_magnitude, max_magnitude, 8, endpoint=True))
            cbar = fig.colorbar(surf, shrink=0.5, aspect=5, ticks=continuous_labels, pad=0.1)
            cbar.set_ticks(list())

            # Manually set colorbar and z axis tick text
            zticks = []
            ztick_labels = []
            bounds = cbar.ax.get_ybound()
            for index, label in enumerate(continuous_labels):
                x = 6.0
                y = bounds[0] + (bounds[1] - bounds[0]) * index / 8

                # Format label
                zticks.append(label)
                if label > 2 or label < -2:
                    label = "$-10^{" + str(int(-label)) + "}$" if label < 0 else "$10^{" + str(int(label)) + "}$"
                else:
                    label = "${}$".format(-10.0**(-label)) if label < 0 else "${}$".format(10.0**label)
                cbar.ax.text(x, y, label)
                ztick_labels.append("    " + label)
            ax.set_zticks(zticks)
            ax.set_zticklabels(ztick_labels)
            if file_type != "pgf":
                ax.set_zticklabels(ztick_labels, **tnrfont)
        else:
            fig.colorbar(surf, shrink=0.5, aspect=5, pad=0.05)

        # Save plot
        out_fname = base_name + '_3dsurface.' + file_type
        fig.savefig(out_fname, dpi=300,
                    bbox_inches='tight', format=file_type)

    if show:
        plt.show()

    return out_fname


def generate_vtp(xcoordinates, ycoordinates, vals, vtp_file, log=False, zmax=-1, interp=-1):
    # Set this to True to generate points
    show_points = False
    # Set this to True to generate polygons
    show_polys = True

    # Flatten arrays
    x_array = xcoordinates[:].ravel()
    y_array = ycoordinates[:].ravel()
    z_array = vals[:].ravel()

    # Interpolate the resolution up to the desired amount
    if interp > 0:
        m = interpolate.interp2d(xcoordinates[0, :], ycoordinates[:, 0], vals, kind='cubic')
        x_array = np.linspace(min(x_array), max(x_array), interp)
        y_array = np.linspace(min(y_array), max(y_array), interp)
        z_array = m(x_array, y_array).ravel()

        x_array, y_array = np.meshgrid(x_array, y_array)
        x_array = x_array.ravel()
        y_array = y_array.ravel()

    print("Here's your output file:{}".format(vtp_file))

    number_points = len(z_array)
    print("number_points = {} points".format(number_points))

    matrix_size = int(math.sqrt(number_points))
    print("matrix_size = {} x {}".format(matrix_size, matrix_size))

    poly_size = matrix_size - 1
    print("poly_size = {} x {}".format(poly_size, poly_size))

    number_polys = poly_size * poly_size
    print("number_polys = {}".format(number_polys))

    min_value_array = [min(x_array), min(y_array), min(z_array)]
    max_value_array = [max(x_array), max(y_array), max(z_array)]
    min_value = min(min_value_array)
    max_value = max(max_value_array)

    averaged_z_value_array = []

    poly_count = 0
    for column_count in range(poly_size):
        stride_value = column_count * matrix_size
        for row_count in range(poly_size):
            temp_index = stride_value + row_count
            averaged_z_value = (z_array[temp_index] + z_array[temp_index + 1] +
                                z_array[temp_index + matrix_size] +
                                z_array[temp_index + matrix_size + 1]) / 4.0
            averaged_z_value_array.append(averaged_z_value)
            poly_count += 1

    avg_min_value = min(averaged_z_value_array)
    avg_max_value = max(averaged_z_value_array)

    output_file = open(vtp_file, 'w')
    output_file.write('<VTKFile type="PolyData" version="1.0" byte_order="LittleEndian" header_type="UInt64">\n')
    output_file.write('  <PolyData>\n')

    if (show_points and show_polys):
        output_file.write('    <Piece NumberOfPoints="{}" NumberOfVerts="{}" NumberOfLines="0" NumberOfStrips="0" NumberOfPolys="{}">\n'.format(number_points, number_points, number_polys))
    elif (show_polys):
        output_file.write('    <Piece NumberOfPoints="{}" NumberOfVerts="0" NumberOfLines="0" NumberOfStrips="0" NumberOfPolys="{}">\n'.format(number_points, number_polys))
    else:
        output_file.write('    <Piece NumberOfPoints="{}" NumberOfVerts="{}" NumberOfLines="0" NumberOfStrips="0" NumberOfPolys="">\n'.format(number_points, number_points))

    # <PointData>
    output_file.write('      <PointData>\n')
    output_file.write('        <DataArray type="Float32" Name="zvalue" NumberOfComponents="1" format="ascii" RangeMin="{}" RangeMax="{}">\n'.format(min_value_array[2], max_value_array[2]))
    for vertexcount in range(number_points):
        if (vertexcount % 6) == 0:
            output_file.write('          ')
        output_file.write('{}'.format(z_array[vertexcount]))
        if (vertexcount % 6) == 5:
            output_file.write('\n')
        else:
            output_file.write(' ')
    if (vertexcount % 6) != 5:
        output_file.write('\n')
    output_file.write('        </DataArray>\n')
    output_file.write('      </PointData>\n')

    # <CellData>
    output_file.write('      <CellData>\n')
    if (show_polys and not show_points):
        output_file.write('        <DataArray type="Float32" Name="averaged zvalue" NumberOfComponents="1" format="ascii" RangeMin="{}" RangeMax="{}">\n'.format(avg_min_value, avg_max_value))
        for vertexcount in range(number_polys):
            if (vertexcount % 6) == 0:
                output_file.write('          ')
            output_file.write('{}'.format(averaged_z_value_array[vertexcount]))
            if (vertexcount % 6) == 5:
                output_file.write('\n')
            else:
                output_file.write(' ')
        if (vertexcount % 6) != 5:
            output_file.write('\n')
        output_file.write('        </DataArray>\n')
    output_file.write('      </CellData>\n')

    # <Points>
    output_file.write('      <Points>\n')
    output_file.write('        <DataArray type="Float32" Name="Points" NumberOfComponents="3" format="ascii" RangeMin="{}" RangeMax="{}">\n'.format(min_value, max_value))
    for vertexcount in range(number_points):
        if (vertexcount % 2) == 0:
            output_file.write('          ')
        output_file.write('{} {} {}'.format(x_array[vertexcount], y_array[vertexcount], z_array[vertexcount]))
        if (vertexcount % 2) == 1:
            output_file.write('\n')
        else:
            output_file.write(' ')
    if (vertexcount % 2) != 1:
        output_file.write('\n')
    output_file.write('        </DataArray>\n')
    output_file.write('      </Points>\n')

    # <Verts>
    def write_data_array(name="", range_min=0):
        output_file.write('        <DataArray type="Int64" Name="{}" format="ascii" RangeMin="{}" RangeMax="{}">\n'.format(name, range_min, number_points - 1 + range_min))
        if (show_points):
            for vertexcount in range(number_points):
                if (vertexcount % 6) == 0:
                    output_file.write('          ')
                output_file.write('{}'.format(vertexcount + range_min))
                if (vertexcount % 6) == 5:
                    output_file.write('\n')
                else:
                    output_file.write(' ')
            if (vertexcount % 6) != 5:
                output_file.write('\n')
        output_file.write('        </DataArray>\n')

    output_file.write('      <Verts>\n')
    write_data_array(name="connectivity", range_min=0)
    write_data_array(name="offsets", range_min=1)
    output_file.write('      </Verts>\n')

    # <Lines>
    output_file.write('      <Lines>\n')
    output_file.write('        <DataArray type="Int64" Name="connectivity" format="ascii" RangeMin="0" RangeMax="{}">\n'.format(number_polys - 1))
    output_file.write('        </DataArray>\n')
    output_file.write('        <DataArray type="Int64" Name="offsets" format="ascii" RangeMin="1" RangeMax="{}">\n'.format(number_polys))
    output_file.write('        </DataArray>\n')
    output_file.write('      </Lines>\n')

    # <Strips>
    output_file.write('      <Strips>\n')
    output_file.write('        <DataArray type="Int64" Name="connectivity" format="ascii" RangeMin="0" RangeMax="{}">\n'.format(number_polys - 1))
    output_file.write('        </DataArray>\n')
    output_file.write('        <DataArray type="Int64" Name="offsets" format="ascii" RangeMin="1" RangeMax="{}">\n'.format(number_polys))
    output_file.write('        </DataArray>\n')
    output_file.write('      </Strips>\n')

    # <Polys>
    output_file.write('      <Polys>\n')
    output_file.write('        <DataArray type="Int64" Name="connectivity" format="ascii" RangeMin="0" RangeMax="{}">\n'.format(number_polys - 1))
    if (show_polys):
        polycount = 0
        for column_count in range(poly_size):
            stride_value = column_count * matrix_size
            for row_count in range(poly_size):
                temp_index = stride_value + row_count
                if (polycount % 2) == 0:
                    output_file.write('          ')
                output_file.write('{} {} {} {}'.format(temp_index, (temp_index + 1), (temp_index + matrix_size + 1), (temp_index + matrix_size)))
                if (polycount % 2) == 1:
                    output_file.write('\n')
                else:
                    output_file.write(' ')
                polycount += 1
        if (polycount % 2) == 1:
            output_file.write('\n')
    output_file.write('        </DataArray>\n')
    output_file.write('        <DataArray type="Int64" Name="offsets" format="ascii" RangeMin="1" RangeMax="{}">\n'.format(number_polys))
    if (show_polys):
        for polycount in range(number_polys):
            if (polycount % 6) == 0:
                output_file.write('          ')
            output_file.write('{}'.format((polycount + 1) * 4))
            if (polycount % 6) == 5:
                output_file.write('\n')
            else:
                output_file.write(' ')
        if (polycount % 6) != 5:
            output_file.write('\n')
    output_file.write('        </DataArray>\n')
    output_file.write('      </Polys>\n')

    output_file.write('    </Piece>\n')
    output_file.write('  </PolyData>\n')
    output_file.write('</VTKFile>\n')
    output_file.write('')
    output_file.close()

    print("Done with file:{}".format(vtp_file))
    return vtp_file


def isqrt(n):
    x = n
    y = (x + 1) // 2
    while y < x:
        x = y
        y = (x + n // x) // 2
    return x


def generate_missing_jobs(csv_path):
    data = pandas.read_csv(csv_path)
    parent_path = "/".join(csv_path.split("/")[:-1])
    data['dim0'] = round(data['dim0'] * data['scale'])
    data['dim1'] = round(data['dim1'] * data['scale'])
    scale = int(data['scale'][0])
    jobs = []
    for d1 in range(-scale, scale + 1):
        for d2 in range(-scale, scale + 1):
            if data.loc[(data['dim0'] == d1) & (data['dim1'] == d2)].empty:
                jobs.append(f"python3 -m reward_surfaces.bin.eval_plane_job {parent_path} --offset1={d1} --offset2={d2}")
    with open(parent_path + "/remaining_jobs.sh", "w") as remaining_jobs:
        for job in jobs:
            remaining_jobs.write(job + "\n")
    print(f"Wrote incomplete jobs to {parent_path + '/remaining_jobs.sh'}")


def plot_plane(csv_fname, out_name=None, env_name=None, key_name="episode_rewards", plot_type="mesh", file_type="png",
               show=False, dir1_scale=1, dir2_scale=1., dir1_name="dim1", dir2_name="dim2", vmin=None, vmax=None,
               logscale=False):
    default_outname = "vis/" + "".join([c for c in csv_fname if re.match(r'\w', c)]) + key_name + "_" + plot_type
    out_name = out_name if out_name is not None else default_outname
    data_fname = csv_fname

    # Check that data is complete and extract x,y values
    data = pandas.read_csv(data_fname)
    dsize = isqrt(len(data['dim0']))
    if dsize <= 1 or dsize**2 != len(data['dim0']):
        print(csv_fname, "is not complete! Exiting")
        generate_missing_jobs(data_fname)
        return None
    xvals = (data['dim0'].values)
    yvals = (data['dim1'].values)
    zvals = (data[key_name].values)

    # Sort x, y, z values according to x + 1000000(dsize^2)(y)
    idxs = np.argsort(xvals + yvals*1000000*len(data['dim0']))
    xvals = xvals[idxs].reshape(dsize, dsize)
    yvals = yvals[idxs].reshape(dsize, dsize)
    zvals = zvals[idxs].reshape(dsize, dsize)

    if vmin is None:
        vmin = np.min(zvals)
    if vmax is None:
        vmax = np.max(zvals)

    vlevel = (vmax-vmin)/15
    out_name = out_name + "_" + key_name.replace('_', '')
    magnitude = data.iloc[0]['magnitude'] if 'magnitude' in data else 1
    dir1_scale = data.iloc[0]['dir1_scale'] if 'dir1_scale' in data else dir1_scale
    dir2_scale = data.iloc[0]['dir2_scale'] if 'dir2_scale' in data else dir2_scale
    return plot_2d_contour(xvals, yvals, zvals, magnitude, out_name, vmin=vmin, vmax=vmax, vlevel=vlevel,
                           plot_type=plot_type, file_type=file_type, show=show, dir1_scale=dir1_scale,
                           dir2_scale=dir2_scale, dir1_name=dir1_name, dir2_name=dir2_name, env_name=env_name,
                           key_name=key_name, logscale=logscale)
