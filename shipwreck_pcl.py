#!/usr/bin/env python

import os
from webbrowser import get
import matplotlib
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from evo.core import metrics, trajectory
from evo.tools import file_interface
from evo.tools import log, plot
import open3d
import copy
from multicolor_line import HandlerDashedLines
import matplotlib.collections as mcol
from matplotlib.lines import Line2D
import math

log.configure_logging(verbose=True)

sns.set(style='white', font='sans-serif',
        font_scale=1.5, color_codes=False)
rc = {
    "lines.linewidth": 1.2,
    "text.usetex": True,
    "font.family": 'sans-serif',
    "pgf.texsystem": 'pdflatex',
    "font.weight": 'bold',
    'text.latex.preamble': r'\usepackage{amsmath} \usepackage{amssymb}'
}

matplotlib.rcParams.update(rc)
plt.rcParams.update(rc)


pose_relation = metrics.PoseRelation.translation_part


def set_aspect_equal_3d(ax: plt.Axes) -> None:
    """
    kudos to https://stackoverflow.com/a/35126679
    :param ax: matplotlib 3D axes object
    """
    xlim = ax.get_xlim3d()
    ylim = ax.get_ylim3d()
    zlim = ax.get_zlim3d()

    from numpy import mean
    xmean = mean(xlim)
    ymean = mean(ylim)
    zmean = mean(zlim)

    plot_radius = max([
        abs(lim - mean_)
        for lims, mean_ in ((xlim, xmean), (ylim, ymean), (zlim, zmean))
        for lim in lims
    ])

    ax.set_xlim3d([xmean - plot_radius, xmean + plot_radius])
    ax.set_ylim3d([ymean - plot_radius, ymean + plot_radius])
    ax.set_zlim3d([zmean - plot_radius, zmean + plot_radius])


def prepare_axis(fig: plt.Figure, plot_mode: plot.PlotMode = plot.PlotMode.xy, subplot_arg: int = 111) -> plt.Axes:
    if plot_mode == plot.PlotMode.xyz:
        ax = fig.add_subplot(subplot_arg, projection="3d")
    else:
        ax = fig.add_subplot(subplot_arg)
        ax.axis("equal")

    xlabel = r'$\boldsymbol{x} \textbf{(m)}$'
    ylabel = r'$\boldsymbol{y} \textbf{(m)}$'
    ax.set_xlabel(xlabel, fontsize=14, fontweight='bold')
    ax.set_ylabel(ylabel, fontsize=14, fontweight='bold')
    if plot_mode == plot.PlotMode.xyz:
        ax.set_zlabel(r'$\boldsymbol{y} \textbf{(m)}$')
    return ax


def plot_traj(ax: plt.Axes, traj: trajectory.PosePath3D, plot_mode: plot.PlotMode = plot.PlotMode.xy,
              style: str = '-', color: str = 'black', label: str = "",
              alpha: float = 1.0) -> None:
    x_idx = 0
    y_idx = 1
    z_idx = 2
    x = traj.positions_xyz[:, x_idx]
    y = traj.positions_xyz[:, y_idx]
    if plot_mode == plot.PlotMode.xyz:
        z = traj.positions_xyz[:, z_idx]
        ax.plot(x, y, z, style, color=color, label=label, alpha=alpha)
        set_aspect_equal_3d(ax)
    else:
        ax.plot(x, y, style, color=color, label=label, alpha=alpha)
    if label:
        ax.legend(frameon=True)


def read_switch_info(filename: str):
    f = open(filename)
    f.readline()
    all_lines = []
    lines = f.readlines()
    for line in lines:
        line = line.strip()
        line = line.split(" ")
        line = [np.double(x) for x in line]
        all_lines.append(line)
    f.close()
    return all_lines


def get_loop_info(filename: str):
    f = open(filename, 'r')
    f.readline()
    lines = f.readlines()

    all_info = []

    for line in lines:
        line = line.strip()
        line = line.split()
        line = [np.double(i) for i in line]
        all_info.append(line)

    return all_info


def plot_loop_info(ax: plt, traj: trajectory.PosePath3D, loopinfo):
    for info in loopinfo:
        stamp1 = info[1]
        indx = np.abs(traj.timestamps - stamp1).argmin()
        assert math.isclose(
            traj.timestamps[indx], stamp1, rel_tol=1e-4)
        traj_xy1 = traj.positions_xyz[indx][:2]

        stamp2 = info[3]
        indx = np.abs(traj.timestamps - stamp2).argmin()
        assert math.isclose(
            traj.timestamps[indx], stamp2, rel_tol=1e-4)
        traj_xy2 = traj.positions_xyz[indx][:2]

        ax.plot([traj_xy1[0], traj_xy2[0]], [
                traj_xy1[1], traj_xy2[1]], '-', color='black')


if __name__ == '__main__':
    folder = '/home/bjoshi/code/slamutils-python/shipwreck_pcl'
    title = r'$\boldsymbol{shipwreck\_lawnmower}$'

    pcd = open3d.io.read_point_cloud(
        os.path.join(folder, 'shipwreck_pcl.ply'))
    pcd_xyz = np.asarray(pcd.points)

    fig = plt.figure(figsize=(8, 8))

    ax = prepare_axis(fig, plot.PlotMode.xy)
    ax.set_title(title, fontsize=16)

    combined_traj = file_interface.read_tum_trajectory_file(
        os.path.join(folder, 'combined_traj.txt'))

    svin_traj = file_interface.read_tum_trajectory_file(
        os.path.join(folder, 'svin_traj_lc.txt'))

    prim_traj = file_interface.read_tum_trajectory_file(
        os.path.join(folder, 'prim_traj.txt'))

    # svin_x = svin_traj.positions_xyz[:, 0]
    # svin_y = svin_traj.positions_xyz[:, 1]
    # svin_plot, = ax.plot(svin_x, svin_y, '-.', color='red')

    # prim_x = prim_traj.positions_xyz[:, 0]
    # prim_y = prim_traj.positions_xyz[:, 1]
    # prim_plot, = ax.plot(prim_x, prim_y, '-.', color='blue')

    # plot_traj(ax, prim_traj, plot_mode=plot.PlotMode.xy,
    #           style='-', color='blue', label=r'$\bf{primitive}$')

    switch_info = read_switch_info(os.path.join(folder, 'switch_info.txt'))
    prev_stamp = 0.0
    switch_times_x = []
    switch_times_y = []
    for info in switch_info:
        uber_copy = copy.deepcopy(combined_traj)
        uber_copy.reduce_to_time_range(prev_stamp, info[3])

        x = uber_copy.positions_xyz[:, 0]
        y = uber_copy.positions_xyz[:, 1]

        color = 'red' if info[0] == 0.0 else 'blue'
        ax.plot(x, y, '-', color=color, linewidth=2)
        uber_indx = np.abs(combined_traj.timestamps - info[3]).argmin()
        uber_xy = combined_traj.positions_xyz[uber_indx][:2]
        switch_times_x.append(uber_xy[0])
        switch_times_y.append(uber_xy[1])

        prev_stamp = info[3]

    uber_copy = copy.deepcopy(combined_traj)
    uber_copy.reduce_to_time_range(prev_stamp, uber_copy.timestamps[-1])
    x = uber_copy.positions_xyz[:, 0]
    y = uber_copy.positions_xyz[:, 1]
    ax.plot(x, y, '-', color='red')

    switch_plot,  = ax.plot(switch_times_x, switch_times_y, 'D',
                            color='green', markersize=8)

    pcl_x = pcd_xyz[:, 0]
    pcl_y = pcd_xyz[:, 1]

    scatter_plot = plt.scatter(pcl_x, pcl_y, s=1, color='dimgray',
                               label=r'$\bf{reconstruction}$')

    # make proxy artists
    # make list of one line -- doesn't matter what the coordinates are
    line = [[(0, 0)]]
    # set up the proxy artist
    lc = mcol.LineCollection(
        2 * line, linestyles=['-', '-'], colors=['red', 'blue'])
    # create the legend

    # ax.legend([svin_plot, prim_plot, lc, switch_plot], [r'$\bf{svin}$', r'$\bf{primitive}$',
    #                                                     r'$\bf{robust}$', r'$\bf{switch\_times}$'],
    #           handler_map={type(lc): HandlerDashedLines()},
    #           frameon=True, prop={'size': 16})

    ax.legend([lc, switch_plot, scatter_plot], [r'$\bf{primitive}$', r'$\bf{switch\_times}$', r'$\bf{reconstruction}$'],
              handler_map={type(lc): HandlerDashedLines()},
              frameon=True, prop={'size': 16})
    plt.savefig('combined_with_wreck.png', bbox_inches='tight')
    plt.show()
