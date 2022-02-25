#!/usr/bin/env python

import os
from webbrowser import get
import matplotlib
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from evo.core import metrics, trajectory, sync
from evo.tools import file_interface
from evo.tools import log, plot
import open3d
import copy
from multicolor_line import HandlerDashedLines
import matplotlib.collections as mcol
from matplotlib.lines import Line2D
import math
import pprint


log.configure_logging(verbose=True)
# matplotlib.use("pgf")

sns.set(style='white', font='sans-serif',
        font_scale=1.5, color_codes=False)
rc = {
    "lines.linewidth": 1.2,
    "text.usetex": True,
    "font.family": 'sans-serif',
    "pgf.texsystem": 'pdflatex',
    'text.latex.preamble': r'\usepackage{amsmath} \usepackage{amssymb}'
}

matplotlib.rcParams.update(rc)

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
    ax.set_xlabel(xlabel, fontsize=14)
    ax.set_ylabel(ylabel, fontsize=14)
    if plot_mode == plot.PlotMode.xyz:
        ax.set_zlabel('$z$ (m)')
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


if __name__ == '__main__':

    folder = '/home/bjoshi/code/slamutils-python/reef_lawnmower'
    title = r'$\boldsymbol{reef\_lawnmower}$'

    # svin_traj = file_interface.read_tum_trajectory_file(
    # os.path.join(folder, 'svin_with_clahe.txt'))
    prim_traj = file_interface.read_tum_trajectory_file(
        os.path.join(folder, 'prim_traj.txt'))
    robust_traj = file_interface.read_tum_trajectory_file(
        os.path.join(folder, 'robust_traj.txt'))
    svin_traj = file_interface.read_tum_trajectory_file(
        os.path.join(folder, 'svin_traj.txt'))
    svin_gt_traj = file_interface.read_tum_trajectory_file(
        os.path.join(folder, 'svin_gt.txt'))

    fig = plt.figure(figsize=(8, 8))

    ax = prepare_axis(fig, plot.PlotMode.xy)
    ax.set_title(title, fontsize=16)

    ax.set_xlim([-20, 2.5])
    ax.set_ylim([-6, 16.5])

    # plot_traj(ax, svin_gt_traj, color='black', label='Ground Truth')
    # plot_traj(ax, robust_traj, color='red', label='Robust'))

    # prim_traj.scale(0.5)
    x = prim_traj.positions_xyz[:, 0]
    y = prim_traj.positions_xyz[:, 1]
    # z = prim_traj.positions_xyz[:, 2]
    prim_plot, = ax.plot(x, y, '-.', color='blue')
    # set_aspect_equal_3d(ax)

    x = svin_traj.positions_xyz[:, 0]
    y = svin_traj.positions_xyz[:, 1]
    z = robust_traj.positions_xyz[:, 2]
    svin_plot, = ax.plot(x, y, '-.', color='red')
    # set_aspect_equal_3d(ax)

    svin_gt_traj.align_origin(robust_traj)
    x = svin_gt_traj.positions_xyz[:, 0]
    y = svin_gt_traj.positions_xyz[:, 1]
    svin_gt_plot, = ax.plot(x, y, '-', color='black')

    switch_info = read_switch_info(os.path.join(folder, 'switch_info.txt'))

    prev_stamp = 0.0

    switch_times_x = []
    switch_times_y = []

    for info in switch_info:
        robust_copy = copy.deepcopy(robust_traj)
        robust_copy.reduce_to_time_range(prev_stamp, info[3])

        x = robust_copy.positions_xyz[:, 0]
        y = robust_copy.positions_xyz[:, 1]

        color = 'red' if info[0] == 0.0 else 'blue'
        ax.plot(x, y, '-', color=color, linewidth=2)

        robust_indx = np.abs(robust_traj.timestamps - info[3]).argmin()
        robust_xy = robust_traj.positions_xyz[robust_indx][:2]
        switch_times_x.append(robust_xy[0])
        switch_times_y.append(robust_xy[1])

        prev_stamp = info[3]

    robust_copy = copy.deepcopy(robust_traj)
    robust_copy.reduce_to_time_range(prev_stamp, robust_copy.timestamps[-1])
    x = robust_copy.positions_xyz[:, 0]
    y = robust_copy.positions_xyz[:, 1]
    ax.plot(x, y, '-', color='red', linewidth=2)

    switch_plot,  = ax.plot(switch_times_x, switch_times_y, 'D',
                            color='green', markersize=8)

    # # make proxy artists
    # # make list of one line -- doesn't matter what the coordinates are
    line = [[(0, 0)]]
    # # set up the proxy artist
    lc = mcol.LineCollection(
        2 * line, linestyles=['-', '-'], colors=['red', 'blue'])
    # # create the legend

    ax.legend([svin_gt_plot, svin_plot, prim_plot, lc, switch_plot], [r'$\bf{svin\_gt}$', r'$\bf{svin}$', r'$\bf{primitive}$',
                                                                      r'$\bf{robust}$', r'$\bf{switch\_times}$'],
              handler_map={type(lc): HandlerDashedLines()},
              frameon=True, prop={'size': 16}, loc='upper right')

    # ax.legend([svin_gt_plot, prim_plot, lc, switch_plot], [r'$\bf{svin\_gt}$', r'$\bf{primitive}$',
    #                                                        r'$\bf{robust}$', r'$\bf{switch\_times}$'],
    #           handler_map={type(lc): HandlerDashedLines()},
    #           frameon=True, prop={'size': 16})

    plt.savefig('coral_reef_lawnmower.png', bbox_inches='tight')

    plt.show()

    # svin_gt_traj, robust_traj = sync.associate_trajectories(
    #     svin_gt_traj, robust_traj, 0.05)

    # print(svin_gt_traj)

    # pose_relation = metrics.PoseRelation.translation_part
    # data = (svin_gt_traj, robust_traj)
    # ape_metric = metrics.APE(pose_relation)
    # ape_metric.process_data(data)
    # ape_stats = ape_metric.get_all_statistics()
    # pprint.pprint(ape_stats)
