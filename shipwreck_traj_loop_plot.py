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
import math

log.configure_logging(verbose=True)

sns.set(style='white', font='sans-serif',
        font_scale=1.5, color_codes=False)
rc = {
    "lines.linewidth": 1.5,
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

    xlabel = "$x$ (m)"
    ylabel = "$y$ (m)"
    ax.set_xlabel(xlabel, fontsize=16)
    ax.set_ylabel(ylabel, fontsize=16)
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

    folder = '/home/bjoshi/code/slamutils-python/shipwreck'
    title = r'shipwreck\_lawnmower'

    svin_traj = file_interface.read_tum_trajectory_file(
        os.path.join(folder, 'svin_traj.txt'))
    prim_traj = file_interface.read_tum_trajectory_file(
        os.path.join(folder, 'prim_traj.txt'))
    uber_traj = file_interface.read_tum_trajectory_file(
        os.path.join(folder, 'uber_traj_no_loop.txt'))

    final_traj = file_interface.read_tum_trajectory_file(
        os.path.join(folder, 'uber_traj_loop.txt'))

    fig = plt.figure(figsize=(8, 8))

    ax = prepare_axis(fig, plot.PlotMode.xy)
    ax.set_title(title, fontsize=16)

    plot_traj(ax, svin_traj, plot_mode=plot.PlotMode.xy,
              style='-', color='red', label='svin')
    plot_traj(ax, prim_traj, plot_mode=plot.PlotMode.xy,
              style='-', color='blue', label='primitive')
    # plot_traj(ax, uber_traj, plot_mode=plot.PlotMode.xy,
    #           style='-', color='green', label='combined')
    plot_traj(ax, final_traj, plot_mode=plot.PlotMode.xy,
              style='-', color='green', label='combined\_lc')

    loop_info_svin = get_loop_info(os.path.join(folder, 'loop_info_svin.txt'))
    plot_loop_info(ax, svin_traj, loop_info_svin)

    loop_info_uber = get_loop_info(os.path.join(folder, 'loop_info_uber.txt'))
    plot_loop_info(ax, final_traj, loop_info_uber)

    plt.savefig('shipwreck_lawnmower.png', bbox_inches='tight')
    plt.show()
