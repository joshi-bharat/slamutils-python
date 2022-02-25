#!/usr/bin/env python

from cProfile import label
from pprint import pprint
import os
import matplotlib
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from evo.core import metrics, trajectory
from evo.tools import file_interface
from evo.tools import log, plot
import copy
import matplotlib.collections as mcol
from matplotlib.legend_handler import HandlerLineCollection, HandlerTuple
from matplotlib.lines import Line2D


class HandlerDashedLines(HandlerLineCollection):
    """
    Custom Handler for LineCollection instances.
    """

    def create_artists(self, legend, orig_handle,
                       xdescent, ydescent, width, height, fontsize, trans):
        # figure out how many lines there are
        numlines = 2
        xdata, xdata_marker = self.get_xdata(legend, xdescent, ydescent,
                                             width, height, fontsize)
        leglines = []
        # divide the vertical space where the lines will go
        # into equal parts based on the number of lines
        ydata = np.full_like(xdata, height / (numlines + 1))
        # for each line, create the line at the proper location
        # and set the dash pattern
        for i in range(numlines):
            legline = Line2D(xdata, ydata * (numlines - i) - ydescent)
            self.update_prop(legline, orig_handle, legend)
            # set color, dash pattern, and linewidth to that
            # of the lines in linecollection
            try:
                color = orig_handle.get_colors()[i]
            except IndexError:
                color = orig_handle.get_colors()[0]
            try:
                dashes = orig_handle.get_dashes()[i]
            except IndexError:
                dashes = orig_handle.get_dashes()[0]
            try:
                lw = orig_handle.get_linewidths()[i]
            except IndexError:
                lw = orig_handle.get_linewidths()[0]
            if dashes[1] is not None:
                legline.set_dashes(dashes[1])
            legline.set_color(color)
            legline.set_transform(trans)
            legline.set_linewidth(lw)
            leglines.append(legline)
        return leglines


log.configure_logging(verbose=True)
# matplotlib.use("pgf")

sns.set(style='white', font='sans-serif',
        font_scale=1.2, color_codes=False)
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

    folder = '/home/bjoshi/ros_workspaces/svin_ws/src/SVIn/pose_graph/output_logs'
    title = r'shipwreck\_lawnmower'

    svin_traj = file_interface.read_tum_trajectory_file(
        os.path.join(folder, 'svin_traj_lc.txt'))
    prim_traj = file_interface.read_tum_trajectory_file(
        os.path.join(folder, 'prim_traj.txt'))
    uber_traj = file_interface.read_tum_trajectory_file(
        os.path.join(folder, 'uber_traj.txt'))

    fig = plt.figure(figsize=(8, 8))

    ax = prepare_axis(fig, plot.PlotMode.xy)
    ax.set_title(title, fontsize=16)

    x = prim_traj.positions_xyz[:, 0]
    y = prim_traj.positions_xyz[:, 1]
    prim_plot, = ax.plot(x, y, '-', color='blue')

    x = svin_traj.positions_xyz[:, 0]
    y = svin_traj.positions_xyz[:, 1]
    svin_plot, = ax.plot(x, y, '-', color='red')
    # plot_traj(ax, prim_traj, color='blue',
    #           label='prim\_estimator', plot_mode=plot.PlotMode.xy)
    # plot_traj(ax, svin_traj, color='red',
    #           label='svin\_estimator', plot_mode=plot.PlotMode.xy)
    # plot_traj(ax, uber_traj, style='-', color='green',
    #           label='uber\_estimator', plot_mode=plot.PlotMode.xy)

    switch_info = read_switch_info(os.path.join(folder, 'switch_info.txt'))

    # for info in switch_info:

    #     svin_stamp = info[1]
    #     svin_indx = np.abs(svin_traj.timestamps - svin_stamp).argmin()
    #     assert math.isclose(
    #         svin_traj.timestamps[svin_indx], svin_stamp, rel_tol=1e-4)
    #     svin_xy = svin_traj.positions_xyz[svin_indx][:2]

    #     uber_stamp = info[3]
    #     uber_indx = np.abs(uber_traj.timestamps - uber_stamp).argmin()
    #     assert math.isclose(
    #         uber_traj.timestamps[uber_indx], uber_stamp, rel_tol=1e-4)
    #     uber_xy = uber_traj.positions_xyz[uber_indx][:2]

    # ax.plot([svin_xy[0], uber_xy[0]], [
    #         svin_xy[1], uber_xy[1]], color='red')

    prev_stamp = 0.0

    switch_times_x = []
    switch_times_y = []

    for info in switch_info:
        uber_copy = copy.deepcopy(uber_traj)
        uber_copy.reduce_to_time_range(prev_stamp, info[3])

        x = uber_copy.positions_xyz[:, 0]
        y = uber_copy.positions_xyz[:, 1]

        color = 'red' if info[0] == 0.0 else 'blue'
        ax.plot(x, y, '-.', color=color)

        uber_indx = np.abs(uber_traj.timestamps - info[3]).argmin()
        uber_xy = uber_traj.positions_xyz[uber_indx][:2]
        switch_times_x.append(uber_xy[0])
        switch_times_y.append(uber_xy[1])

        prev_stamp = info[3]

    uber_copy = copy.deepcopy(uber_traj)
    uber_copy.reduce_to_time_range(prev_stamp, uber_copy.timestamps[-1])
    x = uber_copy.positions_xyz[:, 0]
    y = uber_copy.positions_xyz[:, 1]
    ax.plot(x, y, '-.', color='red')

    switch_plot,  = ax.plot(switch_times_x, switch_times_y, 'D',
                            color='black')
    # make proxy artists
    # make list of one line -- doesn't matter what the coordinates are
    line = [[(0, 0)]]
    # set up the proxy artist
    lc = mcol.LineCollection(
        2 * line, linestyles=['-.', '-.'], colors=['red', 'blue'])
    # create the legend

    ax.legend([prim_plot, svin_plot, lc, switch_plot], ['primitive\_traj', 'svin\_traj', 'combined\_traj', 'switch\_time'], handler_map={type(lc): HandlerDashedLines()},
              frameon=True, prop={'size': 16})

    plt.savefig('shipwreck_lawnmower_switch.png', bbox_inches='tight')

    # plt.show()
