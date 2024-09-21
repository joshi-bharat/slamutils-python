#!/usr/bin/env python

import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

from evo.core import trajectory
from evo.tools import log, plot

log.configure_logging(verbose=True)


sns.set(style="white", font="sans-serif", font_scale=1.2, color_codes=False)
rc = {
    "lines.linewidth": 1.2,
    "text.usetex": True,
    "font.family": "sans-serif",
    "pgf.texsystem": "pdflatex",
    "text.latex.preamble": r"\usepackage{amsmath} \usepackage{amssymb}",
}

matplotlib.rcParams.update(rc)


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

    plot_radius = max(
        [
            abs(lim - mean_)
            for lims, mean_ in ((xlim, xmean), (ylim, ymean), (zlim, zmean))
            for lim in lims
        ]
    )

    ax.set_xlim3d([xmean - plot_radius, xmean + plot_radius])
    ax.set_ylim3d([ymean - plot_radius, ymean + plot_radius])
    ax.set_zlim3d([zmean - plot_radius, zmean + plot_radius])


def prepare_axis(
    fig: plt.Figure, plot_mode: plot.PlotMode = plot.PlotMode.xy, subplot_arg: int = 111
) -> plt.Axes:
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
        ax.set_zlabel("$z$ (m)")
    return ax


def plot_traj(
    ax: plt.Axes,
    traj: trajectory.PosePath3D,
    plot_mode: plot.PlotMode = plot.PlotMode.xy,
    style: str = "-",
    color: str = "black",
    label: str = "",
    alpha: float = 1.0,
) -> None:
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
