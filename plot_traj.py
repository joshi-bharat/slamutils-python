#!/usr/bin/env python

from pprint import pprint
import os
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

from evo.core import sync, metrics, trajectory
from evo.tools import file_interface
from evo.tools import log, plot

log.configure_logging(verbose=True)


sns.set(style='white', font='sans-serif',
        font_scale=1.2, color_codes=False)
rc = {
    "lines.linewidth": 1.0,
    "text.usetex": True,
    "font.family": 'sans-serif',
    "pgf.texsystem": 'pdflatex',
    'text.latex.preamble': [r'\usepackage{amsmath}', r'\usepackage{amssymb}']
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


if __name__ == '__main__':
    # Load the trajectory
    dataset_name = 'openwater'
    root_path = '/home/bjoshi/results'
    display_plot = False
    show_legend = False
    title = r'g1\_openwater'
    dataset_path = os.path.join(root_path, dataset_name)

    # pkgs = ['openvins', 'orbslam3', 'svin2', 'vinsmono']
    pkgs = ['svin2', 'vinsmono']
    colors = {'colmap': 'k', 'openvins': 'b', 'orbslam3': 'g',
              'svin2': 'r', 'vinsmono': 'gold'}
    disp_names = {'colmap': 'COLMAP', 'openvins': 'OpenVINS',
                  'orbslam3': 'ORB-SLAM3', 'svin2': 'SVIn2', 'vinsmono': 'VINS-Mono'}
    tracking = {}
    rmse_errors = {}

    colmap = file_interface.read_tum_trajectory_file(
        os.path.join(dataset_path, 'colmap.txt'))
    colmap_path_length = colmap.path_length

    trajs = {}
    for pkg in pkgs:
        filename = os.path.join(dataset_path, pkg + '.txt')
        trajs[pkg] = file_interface.read_tum_trajectory_file(filename)

    # Align the initial colmap pose to svin [gravity]
    colmap_sync, svin_sync = sync.associate_trajectories(
        colmap, trajs['svin2'])
    tf = colmap_sync.align_origin(svin_sync)

    colmap.transform(tf)

    ape_metric = metrics.APE(pose_relation)

    fig = plt.figure(figsize=(8, 8))
    ax = prepare_axis(fig)
    ax.set_title(title, fontsize=16)
    if show_legend:
        plot_traj(
            ax, colmap, color=colors['colmap'], label=disp_names['colmap'])
    else:
        plot_traj(ax, colmap, color=colors['colmap'])

    total_scale = 0.0
    colmap_poses = colmap.num_poses
    for name, traj in trajs.items():
        print("******** Alignment for: {} ***************".format(name))
        print(traj)
        traj_sync, colmap_sync = sync.associate_trajectories(traj, colmap)
        tracking[name] = traj_sync.num_poses * 100.0 / colmap_poses
        r, t, s = traj_sync.align(colmap_sync, correct_scale=True)
        total_scale += s
        print("scale: {}".format(s))
        ape_metric.process_data((traj_sync, colmap_sync))
        ape_stats = ape_metric.get_all_statistics()
        pprint(ape_stats)
        rmse_errors[name] = ape_stats['rmse']
        if show_legend:
            plot_traj(ax, traj_sync,
                      color=colors[name], label=disp_names[name])
        else:
            plot_traj(ax, traj_sync,
                      color=colors[name])

    avg_scale = total_scale / len(trajs)
    print("Average scale: {}".format(avg_scale))
    colmap_path_length = colmap_path_length / avg_scale
    print('Colmap path length: {:.2f}'.format(colmap_path_length))

    for name, rmse in rmse_errors.items():
        print('RMSE: {} -- {:.3f}'.format(name, rmse))

    for name, track in tracking.items():
        print('{} tracks -- {}%'.format(name, round(track)))

    if display_plot:
        plt.show()
    else:
        plt.savefig(os.path.join(dataset_path, dataset_name+'.png'),
                    bbox_inches='tight')
