#!/usr/bin/env python

from re import L
from types import FrameType
from numpy.lib.function_base import percentile
from evo.core import transformations as tr
from evo.core import lie_algebra as lie
from evo.tools import file_interface, log, plot
from evo.core import sync, metrics, trajectory

import os
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from tqdm import tqdm


sns.set(style='white', font='sans-serif',
        font_scale=1.2, color_codes=False)
rc = {
    "lines.linewidth": 1.0,
    "text.usetex": True,
    "font.family": 'sans-serif',
    "pgf.texsystem": 'pdflatex',
    'text.latex.preamble': r'\usepackage{bm} \usepackage{amsmath}'
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


def get_tags_position(tag_file: str) -> np.ndarray:
    file = open(tag_file, 'r')
    tags = file.readlines()

    tag_positions = []
    for tag in tags:
        tag = tag.strip()
        tag_pose = tag.split(' ')
        id = int(tag_pose[0])
        pose = [float(x) for x in tag_pose[1:]]
        # Not using orientation
        # quat = [pose[6], pose[3], pose[4], pose[5]]
        # t_ct = tr.quaternion_matrix(quat)
        trans = np.array([id, pose[0], pose[1], pose[2], pose[3]])
        tag_positions.append(trans)

    file.close()
    return np.array(tag_positions)


def get_interpolated_pose(stamp: float, traj: trajectory.PoseTrajectory3D) -> lie.se3:

    closest_indx = np.argmin(np.absolute(traj.timestamps - stamp))
    closest_stamp = traj.timestamps[closest_indx]
    # assert(abs(closest_stamp - stamp) < 0.51)

    lower_indx = None
    upper_indx = None
    exact_match = False
    if closest_stamp > stamp:
        lower_indx = closest_indx - 1
        upper_indx = closest_indx
    elif closest_stamp < stamp:
        lower_indx = closest_indx
        upper_indx = closest_indx + 1
    else:
        exact_match = True

    if exact_match:
        trans = traj.positions_xyz[closest_indx]
        quat = traj.orientations_quat_wxyz[closest_indx]
    else:
        lower_stamp = traj.timestamps[lower_indx]
        upper_stamp = traj.timestamps[upper_indx]
        lower_position = traj.positions_xyz[lower_indx]
        upper_position = traj.positions_xyz[upper_indx]
        trans = lower_position + (upper_position - lower_position) * \
            (stamp - lower_stamp) / (upper_stamp - lower_stamp)
        quat = tr.quaternion_slerp(traj.orientations_quat_wxyz[lower_indx],
                                   traj.orientations_quat_wxyz[upper_indx], (stamp - lower_stamp)/(upper_stamp - lower_stamp))

    rotm = tr.quaternion_matrix(quat)[:3, :3]
    result = lie.se3(rotm, trans)
    return result


if __name__ == '__main__':

    cam_to_gt = tr.quaternion_matrix(np.array([-0.5, 0.5, -0.5, 0.5]))
    gt_to_cam = np.linalg.inv(cam_to_gt)

    colmap_scale = 1.398555225

    # Load the trajectory
    dataset_name = 'tags'
    root_path = '/home/bjoshi/results'
    display_plot = False

    title = r'g1\_cavern2'
    dataset_path = os.path.join(root_path, dataset_name)

    pkgs = ['svin2', 'orbslam3', 'vinsmono', 'openvins']

    colors = {'colmap': 'k', 'openvins': 'b', 'orbslam3': 'g',
              'svin2': 'r', 'vinsmono': 'gold'}
    disp_names = {'colmap': 'COLMAP', 'openvins': 'OpenVINS',
                  'orbslam3': 'ORB-SLAM3', 'svin2': 'SVIn2', 'vinsmono': 'VINS-Mono'}

    colmap = file_interface.read_tum_trajectory_file(
        os.path.join(dataset_path, 'colmap.txt'))

    trajs = {}
    for pkg in pkgs:
        filename = os.path.join(dataset_path, pkg + '.txt')
        trajs[pkg] = file_interface.read_tum_trajectory_file(filename)

    # Align the initial colmap pose to svin [gravity]
    colmap_sync, svin_sync = sync.associate_trajectories(
        colmap, trajs['svin2'])
    tf = colmap_sync.align_origin(svin_sync)
    colmap.scale(colmap_scale)
    colmap.transform(tf)
    # colmap.transform(gt_to_cam)

    transforms = {}
    for name, traj in trajs.items():

        traj_sync, colmap_sync = sync.associate_trajectories(traj, colmap)
        r, t, s = traj_sync.align(colmap_sync, correct_scale=True)
        transforms[name] = [r, t, s]

    # read the tags
    tags = get_tags_position(os.path.join(
        dataset_path, 'tags_cavern2_final.txt'))

    fig = plt.figure(figsize=(8, 8))
    ax = prepare_axis(fig, plot_mode=plot.PlotMode.xy)
    ax.set_title(title, fontsize=16)
    plot_traj(
        ax, colmap, plot_mode=plot.PlotMode.xy, color=colors['colmap'])

    tag_trajs = {}
    tag_trajs['colmap'] = colmap
    for pkg in pkgs:
        filename = os.path.join(dataset_path, pkg + '_tags.txt')
        traj = file_interface.read_tum_trajectory_file(filename)
        r, t, s = transforms[pkg]
        traj.scale(s)
        traj.transform(lie.se3(r, t))
        tag_trajs[pkg] = traj
        # plot_traj(
        #     ax, traj, plot_mode=plot.PlotMode.xy, color=colors[pkg], label=disp_names[pkg])

    prev_tag = -1

    global_locs = {'colmap': [],
                   'openvins': [],
                   'orbslam3': [],
                   'svin2': [],
                   'vinsmono': []}

    for tag in tqdm(tags):

        stamp = tag[1]
        tag_position = np.array(
            [tag[2], tag[3], tag[4], 1.0]).reshape(4, 1)
        for name, cur_traj in tag_trajs.items():
            traj_pose = get_interpolated_pose(stamp, cur_traj)
            global_pos = np.dot(traj_pose, tag_position)
            # print(global_pos)

            global_locs[name].append(
                [tag[0], global_pos[0], global_pos[1], global_pos[2]])
            # ax.scatter(global_pos[0], global_pos[1],
            #            color=colors[name], s=5, label=disp_names[name])

    for name, locs in global_locs.items():
        locs = np.array(global_locs[name]).reshape(-1, 4)
        filename = os.path.join(dataset_path, name + '_tag_locs.txt')
        np.savetxt(filename, locs, '%3.6f')
        x = locs[:, 1]
        y = locs[:, 2]
        ax.scatter(x, y,
                   color=colors[name], s=10, label=disp_names[name])

    ax.legend(frameon=True)

    if display_plot:
        plt.show()
    else:
        plt.savefig(os.path.join(dataset_path, dataset_name+'.png'),
                    bbox_inches='tight')
