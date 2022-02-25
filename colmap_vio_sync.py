#!/usr/bin/python

from gettext import translation
import open3d
from evo.tools.settings import SETTINGS
import numpy as np
import pprint
from evo.core import metrics, sync
from evo.tools import log, plot, file_interface
from matplotlib import pyplot as plt
import os
import copy
log.configure_logging(verbose=True, debug=True, silent=False)

# temporarily override some package settings
SETTINGS.plot_usetex = False

if __name__ == '__main__':
    folder = '/home/bjoshi/mexico'
    svin_file = os.path.join(folder, 'svin_traj_camera.txt')
    colmap_file = os.path.join(folder, 'colmap_traj_interpolated.txt')

    svin_traj = file_interface.read_tum_trajectory_file(svin_file)
    colmap_traj = file_interface.read_tum_trajectory_file(colmap_file)

    max_diff = 0.01
    svin_traj, colmap_traj = sync.associate_trajectories(
        svin_traj, colmap_traj, max_diff)

    colmap_aligned_traj = copy.deepcopy(colmap_traj)
    rot, translation, scale = colmap_aligned_traj.align(
        svin_traj, correct_scale=True, correct_only_scale=False)
    transform_4x4 = np.identity(4)
    transform_4x4[:3, :3] = rot
    transform_4x4[:3, 3] = translation

    fig = plt.figure()
    traj_by_label = {
        "colmap (not aligned)": colmap_traj,
        "colmap (aligned)": colmap_aligned_traj,
        "svin": svin_traj
    }

    plot.trajectories(fig, traj_by_label, plot.PlotMode.xyz)
    plt.show()

    colmap_pcd = open3d.io.read_point_cloud(
        os.path.join(folder, 'colmap_pcd.ply'))

    colmap_pcd.scale(scale, np.array([0, 0, 0]))
    colmap_pcd.transform(transform_4x4)

    svin_pcd = open3d.io.read_point_cloud(
        os.path.join(folder, 'svin_traj_camera.ply'))

    open3d.visualization.draw_geometries([colmap_pcd, svin_pcd])

    svin_pcd_array = np.asarray(svin_pcd.points)
    colmap_pcd_array = np.asarray(colmap_pcd.points)
    svin_colors = np.asarray(svin_pcd.colors)
    colmap_colors = np.asarray(colmap_pcd.colors)

    combined_points = np.concatenate(
        (svin_pcd_array, colmap_pcd_array), axis=0)
    combined_colors = np.concatenate((svin_colors, colmap_colors), axis=0)

    combined_pcd = open3d.geometry.PointCloud()
    combined_pcd.points = open3d.utility.Vector3dVector(combined_points)
    combined_pcd.colors = open3d.utility.Vector3dVector(combined_colors)

    open3d.io.write_point_cloud(
        'svin_traj_with_colmap_pointcloud.ply', combined_pcd)
