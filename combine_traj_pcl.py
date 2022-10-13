#!/usr/bin/python

from unittest import result
import open3d
import numpy as np

if __name__ == "__main__":

    traj_file = "/home/bjoshi/ros_workspaces/svin_ws/src/SVIn/pose_graph/svin_results/svin_2022_10_13_16_39_53.txt"
    pcl_file = "/home/bjoshi/ros_workspaces/svin_ws/src/SVIn/pose_graph/reconstruction_results/pointcloud.ply"
    result_file = "/home/bjoshi/ros_workspaces/svin_ws/src/SVIn/pose_graph/reconstruction_results/reconstruction.ply"

    with open(traj_file, "r") as f:
        lines = f.readlines()
        lines = [line.strip() for line in lines]
        lines = [line.split(" ") for line in lines]
        lines = [[float(x) for x in line] for line in lines]

    svin_traj = np.array(lines)
    svin_traj_points = svin_traj[:, 1:4]

    pcd = open3d.geometry.PointCloud()
    svin_points = svin_traj_points
    red_color = [1, 0, 0]
    svin_colors = np.array([red_color] * len(svin_traj_points))

    sparse_pcd = open3d.io.read_point_cloud(pcl_file)
    sparse_points = np.asarray(sparse_pcd.points)
    sparse_colors = np.asarray(sparse_pcd.colors)

    combined_points = np.concatenate((svin_points, sparse_points), axis=0)
    combined_colors = np.concatenate((svin_colors, sparse_colors), axis=0)

    pcd.points = open3d.utility.Vector3dVector(combined_points)
    pcd.colors = open3d.utility.Vector3dVector(combined_colors)
    open3d.visualization.draw_geometries([pcd])

    open3d.io.write_point_cloud(result_file, pcd)
