#!/usr/bin/python3

import glob
import os

import numpy as np
import open3d
import pandas as pd
from evo.core import trajectory
from evo.tools import plot
from matplotlib import pyplot as plt
from plot_traj import plot_traj, prepare_axis
from typing import List, Dict, Optional
from gtsam import Pose3
import copy
from colmap_utils import (
    compute_baseline,
    display_stereo_depths,
    read_trajectory,
    read_array,
    write_array,
    write_evo_traj,
    write_gtsam_poses,
)


def get_extrinsics(
    left_poses: Dict[str, Pose3], right_poses: Dict[str, Pose3]
) -> List[np.array]:
    baseline = []
    quats = []
    for stamp, lpose in left_poses.items():
        if not stamp in right_poses:
            print("No right image for: {}".format(stamp))
            continue
        else:
            rpose = right_poses[stamp]
        diff = rpose.between(lpose)
        trans = diff.translation()
        quat = diff.rotation().quaternion()
        quats.append(quat)
        baseline.append([trans[0], trans[1], trans[2]])

    trans = np.array(baseline).astype(np.float64)
    quats = np.array(quats).astype(np.float64)

    return trans, quats


def create_trajectory(
    left_poses: Dict[str, Pose3], right_poses: Dict[str, Pose3]
) -> List:
    left_camera_poses = []
    right_camera_poses = []
    original_stamps = []
    for stamp, lpose in sorted(left_poses.items()):
        if stamp in right_poses:
            trans = lpose.translation()
            quat = lpose.rotation().quaternion()  # qw qx qy qz
            original_stamps.append(stamp)
            float_stamp = stamp[:-9] + "." + stamp[-9:]

            data = [
                float(float_stamp),
                trans[0],
                trans[1],
                trans[2],
                quat[0],
                quat[1],
                quat[2],
                quat[3],
            ]
            left_camera_poses.append(data)

            trans = right_poses[stamp].translation()
            quat = right_poses[stamp].rotation().quaternion()
            data = [
                float(float_stamp),
                trans[0],
                trans[1],
                trans[2],
                quat[0],
                quat[1],
                quat[2],
                quat[3],
            ]
            right_camera_poses.append(data)

    left_df = pd.DataFrame(
        left_camera_poses, columns=["stamp", "tx", "ty", "tz", "qw", "qx", "qy", "qz"]
    )
    right_df = pd.DataFrame(
        right_camera_poses, columns=["stamp", "tx", "ty", "tz", "qw", "qx", "qy", "qz"]
    )

    trans = left_df[["tx", "ty", "tz"]].to_numpy()
    stamps = left_df["stamp"].to_numpy()
    quat = left_df[["qw", "qx", "qy", "qz"]].to_numpy()

    left_traj = trajectory.PoseTrajectory3D(trans, quat, stamps)

    trans = right_df[["tx", "ty", "tz"]].to_numpy()
    stamps = right_df["stamp"].to_numpy()
    quat = right_df[["qw", "qx", "qy", "qz"]].to_numpy()

    right_traj = trajectory.PoseTrajectory3D(trans, quat, stamps)
    return left_traj, right_traj, original_stamps


def read_depth_maps(
    folder: str, type: Optional[str] = "geometric"
) -> List[Dict[str, np.array]]:

    left_files = {}
    right_files = {}

    if type == "photometric":
        left_files = glob.glob(os.path.join(folder, "left" "*.photometric.bin"))
        right_files = glob.glob(os.path.join(folder, "right" "*.photometric.bin"))
        print(f"Reading {len(left_files)}photometric depth maps from {folder}")

    else:
        left_files = glob.glob(os.path.join(folder, "left", "*.geometric.bin"))
        right_files = glob.glob(os.path.join(folder, "right", "*.geometric.bin"))
        print(f"Reading {len(right_files)} geometric depth maps from {folder}")

    left_files.sort()
    right_files.sort()

    left_depths = {}
    right_depths = {}

    for left_file in left_files:
        depth_map = read_array(left_file)
        filename = os.path.split(left_file)[-1]
        left_depths[filename] = depth_map

    for right_file in right_files:
        depth_map = read_array(right_file)
        filename = os.path.split(right_file)[-1]
        right_depths[filename] = depth_map

    return left_depths, right_depths


def write_stereo_images(
    base_dir: str, left_depths: Dict[str, np.array], right_depths: Dict[str, np.array]
):

    left_dir = os.path.join(base_dir, "left")
    if not os.path.exists(left_dir):
        os.makedirs(left_dir)
    right_dir = os.path.join(base_dir, "right")
    if not os.path.exists(right_dir):
        os.makedirs(right_dir)

    for bare_name, depth_map in left_depths.items():
        filename = os.path.join(left_dir, bare_name)
        write_array(depth_map, filename)

    for bare_name, depth_map in right_depths.items():
        filename = os.path.join(right_dir, bare_name)
        write_array(depth_map, filename)


if __name__ == "__main__":

    dataset_name = "shipwreck_stavronikita"
    base_dir = "/home/bjoshi/colmap"
    display = False
    type = "geometric"

    dataset_to_robot = {
        "cave_mexico": "stereo_rig_2",
        "cavern_florida": "stereo_rig_1",
        "shipwreck_pamir": "stereo_rig_1",
        "shipwreck_stavronikita": "speedo1",
    }

    baselines = {
        "stereo_rig_2": np.array([-0.14242797, 0.00023668, -0.00122632]),
        "stereo_rig_1": np.array(
            [-0.13893451786105684, 0.0008113329400986262, 9.18232813713922e-05]
        ),
        "speedo1": np.array(
            [-0.0917754438229044, 0.005385621624085515, -0.0008669897458275737]
        ),
    }

    correct_baseline = baselines[dataset_to_robot[dataset_name]]
    colmap_folder = os.path.join(base_dir, dataset_name)
    colmap_left_poses, colmap_right_poses = read_trajectory(
        os.path.join(colmap_folder, "images.txt")
    )
    colmap_trans, colmap_quats = get_extrinsics(colmap_left_poses, colmap_right_poses)

    trans_df = pd.DataFrame(colmap_trans, columns=["x", "y", "z"])
    trans_df.plot(kind="box")
    plt.show()

    stereo_folder = os.path.join(colmap_folder, "stereo")
    colmap_stereo_left_poses, colmap_stereo_right_poses = read_trajectory(
        os.path.join(stereo_folder, "images.txt")
    )
    stereo_trans, stereo_quats = get_extrinsics(
        colmap_stereo_left_poses, colmap_stereo_right_poses
    )
    baseline = np.average(stereo_trans, axis=0)

    stereo_trans_df = pd.DataFrame(stereo_trans, columns=["x", "y", "z"])
    stereo_trans_df.plot(kind="box")
    plt.show()

    print(f"Computed baseline: {baseline}")
    scaling_factor = np.linalg.norm(correct_baseline) / np.linalg.norm(baseline)
    print(f"Scaling factor: {scaling_factor}")

    depth_map_folder = os.path.join(colmap_folder, "stereo", "stereo", "depth_maps")

    left_depths, right_depths = read_depth_maps(depth_map_folder, type)

    scaled_left_depths = {}
    for filename, left_depth_map in left_depths.items():
        scaled_left_depths[filename] = left_depth_map * scaling_factor

    scaled_right_depths = {}
    for filename, right_depth_map in right_depths.items():
        scaled_right_depths[filename] = right_depth_map * scaling_factor

    if display:
        display_stereo_depths(scaled_left_depths, scaled_right_depths)

    print("Writing scaled depth maps")
    write_stereo_images(
        os.path.join(colmap_folder, "stereo", "stereo", "depth_maps_scaled"),
        scaled_left_depths,
        scaled_right_depths,
    )

    write_gtsam_poses(
        colmap_left_poses,
        os.path.join(colmap_folder, "stereo", "colmap_traj_left.txt"),
    )
    write_gtsam_poses(
        colmap_right_poses,
        os.path.join(colmap_folder, "stereo", "colmap_traj_right.txt"),
    )
    write_gtsam_poses(
        copy.deepcopy(colmap_stereo_left_poses),
        os.path.join(colmap_folder, "stereo", "colmap_stereo_traj_left.txt"),
    )
    write_gtsam_poses(
        copy.deepcopy(colmap_stereo_right_poses),
        os.path.join(colmap_folder, "stereo", "colmap_stereo_traj_right.txt"),
    )

    (
        colmap_stereo_scale_traj_left,
        colmap_stereo_scale_traj_right,
        orignial_stamps,
    ) = create_trajectory(colmap_stereo_left_poses, colmap_stereo_right_poses)
    colmap_stereo_scale_traj_left.scale(scaling_factor)
    colmap_stereo_scale_traj_right.scale(scaling_factor)

    baseline_after_scaling = compute_baseline(
        colmap_stereo_scale_traj_left, colmap_stereo_scale_traj_right
    )
    print(f"Baseline after scaling: {np.average(baseline_after_scaling, axis=0)}")
    final_baseline = pd.DataFrame(baseline_after_scaling, columns=["x", "y", "z"])
    final_baseline.plot(kind="box")
    plt.show()

    write_evo_traj(
        os.path.join(colmap_folder, "stereo", "colmap_stereo_scaled_traj_left.txt"),
        orignial_stamps,
        colmap_stereo_scale_traj_left,
    )
    write_evo_traj(
        os.path.join(colmap_folder, "stereo", "colmap_stereo_scaled_traj_right.txt"),
        orignial_stamps,
        colmap_stereo_scale_traj_right,
    )

    colmap_pcd = open3d.io.read_point_cloud(
        os.path.join(colmap_folder, "stereo", "fused.ply")
    )
    colmap_pcd.scale(scaling_factor, np.array([0, 0, 0]))
    open3d.io.write_point_cloud(
        os.path.join(colmap_folder, "stereo", "colmap_stereo_pointcloud_scaled.ply"),
        colmap_pcd,
    )
