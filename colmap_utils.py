#!/usr/bin/python3

import struct
from typing import Dict, List, Optional

from gtsam import Pose3, Rot3, Point3
import numpy as np
from matplotlib import animation
from matplotlib import pyplot as plt
from evo.core import trajectory


def read_trajectory(file: str) -> List[Dict]:
    left_poses = {}
    right_poses = {}

    fs = open(file)
    for _ in range(4):
        fs.readline()

    lines = fs.readlines()
    print(f"Read {len(lines)} lines from {file}")

    # Image list with two lines of data per image:
    #   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME
    #   POINTS2D[] as (X, Y, POINT3D_ID)

    # Skipping odd lines as we need only camera poses.
    image_infos = lines[::2]
    left_images = 0
    right_images = 0

    for image_info in image_infos:
        if image_info.startswith("#"):
            continue

        image_info = image_info.strip()
        info_array = image_info.split(" ")
        cam_id = info_array[8]
        image_name = info_array[9].split("/")[1]
        stamp = image_name.split(".")[0]

        trans_array = np.array(info_array[5:8], dtype=np.float64)
        quat = Rot3.Quaternion(
            float(info_array[1]),
            float(info_array[2]),
            float(info_array[3]),
            float(info_array[4]),
        )  # qw qx qy qz order
        trans = Point3(trans_array)
        pose = Pose3(quat, trans)

        # cam_pose is inverse
        pose_inv = pose.inverse()

        if cam_id == "1":
            left_poses[stamp] = pose_inv
            left_images += 1
        elif cam_id == "2":
            right_poses[stamp] = pose_inv
            right_images += 1

    fs.close()

    print(f"Read {left_images} left images and {right_images} right images")
    return left_poses, right_poses


def read_array(filename: str):
    with open(filename, "rb") as fid:
        width, height, channels = np.genfromtxt(
            fid, delimiter="&", max_rows=1, usecols=(0, 1, 2), dtype=int
        )
        fid.seek(0)
        num_delimiter = 0
        byte = fid.read(1)
        while True:
            if byte == b"&":
                num_delimiter += 1
                if num_delimiter >= 3:
                    break
            byte = fid.read(1)
        array = np.fromfile(fid, np.float32)
    array = array.reshape((width, height, channels), order="F")
    return np.transpose(array, (1, 0, 2)).squeeze()


def write_array(array, path):
    """
    see: src/mvs/mat.h
        void Mat<T>::Write(const std::string& path)
    """
    assert array.dtype == np.float32
    if len(array.shape) == 2:
        height, width = array.shape
        channels = 1
    elif len(array.shape) == 3:
        height, width, channels = array.shape
    else:
        assert False

    with open(path, "w") as fid:
        fid.write(str(width) + "&" + str(height) + "&" + str(channels) + "&")

    with open(path, "ab") as fid:
        if len(array.shape) == 2:
            array_trans = np.transpose(array, (1, 0))
        elif len(array.shape) == 3:
            array_trans = np.transpose(array, (1, 0, 2))
        else:
            assert False
        data_1d = array_trans.reshape(-1, order="F")
        data_list = data_1d.tolist()
        endian_character = "<"
        format_char_sequence = "".join(["f"] * len(data_list))
        byte_data = struct.pack(endian_character + format_char_sequence, *data_list)
        fid.write(byte_data)


def display_stereo_depths(
    left_depths: Dict[str, np.array],
    right_depths: Dict[str, np.array],
    title: Optional[str] = "Stereo Depth Maps",
):
    images = []
    fig = plt.figure(figsize=(20, 10))

    for filename, left_depth_map in left_depths.items():
        if filename in right_depths:
            combined_map = np.hstack((left_depth_map, right_depths[filename]))
            min_depth, max_depth = np.percentile(combined_map, [1, 99])
            combined_map[combined_map < min_depth] = min_depth
            combined_map[combined_map > max_depth] = max_depth

            images.append([plt.imshow(combined_map, animated=True, cmap="summer")])
    ani = animation.ArtistAnimation(fig, images, interval=50, blit=True, repeat=False)
    plt.title(title)
    plt.show()


def compute_baseline(
    left_poses: trajectory.PoseTrajectory3D, right_poses: trajectory.PoseTrajectory3D
) -> np.array:
    baseline = []
    for i in range(len(left_poses.timestamps)):
        left_trans = left_poses.positions_xyz[i, :]
        left_quat = left_poses.orientations_quat_wxyz[i, :]
        left_pose = Pose3(
            Rot3.Quaternion(left_quat[0], left_quat[1], left_quat[2], left_quat[3]),
            Point3(left_trans),
        )

        right_trans = right_poses.positions_xyz[i, :]
        right_quat = right_poses.orientations_quat_wxyz[i, :]
        right_pose = Pose3(
            Rot3.Quaternion(right_quat[0], right_quat[1], right_quat[2], right_quat[3]),
            Point3(right_trans),
        )

        diff = right_pose.between(left_pose)
        trans = diff.translation()
        baseline.append([trans[0], trans[1], trans[2]])

    trans = np.array(baseline).astype(np.float64)
    return trans


def write_gtsam_poses(gtsam_poses: Dict[str, Pose3], filename: str):
    with open(filename, "w") as f:
        for stamp, pose in sorted(gtsam_poses.items()):
            stamp = stamp[:-9] + "." + stamp[-9:]
            trans = pose.translation()
            quat_wxyz = pose.rotation().quaternion()
            f.write(
                f"{stamp} {trans[0]} {trans[1]} {trans[2]} {quat_wxyz[1]} {quat_wxyz[2]} {quat_wxyz[3]} {quat_wxyz[0]}\n"
            )


def write_evo_traj(filename: str, stamps: List[str], traj: trajectory.PoseTrajectory3D):

    with open(filename, "w") as f:
        for stamp, trans, quat in zip(
            stamps, traj.positions_xyz, traj.orientations_quat_wxyz
        ):
            stamp = stamp[:-9] + "." + stamp[-9:]
            f.write(
                f"{stamp} {trans[0]} {trans[1]} {trans[2]} {quat[1]} {quat[2]} {quat[3]} {quat[0]}\n"
            )
