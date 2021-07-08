#!/usr/bin/python3

import gtsam
import numpy as np
import pandas as pd

left_poses = {}
right_poses = {}

def read_trajectory(file):
    fs = open(file)
    for i in range(4):
        fs.readline()

    lines = fs.readlines()
    print('Read {} lines from {}.'.format(len(lines), filename))

    # Image list with two lines of data per image:
    #   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME
    #   POINTS2D[] as (X, Y, POINT3D_ID)

    # Skipping odd lines as we need only camera poses.
    image_infos = lines[::2]

    camera_poses = []

    for image_info in image_infos:
        if image_info.startswith('#'):
            continue

        image_info = image_info.strip()
        info_array = image_info.split(' ')
        cam_id = info_array[8]
        image_name = info_array[9].split('_')[1]
        stamp = image_name.split('.')[0]

        trans_array = np.array(info_array[5:8], dtype=np.float64)
        quat = gtsam.Rot3.Quaternion(float(info_array[1]), float(info_array[2]), float(info_array[3]), float(info_array[4]))  # qw qx qy qz order
        trans = gtsam.Point3(trans_array)
        pose = gtsam.Pose3(quat, trans)

        #cam_pose is inverse
        pose_inv = pose.inverse()
        trans_inv = pose_inv.translation()

        if cam_id == '1':
            # camera_poses.append([stamp, trans_inv.x(), trans_inv.y(), trans_inv.z()])
            left_poses[image_name] = pose_inv
        elif cam_id == '2':
            # camera_poses.append([stamp, trans_inv.x(), trans_inv.y(), trans_inv.z()])
            right_poses[image_name] = pose_inv

    # df = pd.DataFrame(camera_poses, columns=['image_name', 'tx', 'ty', 'tz'])
    # df.sort_values(by=['image_name'], inplace=True)
    # df.to_csv('colmap_poses_right.txt', index=False, header=False)

    fs.close()


if __name__ == '__main__':
    filename = '/home/bjoshi/colmap_cave_stereo/latest/images.txt'
    read_trajectory(filename)

    baseline = []
    quats = []
    for limg, lpose in left_poses.items():
        if not limg in right_poses:
            print('No right image for: {}'.format(limg))
            continue
        else:
            rpose = right_poses[limg]
        diff = rpose.between(lpose)
        trans  = diff.translation()
        quat = diff.rotation().quaternion()
        quats.append(quat)
        baseline.append([trans.x(), trans.y(), trans.z()])

    baseline = np.array(baseline).astype(np.float64)
    quats = np.array(quats).astype(np.float64)

    avg_baseline = np.average(baseline, axis=0)
    sd = np.std(baseline, axis=0)
    avg_quat = np.average(quats, axis=0)
    quat_sd = np.std(quats, axis=0)
    print('Stereo Rig Baseline: {} {} {}'.format(avg_baseline,u'\u00B1', sd))
    print('Quat: {} {} {}'.format(avg_quat,u'\u00B1', quat_sd))