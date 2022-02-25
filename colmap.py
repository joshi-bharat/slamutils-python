#!/usr/bin/python

from numpy.linalg.linalg import inv, norm
import pandas as pd
import numpy as np

import gtsam

if __name__ == '__main__':
    filename = '/home/bjoshi/mexico/images.txt'
    output = '/home/bjoshi/mexico/colmap_traj.txt'

    fs = open(filename)
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
        # only left_cam
        if(int(cam_id) == 1):
            # print(info_array)
            stamp = info_array[9].split('.')[0].split('_')[1]
            # stamps.append(stamp)
            trans = np.array(info_array[5:8], dtype=np.float64)
            quat = gtsam.Rot3.Quaternion(float(info_array[1]), float(info_array[2]), float(
                info_array[3]), float(info_array[4]))  # qw qx qy qz order
            trans = gtsam.Point3(trans)
            pose = gtsam.Pose3(quat, trans)

            pose_inv = pose.inverse()
            trans = pose_inv.translation()
            quat = pose_inv.rotation().quaternion()  # qw qx qy qz

            stamp = float(stamp)/1000000000
            data = [str(stamp), trans.x(), trans.y(), trans.z(),
                    quat[1], quat[2], quat[3], quat[0]]
            camera_poses.append(data)

    df = pd.DataFrame(camera_poses, columns=[
        'stamp', 'tx', 'ty', 'tz', 'qx', 'qy', 'qz', 'qw'])
    df.sort_values(by=['stamp'], inplace=True)
    df.to_csv(output, index=False, header=False, sep=' ')
    print(df)
