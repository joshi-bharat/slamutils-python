#!/usr/bin/python


from numpy import linalg
from numpy.linalg.linalg import inv
import pandas as pd
import numpy as np

from tf.transformations import *

if __name__ == '__main__':
    filename = '/home/bjoshi/Downloads/images.txt'
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
    all_trans = []

    for image_info in image_infos:
        if image_info.startswith('#'):
            continue

        image_info = image_info.strip()
        info_array = image_info.split(' ')
        cam_id = info_array[8]
        # only left_cam
        if(int(cam_id) == 1):
            # print(info_array)
            stamp = info_array[9].split('_')[1].split('.')[0]
            trans = np.array(info_array[5:8], dtype=np.float64)
            quat = (float(info_array[2]), float(info_array[3]), float(info_array[4]),
                    float(info_array[1]))  # qx qy qz qw order
            rotm = quaternion_matrix(quat)[:3, :3]
            trans_result = - (np.linalg.inv(rotm).dot(trans))
            # print("result: {}".format(trans_result))
            data = [stamp] + trans_result.tolist()
            camera_poses.append(data)
            all_trans.append(trans_result)
        # print(data)

    all_trans = np.array(all_trans)
    relative_trans = np.diff(all_trans, axis=0)
    dist = np.linalg.norm(relative_trans, axis=1)
    result = np.where(dist > 3)
    print(result)

    # print(camera_poses)
    df = pd.DataFrame(camera_poses, columns=['stamp', 'tx', 'ty', 'tz'])
    df.sort_values(by=['stamp'])
    df.to_csv('colmap_poses.txt', index=False, header=False)
    # print(df)
