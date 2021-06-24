#!/usr/bin/python3

from typing import Collection
import pandas as pd

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

    print("Remaining lines: {}".format(len(image_infos)))

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
            stamp = info_array[9].split('_')[1].split('.')[0]
            trans = info_array[5:8]
            quat = 
            data = [stamp] + 
            camera_poses.append(data)
        # print(data)

    # print(camera_poses)
    df = pd.DataFrame(camera_poses, columns=['stamp', 'tx', 'ty', 'tz'])
    # df.sort_values(by=['stamp'])
    df.to_csv('colmap_poses.txt', index=False, header=False)
    print(df)
