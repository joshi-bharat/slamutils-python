#!/usr/bin/python

import math
import pandas as pd
import numpy as np

from evo.core import transformations as tr
if __name__ == "__main__":

    filename = '/home/bjoshi/results/openwater/vins_result_loop.csv'
    output = '/home/bjoshi/results/openwater/vinsmono.txt'

    rot = tr.euler_matrix(-math.pi/2.0, 0.0, math.pi)

    all_poses = []
    with open(filename) as f:
        lines = f.readlines()
        for line in lines:
            line_arr = line.strip().split(',')
            line_arr = [float(x) for x in line_arr]
            stamp = line_arr[0]
            stamp = stamp/1000000000
            trans = [line_arr[1], line_arr[2], line_arr[3]]
            quat = [line_arr[4], line_arr[5], line_arr[6], line_arr[7]]
            rotm = np.dot(tr.quaternion_matrix(quat), rot)
            quat = tr.quaternion_from_matrix(rotm)
            all_poses.append([str(stamp), trans[0], trans[1],
                              trans[2], quat[1], quat[2], quat[3], quat[0]])

    df = pd.DataFrame(all_poses, columns=[
                      'stamp', 'tx', 'ty', 'tz', 'qx', 'qy', 'qz', 'qw'])
    df.sort_values(by=['stamp'])
    df.to_csv(output, index=False, header=False, sep=' ')
    print(df)
