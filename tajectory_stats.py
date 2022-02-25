#!/usr/bin/env python

import os
from evo.core import metrics, trajectory, sync
from evo.tools import file_interface
import pprint
import numpy as np

from coral_square_plots import plot_traj

if __name__ == '__main__':
    folder = "./reef_lawnmower_trajs"

    gt_traj = file_interface.read_tum_trajectory_file(
        os.path.join(folder, "svin_gt.txt"))

    svin_traj = file_interface.read_tum_trajectory_file(
        os.path.join(folder, "svin_traj_5_20_1.txt"))

    print(gt_traj)

    # rmse_errors = []
    # for i in range(5):

    #     file_name = os.path.join(folder, "robust_traj_5_20_{}.txt".format(i))
    #     traj = file_interface.read_tum_trajectory_file(file_name)

    #     traj.align_origin(gt_traj)

    #     gt_traj, traj = sync.associate_trajectories(gt_traj, traj, 0.1)

    #     pose_relation = metrics.PoseRelation.translation_part
    #     data = (gt_traj, traj)
    #     ape_metric = metrics.APE(pose_relation)
    #     ape_metric.process_data(data)
    #     ape_stats = ape_metric.get_all_statistics()
    #     # pprint.pprint(ape_stats)

    #     rmse_errors.append(ape_stats['rmse'])

    # rmse_errors = np.array(rmse_errors)
    # print("RMSE Mean: {:.3f}".format(rmse_errors.mean()))
    # print("RMSE std: {:.3f}".format(rmse_errors.std()))
