#!/usr/bin/python

from evo.tools.settings import SETTINGS
import numpy as np
import pprint
from evo.core import metrics, sync
from evo.tools import log, plot, file_interface
from matplotlib import pyplot as plt
import os
import copy
from slam_utils.colmap_utils import get_stamps_from_tum_trajectory, write_evo_traj

log.configure_logging(verbose=True, debug=False, silent=False)

# temporarily override some package settings
SETTINGS.plot_usetex = False


def align_trajectories(
    ref_traj, traj, max_diff=0.1, align_origin=False, correct_scale=False
):

    ref_sync, traj_sync = sync.associate_trajectories(ref_traj, traj, max_diff)

    aligned_traj = copy.deepcopy(traj_sync)
    transform_4x4 = np.identity(4)
    scale = 1.0
    if align_origin:
        transform_4x4 = aligned_traj.align_origin(ref_traj)
    else:
        rot, translation, scale = aligned_traj.align(
            ref_sync, correct_scale=correct_scale
        )
        transform_4x4[:3, :3] = rot
        transform_4x4[:3, 3] = translation

    return transform_4x4, scale, aligned_traj


def plot_trajs(ref_traj, traj, aligned_traj):
    fig = plt.figure()
    traj_by_label = {"svin": traj, "svin (aligned)": aligned_traj, "colmap": ref_traj}

    plot.trajectories(fig, traj_by_label, plot.PlotMode.xyz)
    plt.show()


def write_traj(traj, filename):
    file_interface.save_trajectory(traj, filename)


if __name__ == "__main__":

    dataset_name = "coral_reef"
    folder = "/home/bjoshi/code/slamutils-python/icra"

    colmap_file = os.path.join(folder, f"colmap_{dataset_name}.txt")
    svin_file = os.path.join(folder, f"svin.txt")

    svin_traj = file_interface.read_tum_trajectory_file(svin_file)
    colmap_traj = file_interface.read_tum_trajectory_file(colmap_file)

    transform_4x4, scale, svin_aligned_traj = align_trajectories(
        colmap_traj, svin_traj, align_origin=False, correct_scale=True
    )

    plot_trajs(colmap_traj, svin_traj, svin_aligned_traj)

    pose_relation = metrics.PoseRelation.translation_part
    data = (colmap_traj, svin_aligned_traj)
    print(colmap_traj)
    print(svin_aligned_traj)
    ape_metric = metrics.APE(pose_relation)
    ape_metric.process_data(data)

    ape_stats = ape_metric.get_all_statistics()
    pprint.pprint(ape_stats)

    original_stamps = get_stamps_from_tum_trajectory(svin_file)
    svin_aligned_file = os.path.join(folder, f"{dataset_name}_svin_aligned_traj.txt")
    write_evo_traj(svin_aligned_file, original_stamps, svin_aligned_traj)
