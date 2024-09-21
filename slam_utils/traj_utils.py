from evo.core.trajectory import PoseTrajectory3D
import numpy as np
from scipy.spatial.transform import Rotation
from scipy.spatial.transform import Slerp
import pandas as pd

np.set_printoptions(suppress=True)

def interpolate_traj(
    ref_timestamps: np.ndarray, traj: PoseTrajectory3D
) -> PoseTrajectory3D:

    positions = traj.positions_xyz
    x = positions[:, 0]
    y = positions[:, 1]
    z = positions[:, 2]
    timestamps = traj.timestamps
    for i in range(len(timestamps)-1):
        if timestamps[i+1] <= timestamps[i]:
            print(i)

    first_index = 0
    last_index = len(ref_timestamps) - 1
    while ref_timestamps[first_index] < timestamps[0]:
        first_index += 1
    while ref_timestamps[last_index] > timestamps[-1]:
        last_index -= 1
    ref_timestamps = ref_timestamps[first_index:last_index]
    
    # interpolate positions
    x_interp_positions = np.interp(ref_timestamps, timestamps, x, left=None, right=None)
    y_interp_positions = np.interp(ref_timestamps, timestamps, y, left=None, right=None)
    z_interp_positions = np.interp(ref_timestamps, timestamps, z, left=None, right=None)

    # interpolate orientations
    quat_wxyz = traj.orientations_quat_wxyz
    qw = np.reshape(quat_wxyz[:, 0], (-1, 1))
    qxyz = quat_wxyz[:, 1:]
    quat_xyzw = np.concatenate((qxyz, qw), axis=1)
    rot = Rotation.from_quat(quat_xyzw)
    slerp = Slerp(timestamps, rot)
    interp_rot = slerp(ref_timestamps).as_quat()
    interp_quat_wxyz = np.concatenate(
        (np.reshape(interp_rot[:, 3], (-1, 1)), interp_rot[:, :3]), axis=1
    )

    interp_trans = np.stack(
        (x_interp_positions, y_interp_positions, z_interp_positions), axis=1
    )

    return PoseTrajectory3D(interp_trans, interp_quat_wxyz, ref_timestamps)

def read_tum_trajectory(filename: str)-> PoseTrajectory3D:
    all_lines = []
    with open(filename, newline='') as f:
        lines = f.readlines()
        for line in lines:
            if line.startswith('#'):
                continue
            else:
                line = line.strip()
                line = line.split()
                line = [float(x) for x in line]
                all_lines.append(line)
    data = np.array(all_lines).astype(float)
    data = data[:, :8]
    df = pd.DataFrame(
        data, columns=["stamp", "tx", "ty", "tz", "qw", "qx", "qy", "qz"]
    )
    df.sort_values(by=['stamp'], inplace=True)
    data = df.to_numpy()
    stamps = data[:, 0]  # n x 1
    xyz = data[:, 1:4]  # n x 3
    quat = data[:, 4:]  # n x 4
    quat = np.roll(quat, 1, axis=1)  # shift 1 column -> w in front column
    if not hasattr(filename, 'read'):  # if not file handle
        print("Loaded {} stamps and poses from: {}".format(
            len(stamps), filename))
    return PoseTrajectory3D(xyz, quat, stamps)
