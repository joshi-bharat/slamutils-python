from evo.core.trajectory import PoseTrajectory3D
import numpy as np
from scipy.spatial.transform import Rotation
from scipy.spatial.transform import Slerp

np.set_printoptions(suppress=True)


def interpolate_traj(
    ref_timestamps: np.ndarray, traj: PoseTrajectory3D
) -> PoseTrajectory3D:

    positions = traj.positions_xyz
    x = positions[:, 0]
    y = positions[:, 1]
    z = positions[:, 2]
    timestamps = traj.timestamps

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
