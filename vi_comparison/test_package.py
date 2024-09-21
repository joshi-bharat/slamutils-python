from pathlib import Path
from evo.tools import log, plot, file_interface
from evo.core import sync, metrics
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt

from slam_utils.traj_utils import interpolate_traj, read_tum_trajectory
from slam_utils.plot_traj import prepare_axis, plot_traj

log.configure_logging(verbose=True)

sns.set(style="white", font="sans-serif", font_scale=1.2, color_codes=False)
rc = {
    "lines.linewidth": 1.5,
    "text.usetex": True,
    "font.family": "sans-serif",
    "font.size": 16,
    "pgf.texsystem": "pdflatex",
    "text.latex.preamble": r"\usepackage{amsmath} \usepackage{amssymb}",
}

matplotlib.rcParams.update(rc)

pose_relation = metrics.PoseRelation.translation_part
ape_metric = metrics.APE(pose_relation)

if __name__ == "__main__":

    dataset = "cave"
    title = r"\textbf{Cave}"
    type = "mono_imu"
    folder = Path("/home/bjoshi/code/slamutils-python/vi_comparison")

    colmap_file = folder / "data" / f"{dataset}" / "colmap.txt"
    colmap_traj = read_tum_trajectory(colmap_file)

    colmap_time = colmap_traj.timestamps[-1] - colmap_traj.timestamps[0]

    fig = plt.figure(figsize=(10, 10))
    ax = prepare_axis(fig, plot_mode=plot.PlotMode.xyz)

    plot_traj(
        ax, colmap_traj, color="black", label="colmap", plot_mode=plot.PlotMode.xyz
    )

    results = {}

    print("Trajectory length:", colmap_traj.path_length)

    filename = (
        folder / "data" / f"{dataset}" / "mono_imu" / "okvis_estimator_output.csv"
    )
    traj = file_interface.read_euroc_csv_trajectory(filename)

    traj_time = traj.timestamps[-1] - traj.timestamps[0]

    traj = interpolate_traj(colmap_traj.timestamps, traj)
    colmap_sync, traj_sync = sync.associate_trajectories(colmap_traj, traj)

    r, t, s = traj_sync.align(colmap_sync, correct_scale=True)

    ape_metric.process_data((colmap_sync, traj_sync))
    ape_stats = ape_metric.get_all_statistics()

    print(ape_stats)

    tracking_percent = round(traj_time / colmap_time * 100, 1)
    print(f"Tracking : {tracking_percent}%")

    plot_traj(ax, traj_sync, color="red", plot_mode=plot.PlotMode.xyz)

    ax.tick_params(axis="both", labelsize=16)
    plt.show()
