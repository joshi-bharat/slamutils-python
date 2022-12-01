from pathlib import Path
from pprint import pprint
from evo.tools import log, plot, file_interface
from evo.core import sync, metrics, trajectory
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt

from interpolate_traj import interpolate_traj
from plot_traj import set_aspect_equal_3d, prepare_axis, plot_traj

log.configure_logging(verbose=True)

sns.set(style="white", font="sans-serif", font_scale=1.2, color_codes=False)
rc = {
    "lines.linewidth": 1.2,
    "text.usetex": True,
    "font.family": "sans-serif",
    "pgf.texsystem": "pdflatex",
    "text.latex.preamble": r"\usepackage{amsmath} \usepackage{amssymb}",
}

matplotlib.rcParams.update(rc)

pose_relation = metrics.PoseRelation.translation_part
ape_metric = metrics.APE(pose_relation)

if __name__ == "__main__":

    dataset = "cave"
    folder = Path("/home/bjoshi/code/slamutils-python/comparison_paper")

    colmap_file = folder / "data" / f"{dataset}" / "colmap.txt"
    colmap_traj = file_interface.read_tum_trajectory_file(colmap_file)

    orb = folder / "data" / f"{dataset}" / "mono" / "orb_slam3.txt"
    orb_traj = file_interface.read_tum_trajectory_file(orb)

    orb_interp = interpolate_traj(colmap_traj.timestamps, orb_traj)

    colmap_sync, orb_sync = sync.associate_trajectories(colmap_traj, orb_interp)

    r, t, s = orb_sync.align(colmap_sync, correct_scale=True)
    print("scale: {}".format(s))
    ape_metric.process_data((orb_sync, colmap_sync))
    ape_stats = ape_metric.get_all_statistics()
    pprint(ape_stats)

    fig = plt.figure(figsize=(8, 8))
    ax = prepare_axis(fig, plot_mode=plot.PlotMode.xyz)
    ax.set_title("test", fontsize=16)

    plot_traj(ax, orb_sync, plot_mode=plot.PlotMode.xyz, color="r")
    plt.show()
