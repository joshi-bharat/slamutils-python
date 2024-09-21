from pathlib import Path
from typing import List, Dict

import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from evo.core import sync, metrics
from evo.tools import log, file_interface
from tabulate import tabulate

from slam_utils.colmap_utils import write_evo_traj, get_stamps_from_tum_trajectory
from slam_utils.plot_traj import prepare_axis, plot_traj
from slam_utils.traj_utils import interpolate_traj, read_tum_trajectory

log.configure_logging(verbose=True)

sns.set_theme(style="white", font="sans-serif", font_scale=1.2, color_codes=False)
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


def analyse_datasets(
    base_dir: Path,
    datasets: List[str],
    algorithms: Dict[str, List[str]],
    modes: List[str],
    titles: List[str],
    traj_formats: Dict[str, str],
):
    for dataset_i, dataset in enumerate(datasets):
        for mode in modes:
            colmap_file = base_dir / "data" / f"{dataset}" / "colmap.txt"
            colmap_traj = read_tum_trajectory(colmap_file)

            svin_traj = read_tum_trajectory(base_dir / "data" / f"{dataset}" / "svin.txt")
            colmap_traj.align_origin(svin_traj)

            stamps = get_stamps_from_tum_trajectory(colmap_file)

            file_name = folder / "data" / f"{dataset}" / "colmap_aligned.txt"
            write_evo_traj(file_name, stamps, colmap_traj)

            colmap_time = colmap_traj.timestamps[-1] - colmap_traj.timestamps[0]

            fig = plt.figure(figsize=(10, 10))
            ax = prepare_axis(fig)

            plot_traj(ax, colmap_traj, color=colors["colmap"], label=disp_names["colmap"])

            results = {}

            print("Trajectory length:", colmap_traj.path_length)

            for i, algo in enumerate(algorithms[mode]):
                filename = folder / "data" / f"{dataset}" / f"{mode}" / f"{algo}.txt"
                print(f"Reading traj: {filename}")
                if algo == "stereo_msckf" and mode == "stereo_imu" and dataset == "coral_reef":
                    print("Stereo MSCKF failed on coral reef dataset. Skipping ..")
                    continue

                if dataset == "cave":
                    traj_formats["openvins_lc"] = "euroc"
                else:
                    traj_formats["openvins_lc"] = "tum"

                if traj_formats[algo] == "tum":
                    traj = read_tum_trajectory(filename)
                else:
                    traj = file_interface.read_euroc_csv_trajectory(filename)

                traj_time = traj.timestamps[-1] - traj.timestamps[0]

                traj = interpolate_traj(colmap_traj.timestamps, traj)
                colmap_sync, traj_sync = sync.associate_trajectories(colmap_traj, traj)

                r, t, s = traj_sync.align(colmap_sync, correct_scale=True)

                ape_metric.process_data((colmap_sync, traj_sync))
                ape_stats = ape_metric.get_all_statistics()

                results[algo] = [round(ape_stats["rmse"], 2)]

                tracking_percent = round(traj_time / colmap_time * 100, 1)
                results[algo].append(tracking_percent)

                plot_traj(ax, traj_sync, color=colors[algo], label=disp_names[algo])

            ax.tick_params(axis="both", labelsize=16)

            fig.savefig(base_dir / f"{mode}_{dataset}_results.png", bbox_inches="tight")

            if type == "mono_imu" or type == "stereo_imu":
                ncols = (len(algorithms[mode]) + 2) // 2
            else:
                ncols = len(algorithms[mode]) + 1

            figlegend = plt.figure(figsize=(15, 3))
            ax_legend = figlegend.add_subplot(111)
            ax_legend.legend(
                *ax.get_legend_handles_labels(),
                loc="center",
                ncol=ncols,
                prop={"size": 20},
            )
            ax_legend.axis("off")
            ax.set_title(titles[dataset_i])
            figlegend.savefig(base_dir / f"{mode}_legend.png", bbox_inches="tight")
            plt.show()

            print(tabulate(results, headers="keys"))


if __name__ == "__main__":

    datasets = ["bus_outside", "cave", "cemetery", "coral_reef"]
    titles = [
        r"\textbf{Bus  Outside}",
        r"\textbf{Cave}",
        r"\textbf{Cemetery}",
        r"\textbf{Coral Reef}",
    ]
    types = ["mono", "mono_imu", "stereo", "stereo_imu"]
    folder = Path("/home/bharatjoshi/code/slamutils-python/vi_comparison")

    algorithms = {
        "mono": ["dso", "dso_lc", "orb_slam3", "orb_slam3_lc", "svo", "svo_lc"],
        "mono_imu": [
            "dm_vio",
            "kimera",
            "okvis",
            "openvins",
            "openvins_lc",
            "orb_slam3",
            "orb_slam3_lc",
            "rovio",
            "svin_lc",
            "svo",
            "svo_lc",
            "vins_fusion",
            "vins_fusion_lc",
        ],
        "stereo": [
            "orb_slam3",
            "orb_slam3_lc",
            "svo",
            "svo_lc",
            "vins_fusion",
            "vins_fusion_lc",
        ],
        "stereo_imu": [
            "kimera",
            "kimera_lc",
            "okvis",
            "openvins",
            "openvins_lc",
            "orb_slam3",
            "orb_slam3_lc",
            "stereo_msckf",
            "svin_lc",
            "svo",
            "svo_lc",
            "vins_fusion",
            "vins_fusion_lc",
        ],
    }

    traj_formats = {
        "dso": "tum",
        "dso_lc": "tum",
        "orb_slam3": "tum",
        "orb_slam3_lc": "tum",
        "svin": "tum",
        "svin_lc": "tum",
        "svo": "tum",
        "svo_lc": "tum",
        "dm_vio": "tum",
        "kimera": "euroc",
        "kimera_lc": "euroc",
        "okvis": "euroc",
        "openvins": "tum",
        "openvins_lc": "euroc",
        "rovio": "tum",
        "vins_fusion": "tum",
        "vins_fusion_lc": "tum",
        "stereo_msckf": "tum",
    }

    colors = {
        "dso": "magenta",
        "dso_lc": "mediumblue",
        "orb_slam3": "tomato",
        "orb_slam3_lc": "brown",
        "svo": "lime",
        "svo_lc": "forestgreen",
        "dm_vio": "darkorange",
        "kimera": "violet",
        "kimera_lc": "gold",
        "okvis": "deeppink",
        "openvins": "cyan",
        "openvins_lc": "olive",
        "rovio": "indigo",
        "vins_fusion": "royalblue",
        "vins_fusion_lc": "darkblue",
        "stereo_msckf": "purple",
        "colmap": "black",
        "svin_lc": "darkslategrey",
    }

    disp_names = {
        "dso": r"\textbf{dso}",
        "dso_lc": r"\textbf{dso\_lc}",
        "orb_slam3": r"\textbf{orb\_slam3}",
        "orb_slam3_lc": r"\textbf{orb\_slam3\_lc}",
        "svo": r"\textbf{svo}",
        "svo_lc": r"\textbf{svo\_lc}",
        "dm_vio": r"\textbf{dm\_vio}",
        "kimera": r"\textbf{kimera}",
        "kimera_lc": r"\textbf{kimera\_lc}",
        "okvis": r"\textbf{okvis}",
        "openvins": r"\textbf{openvins}",
        "openvins_lc": r"\textbf{openvins\_lc}",
        "rovio": r"\textbf{rovio}",
        "vins_fusion": r"\textbf{vins_fusion}",
        "vins_fusion_lc": r"\textbf{vins_fusion_lc}",
        "stereo_msckf": r"\textbf{stereo_msckf}",
        "colmap": r"\textbf{colmap}",
        "svin_lc": r"\textbf{svin\_lc}",
    }

    analyse_datasets(folder, datasets, algorithms, types, titles, traj_formats)
    # colmap_file = folder / "data" / f"{dataset}" / "colmap.txt"
    # colmap_traj = read_tum_trajectory(colmap_file)
    #
    # svin_traj = read_tum_trajectory(folder / "data" / f"{dataset}" / "svin.txt")
    # colmap_traj.align_origin(svin_traj)
    # colmap_time = colmap_traj.timestamps[-1] - colmap_traj.timestamps[0]
    #
    # fig = plt.figure(figsize=(10, 10))
    # ax = prepare_axis(fig)
    #
    # plot_traj(ax, colmap_traj, color=colors["colmap"], label=disp_names["colmap"])
    #
    # results = {}
    #
    # print("Trajectory length:", colmap_traj.path_length)
    #
    # for algo in algorithms[type]:
    #     filename = folder / "data" / f"{dataset}" / f"{type}" / f"{algo}.txt"
    #     if traj_formats[algo] == "tum":
    #         traj = read_tum_trajectory(filename)
    #     else:
    #         traj = file_interface.read_euroc_csv_trajectory(filename)
    #
    #     traj_time = traj.timestamps[-1] - traj.timestamps[0]
    #
    #     traj = interpolate_traj(colmap_traj.timestamps, traj)
    #     colmap_sync, traj_sync = sync.associate_trajectories(colmap_traj, traj)
    #
    #     r, t, s = traj_sync.align(colmap_sync, correct_scale=True)
    #
    #     ape_metric.process_data((colmap_sync, traj_sync))
    #     ape_stats = ape_metric.get_all_statistics()
    #
    #     results[algo] = [round(ape_stats["rmse"], 2)]
    #
    #     tracking_percent = round(traj_time / colmap_time * 100, 1)
    #     results[algo].append(tracking_percent)
    #
    #     plot_traj(ax, traj_sync, color=colors[algo], label=disp_names[algo])
    #
    # ax.tick_params(axis="both", labelsize=16)
    #
    # fig.savefig(f"{type}_{dataset}_results.png", bbox_inches="tight")

    # if type == "mono_imu" or type == "stereo_imu":
    #     ncols = (len(algorithms[type]) + 2) // 2
    # else:
    #     ncols = len(algorithms[type]) + 1
    #
    # figlegend = plt.figure(figsize=(15, 3))
    # ax_legend = figlegend.add_subplot(111)
    # ax_legend.legend(
    #     *ax.get_legend_handles_labels(),
    #     loc="center",
    #     ncol=ncols,
    #     prop={"size": 20},
    # )
    # ax_legend.axis("off")
    # figlegend.savefig(f"{type}_legend.png", bbox_inches="tight")
    # plt.show()
    #
    # print(tabulate(results, headers="keys"))
