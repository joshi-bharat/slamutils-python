# !/usr/bin/python

import os
import numpy as np
from numpy.lib.function_base import average
import pandas as pd
from matplotlib import pyplot as plt

rc = {
    "lines.linewidth": 1.5,
    "text.usetex": True,
    "font.family": 'sans-serif',
    "pgf.texsystem": 'pdflatex',
    'text.latex.preamble': r'\usepackage{amsmath} \usepackage{amssymb}'
}

plt.rcParams.update(rc)

if __name__ == '__main__':

    dataset_path = '/home/bjoshi/results/tags'
    pkgs = ['colmap', 'openvins', 'orbslam3', 'svin2', 'vinsmono']
    disp_names = ['COLMAP', 'OpenVINS', 'ORB-SLAM3', 'SVIn2', 'VINS-Mono']
    colors = ['k', 'b', 'g', 'r', 'gold']

    tags = ['1', '3', '5', '7', '10']

    display_plot = True

    distance_errors_pkg = []
    stds_per_pkg = {}
    average_dist_error = {}

    for pkg in pkgs:
        data = np.loadtxt(os.path.join(dataset_path, pkg + '_tag_locs.txt'))

        stds = []
        distances = []
        distance_errors = []
        for tag_id in tags:
            tag_data = data[np.where(data[:, 0] == int(tag_id))]
            std = np.std(tag_data[:, 1:], axis=0)

            average_pos = np.average(tag_data[:, 1:], axis=0)
            dist = tag_data[:, 1:] - average_pos
            distance_error = np.linalg.norm(dist, axis=1)

            euler_dist = np.linalg.norm(tag_data[:, 1:], axis=1)
            average_dist = np.mean(euler_dist)
            diff = euler_dist - average_dist
            stds.append(std)
            distance_errors.extend(distance_error)
            distances.extend(diff)
            # print(distances)
            # tags[tag_id]=tag_data[:, 1:3]

        stds = np.array(stds)
        avg_std = np.mean(stds, axis=0)
        distance_errors_pkg.append(np.array(distances))
        stds_per_pkg[pkg] = avg_std
        average_dist_error[pkg] = np.mean(distance_errors, axis=0)

    print(stds_per_pkg)
    print(average_dist_error)

    distances = np.array(distance_errors_pkg).transpose()
    distance_df = pd.DataFrame(distances, columns=disp_names)

    boxplot = distance_df.boxplot(
        grid=False, showfliers=False, return_type='both')
    axes = boxplot.ax
    axes.tick_params(
        axis='x',          # changes apply to the x-axis
        which='both',      # both major and minor ticks are affected
        bottom=False,      # ticks along the bottom edge are off
        top=False)         # ticks along the top edge are off)

    # For the size of the measurment ticks
    for tick in axes.yaxis.get_major_ticks():
        tick.label.set_fontsize(12)
    for tick in axes.xaxis.get_major_ticks():
        tick.label.set_fontsize(12)

    axes.set_ylabel('Distance error (m)', fontsize=12)
    axes.set_title(r'g1\_cavern2', fontsize=16)

    lines = boxplot.lines
    whiskers = lines['whiskers']
    caps = lines['caps']
    boxes = lines['boxes']
    medians = lines['medians']

    for counter, val in enumerate(whiskers):
        val.set_color(colors[int(counter/2)])
        val.set_linewidth(1.5)
    for counter, val in enumerate(caps):
        val.set_color(colors[int(counter/2)])
        val.set_linewidth(1.5)
    for counter, val in enumerate(boxes):
        val.set_color(colors[counter])
        val.set_linewidth(1.5)
    for counter, val in enumerate(medians):
        val.set_color(colors[counter])
        val.set_linewidth(1.5)

    for axis in ['top', 'bottom', 'left', 'right']:
        axes.spines[axis].set_linewidth(1.0)
    plt.plot()
    if display_plot:
        plt.show()
    else:
        plt.savefig(os.path.join(dataset_path, 'tag_distance_error'+'.png'),
                    bbox_inches='tight')
