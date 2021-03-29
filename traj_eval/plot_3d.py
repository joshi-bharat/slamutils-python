#!/usr/bin/env python3

from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
import argparse

from traj_reader import read_traj
from colorama import init, Fore

init(autoreset=True)


def plot3d(x, y, z):
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.plot3D(x, y, z, 'green')

    plt.show()


if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description='''Just 3D data plot''')

    parser.add_argument(
        '--traj_file', type=str,
        help="Trajectory to plot.")

    args = parser.parse_args()

    x, y, z = read_traj(args.traj_file)

    print(Fore.GREEN + 'Read trajectory file with {} rows'.format(len(x)))

    plot3d(x, y, z)
