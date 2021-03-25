#!/usr/bin/env python3

import os
import numpy as np

from colorama import init, Fore

init(autoreset=True)


def read_traj(filename):
    """
        assume the first column of the file contains timestamp in nanosecs  
    """

    file = open(filename)
    data = file.read()
    lines = data.replace(",", " ").replace("\t", " ").split('\n')
    data_list = [[v.strip() for v in line.split(" ") if v.strip() != ""]
                 for line in lines if len(line) > 0 and line[0] != "#"]

    x = [float(v[1]) for v in data_list]
    y = [float(v[2]) for v in data_list]
    z = [float(v[3]) for v in data_list]

    assert(len(x) == len(y) == len(z))
    # x = np.array(x)
    # y = np.array(y)
    # z = np.array(z)

    return x, y, z
