#!/usr/bin/env python

import argparse
import os
import subprocess

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Create extra files to run dm-vio for dataset in euroc format."
    )

    parser.add_argument(
        "--folder",
        type=str,
        help="Location where the dataset shall be downloaded to.",
        required=True,
    )

    args = parser.parse_args()
    folder = args.folder
    camera_source = os.path.join(folder, "mav0/cam0/data.csv")
    camera_target = os.path.join(folder, "mav0/cam0/times.txt")

    with open(camera_source, "r") as f:
        lines = f.readlines()
    
    lines = [line.split(',')[0] for line in lines if not line.startswith('#')]
    with open(camera_target, 'w') as target_file:
        for line in lines:
            nanoseconds = int(round(int(line[-9:]), -2) / 100)
            seconds = int(line[:-9])
            target_file.write(f"{line} {seconds}.{nanoseconds}\n")


    # Create IMU file. --> just remove comment lines and replace commas with spaces.
    imu_source = os.path.join(folder , 'mav0' , 'imu0' , 'data.csv')
    imu_target = os.path.join(folder , 'mav0' , 'imu0' , 'imu.txt')
    with open(imu_source, 'r') as source_file:
        lines = source_file.readlines()
    lines = [line.replace(',', ' ') for line in lines if not line.startswith('#')]
    with open(imu_target, 'w') as target_file:
        target_file.writelines(lines)

    imu_dmvio = os.path.join(folder , 'mav0' , 'imu0' , 'imu_dmvio.txt')
    subprocess.run(f"python interpolate_imu_file.py --input {imu_target} --times {camera_target} --output {imu_dmvio}", shell=True)

