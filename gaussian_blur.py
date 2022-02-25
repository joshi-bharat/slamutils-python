#!/usr/bin/env python

import copy
from turtle import st
import rosbag
from tqdm import tqdm
import numpy as np
import cv_bridge
import cv2
from skimage.util import random_noise
import random


if __name__ == '__main__':

    debug = True

    bagfile = '/home/bjoshi/Downloads/afrl_bbdos2019_coral_lmw_cut.bag'
    bag = rosbag.Bag(bagfile, 'r')
    outbag = rosbag.Bag(
        '/home/bjoshi/Downloads/afrl_bbdos2019_coral_lmw_blur_1_60.bag', 'w')

    bridge = cv_bridge.CvBridge()

    bag_length = 314
    segments = 1
    duration = 60
    skip_start_end = 30
    # do not want to blur the first and last part of bag file
    segment_duration = (bag_length - skip_start_end * 2) / segments

    start_times = []
    bag_start = bag.get_start_time()

    for i in range(segments):
        start = int(skip_start_end + i * segment_duration)
        end = int(start + segment_duration - duration)

        start_time = random.randint(start, end)
        print(start_time)
        start_times.append(float(start_time) + bag_start)

    for topic, msg, t in tqdm(bag.read_messages(),
                              total=bag.get_message_count()):
        if topic in ['/cam_fl/image_raw/compressed', '/cam_fr/image_raw/compressed']:

            if any(t.to_sec() >= start_time and t.to_sec() <= start_time+duration for start_time in start_times):
                cv_image = bridge.compressed_imgmsg_to_cv2(msg, 'bgr8')
                cv_image_blur = cv2.GaussianBlur(
                    copy.deepcopy(cv_image), (21, 21), sigmaX=11, sigmaY=11)

                compressed_msg = bridge.cv2_to_compressed_imgmsg(cv_image_blur)
                compressed_msg.header = msg.header
                outbag.write(topic, compressed_msg, t)

            else:
                outbag.write(topic, msg, t)

        else:
            outbag.write(topic, msg, t)

    bag.close()
    outbag.close()
