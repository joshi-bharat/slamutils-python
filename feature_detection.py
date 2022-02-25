#!/usr/bin/env python


import cv2
if __name__ == '__main__':

    cv_half = cv2.imread('/home/bjoshi/Downloads/WreckHalf.png')
    cv_full = cv2.imread('/home/bjoshi/Downloads/WreckFull.png')
    cv_sliver = cv2.imread('/home/bjoshi/Downloads/WreckSliver.png')

    cv_half_gray = cv2.cvtColor(cv_half, cv2.COLOR_BGR2GRAY)
    cv_full_gray = cv2.cvtColor(cv_full, cv2.COLOR_BGR2GRAY)
    cv_sliver_gray = cv2.cvtColor(cv_sliver, cv2.COLOR_BGR2GRAY)

    brisk_detector = cv2.BRISK_create(thresh=20, octaves=1)

    kps_half = brisk_detector.detect(cv_half_gray, None)
    cv2.drawKeypoints(cv_half, kps_half, cv_half, color=(
        0, 0, 255), flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    kps_full = brisk_detector.detect(cv_full_gray, None)
    cv2.drawKeypoints(cv_full, kps_full, cv_full, color=(
        0, 0, 255), flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    kps_sliver = brisk_detector.detect(cv_sliver_gray, None)
    cv2.drawKeypoints(cv_sliver, kps_sliver, cv_sliver, color=(
        0, 0, 255), flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    # cv2.imshow('image', cv_half)
    # cv2.waitKey(0)

    # cv2.imshow('image1', cv_full)
    # cv2.waitKey(0)

    # cv2.imshow('image2', cv_sliver)
    # cv2.waitKey(0)

    # cv2.destroyAllWindows()

    cv2.imwrite('/home/bjoshi/Downloads/WreckHalf_BRISK.png', cv_half)
    cv2.imwrite('/home/bjoshi/Downloads/WreckFull_BRISK.png', cv_full)
    cv2.imwrite('/home/bjoshi/Downloads/WreckSliver_BRISK.png', cv_sliver)
