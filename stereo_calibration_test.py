import cv2
import yaml
import numpy as np


def read_kalibr_file(filename):
    with open(filename, 'r') as f:
        data = yaml.load(f, Loader=yaml.FullLoader)

    K1 = data['cam0']['intrinsics']
    K1 = np.array([[K1[0], 0.0, K1[2]],
                   [0.0, K1[1], K1[3]],
                   [0.0, 0.0, 1.0]])
    K2 = data['cam1']['intrinsics']
    K2 = np.array([[K2[0], 0.0, K2[2]],
                   [0.0, K2[1], K2[3]],
                   [0.0, 0.0, 1.0]])

    d1 = np.array(data['cam0']['distortion_coeffs'])
    d2 = np.array(data['cam1']['distortion_coeffs'])
    size = data['cam0']['resolution']
    trans_c1_c0 = np.array(data['cam1']['T_cn_cnm1'])
    R_c1_c0 = np.array(trans_c1_c0[0:3, 0:3])
    T_c1_c0 = np.array(trans_c1_c0[0:3, 3])
    distortion_type = data['cam0']['distortion_model']

    # print('K1: ', K1)
    # print('K2: ', K2)
    # print('d1: ', d1)
    # print('d2: ', d2)
    # print('size: ', size)
    # print('T_c1_c0: ', T_c1_c0)
    # print('R_c1_c0: ', R_c1_c0)

    return K1, K2, d1, d2, distortion_type, size, T_c1_c0, R_c1_c0


calib_file = './stereo_rig_calib.yaml'
left = cv2.imread('./left_new.png')
right = cv2.imread('./right_new.png')

K1, K2, d1, d2, distortion_type, size, T_c1_c0, R_c1_c0 = read_kalibr_file(
    calib_file)


result = None

size = (1600, 1200)
if distortion_type == 'radtan':
    result = cv2.stereoRectify(cameraMatrix1=K1, cameraMatrix2=K2,  distCoeffs1=d1,
                               distCoeffs2=d2, imageSize=size,  R=R_c1_c0, T=T_c1_c0,
                               flags=cv2.CALIB_ZERO_DISPARITY, alpha=0)

else:
    result = cv2.fisheye.stereoRectify(cameraMatrix1=K1, cameraMatrix2=K2,  distCoeffs1=d1,
                                       distCoeffs2=d2, imageSize=size,  R=R_c1_c0, T=T_c1_c0,
                                       flags=cv2.CALIB_ZERO_DISPARITY, alpha=0)

R1 = result[0]
R2 = result[1]
P1 = result[2]
P2 = result[3]
Q = result[4]

print('R1: ', R1)
print('R2: ', R2)
print('P1: ', P1)
print('P2: ', P2)

baseline = 1.0 / Q[3, 2]
fx = P1[0, 0]
bf = baseline * fx
print('baseline: ', baseline)

mapx_1, mapy_1 = cv2.initUndistortRectifyMap(
    cameraMatrix=K1, distCoeffs=d1, R=R1, newCameraMatrix=P1, size=size, m1type=cv2.CV_32FC1)
mapx_2, mapy_2 = cv2.initUndistortRectifyMap(
    cameraMatrix=K2, distCoeffs=d2, R=R2, newCameraMatrix=P2, size=size,  m1type=cv2.CV_32FC1)


left_rectified = cv2.remap(left, mapx_1, mapy_1, cv2.INTER_LINEAR)
right_rectified = cv2.remap(right, mapx_2, mapy_2, cv2.INTER_LINEAR)


left_rectified = cv2.resize(left_rectified, (800, 600),
                            interpolation=cv2.INTER_LINEAR)
right_rectified = cv2.resize(right_rectified, (800, 600),
                             interpolation=cv2.INTER_LINEAR)

# left = cv2.resize(left, (800, 600),
#                   interpolation=cv2.INTER_LINEAR)
# right = cv2.resize(right, (800, 600),
#                    interpolation=cv2.INTER_LINEAR)
concat = cv2.hconcat([left_rectified, right_rectified])

for i in range(120):
    start = (0, i * 20)
    end = (3200, i * 20)
    cv2.line(concat, start, end, (0, 0, 255), 1)

cv2.imshow('concat', concat)
cv2.waitKey(0)
# cv2.imshow('right', right_rectified)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
