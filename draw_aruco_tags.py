import  cv2

total_items  = 9

aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_ARUCO_ORIGINAL)
aruco_params = cv2.aruco.DetectorParameters_create()

# img = cv2.aruco.drawMarker(aruco_dict, 582, 200)
# cv2.imshow("Aruco Tag", img)
# cv2.waitKey()
# cv2.destroyAllWindows()
for i in range(total_items):
    img = cv2.aruco.drawMarker(aruco_dict, i, 200)
    cv2.imwrite('marker{}.png'.format(i), img)