import rospy
from nav_msgs.msg import Odometry

filename = "data/cave/stereo_imu/stereo_msckf1.txt"
topic_name = "/firefly_sbx/vio/odom"


def odom_callback(odom_msg: Odometry):
    global filename
    poistion = odom_msg.pose.pose.position
    orientation = odom_msg.pose.pose.orientation

    with open(filename, "a") as f:
        f.write(
            f"{odom_msg.header.stamp.secs}.{odom_msg.header.stamp.nsecs} {poistion.x} {poistion.y} {poistion.z} {orientation.x} {orientation.y} {orientation.z} {orientation.w}\n"
        )


if __name__ == "__main__":

    rospy.init_node("record_odom", anonymous=True)
    f = open(filename, "w")
    f.close()
    rospy.Subscriber(topic_name, Odometry, odom_callback)

    while not rospy.is_shutdown():
        rospy.spin()
