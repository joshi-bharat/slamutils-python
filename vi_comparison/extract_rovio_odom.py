import rosbag

bag_file = "/home/bjoshi/Downloads/rovio/2022-12-12-17-04-16_25_4_6_1_0.bag"
traj_file = (
    "/home/bjoshi/code/slamutils-python/vi_comparison/data/cave/mono_imu/rovio.txt"
)

with rosbag.Bag(bag_file) as bag:
    with open(traj_file, "w") as f:
        for topic, msg, t in bag.read_messages(topics=["/rovio/odometry"]):
            position = msg.pose.pose.position
            orientation = msg.pose.pose.orientation
            f.write(
                f"{msg.header.stamp.to_sec()} {position.x} {position.y} {position.z} {orientation.x} {orientation.y} {orientation.z} {orientation.w}\n"
            )
