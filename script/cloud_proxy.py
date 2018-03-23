# numpy
import numpy as np

# ros
import tf
import rospy
import tf2_ros
from std_msgs.msg import Header
from sensor_msgs import point_cloud2
from sensor_msgs.msg import PointCloud2, PointField


class CloudProxy:
    def __init__(self):
        self.has_cloud = True
        self.active_cloud_msg = None
        self.active_cloud_sub = rospy.Subscriber("/camera/depth/points", PointCloud2, self.callback, queue_size=1)
        # self.cloud_pub = rospy.Publisher("/cloud_rviz", PointCloud2, queue_size=1)

    def callback(self, msg):
        """
        callback function after receiving cloud
        :param msg: PointCloud2
        """
        if not self.has_cloud:
            self.active_cloud_msg = msg
            self.has_cloud = True

    def get_cloud(self):
        """
        read point cloud from camera
        :return: nx3 point cloud
        """
        self.has_cloud = False
        while not self.has_cloud:
            rospy.sleep(0.01)

        # cloud_time = self.active_cloud_msg.header.stamp
        # cloud_frame = self.active_cloud_msg.header.frame_id
        cloud = np.array(list(point_cloud2.read_points(self.active_cloud_msg)))[:, 0:3]
        mask = np.logical_not(np.isnan(cloud).any(axis=1))
        cloud = cloud[mask]

        print 'received cloud with {} points.'.format(cloud.shape[0])
        return cloud

    @staticmethod
    def convert_to_point_cloud2(cloud):
        """
        convert point cloud into point_cloud2 msg
        :param cloud: nx3 point cloud
        :return: cloud into point_cloud2 msg
        """
        header = Header()
        header.frame_id = "base_link"
        header.stamp = rospy.Time.now()
        return point_cloud2.create_cloud_xyz32(header, cloud)
