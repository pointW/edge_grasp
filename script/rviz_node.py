# ROS
import rospy
from sensor_msgs.msg import PointCloud2 as PointCloud2Msg, PointField
from visualization_msgs.msg import Marker, MarkerArray
import tf2_ros


class RvizNode:
    def __init__(self):
        print 'initializing rviz node'
        self.cloud_pub = rospy.Publisher("/cloud_rviz", PointCloud2Msg, queue_size=1)
        self.all_grasps_pub = rospy.Publisher("/all_grasps", MarkerArray, queue_size=1)
        self.grasp_space_pub = rospy.Publisher("/grasp_space", Marker, queue_size=1)
        # create TF listener to receive transforms
        # self.tfBuffer = tf2_ros.Buffer()
        # self.listener = tf2_ros.TransformListener(self.tfBuffer)
        # self.tfBuffer.lookup_transform("base_link", "ee_link", rospy.Time(0), rospy.Duration(4.0))
