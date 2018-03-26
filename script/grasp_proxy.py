# python
import time

# numpy
import numpy as np

# ROS
import rospy
from sensor_msgs import point_cloud2
from sensor_msgs.msg import PointCloud2
from std_msgs.msg import Header, Int64

# GPD
from gpd.msg import GraspConfigList, CloudSamples, CloudSources

# self
from grasp import Grasp


class GraspProxy:
    def __init__(self):
        self.grasp_sub = rospy.Subscriber("/detect_grasps/clustered_grasps", GraspConfigList, self.callback)
        self.cloud_pub = rospy.Publisher("/cloud_stitched", PointCloud2, queue_size=1)
        self.wait_for_grasps = False
        self.grasps_msg = None

    def callback(self, msg):
        """
        callback function after receiving grasps,
        if waiting for grasps, set self.grasps_msg
        :param msg: GraspConfigList
        """
        if self.wait_for_grasps:
            self.grasps_msg = msg
            self.wait_for_grasps = False

    def detect_grasps_whole_cloud(self, cloud, offsets):
        """
        create point cloud message, publish to gpd, receive grasp message
        :param cloud: nx3 point cloud for detecting grasps
        :param offsets: grasp offsets
        :return: list[Grasp]
        """
        start_time = time.time()

        header = Header()
        header.stamp = rospy.Time.now()
        header.frame_id = "/base_link"
        cloud_source = point_cloud2.create_cloud_xyz32(header, cloud)

        print 'message processing time {} seconds'.format(time.time() - start_time)

        self.wait_for_grasps = True
        while self.wait_for_grasps:
            self.cloud_pub.publish(cloud_source)
            print 'published point cloud, waiting for grasps ...'
            rospy.sleep(0.5)

        print 'received', len(self.grasps_msg.grasps), 'grasps'
        return self.process_grasps(self.grasps_msg.grasps, offsets)

    @staticmethod
    def process_grasps(grasp_list, offsets):
        """
        translate grasp message into Grasp
        :param grasp_list: grasp message
        :param offsets: grasp offsets
        :return: list[Grasp]
        """
        grasps = []
        for grasp in grasp_list:
            top = np.array([grasp.top.x, grasp.top.y, grasp.top.z])
            bottom = np.array([grasp.bottom.x, grasp.bottom.y, grasp.bottom.z])
            axis = np.array([grasp.axis.x, grasp.axis.y, grasp.axis.z])
            approach = np.array([grasp.approach.x, grasp.approach.y, grasp.approach.z])
            binormal = np.array([grasp.binormal.x, grasp.binormal.y, grasp.binormal.z])
            width = grasp.width.data
            score = grasp.score.data

            grasp = Grasp(top, bottom, axis, approach, width, offsets, binormal, score)
            grasps.append(grasp)
        return grasps
