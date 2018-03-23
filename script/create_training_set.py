# python
import os

# ros
import rospy

# numpy
import numpy as np
from scipy.spatial import cKDTree

# self
from grasp_proxy import GraspProxy
from rviz_node import RvizNode
from plot_rviz import PlotRviz
import plot
from cloud_proxy import CloudProxy
import point_cloud_util
from hand_descriptor import HandDescriptor

# point_cloud
import point_cloud


workspace = [(-1.0, 1.0), (0, 1), (-0.50, 0.50)]
p = [-0.1, 0.75, -0.2]
grasp_space = [(p[0]-0.5, p[0]+0.5), (p[1]-0.5, p[1]), (p[2]-0.1, p[2]+0.05)]

grasp_offsets = [0.23, 0, 0.16]
grasp_offsets[1] = grasp_offsets[2] + (grasp_offsets[0]-grasp_offsets[2])/2.0

rospy.init_node("edge_grasp")
cloud_proxy = CloudProxy()
grasp_proxy = GraspProxy()
rviz_node = RvizNode()
plot_rviz = PlotRviz()
plot_rviz.initRos()

cloud = point_cloud.LoadPcd(os.path.dirname(__file__) + '/../data/cloud.pcd')
cloud = point_cloud_util.filter_workspace(workspace, cloud)

grasp_cloud = point_cloud_util.filter_workspace(grasp_space, cloud)
grasps = grasp_proxy.detect_grasps_whole_cloud(grasp_cloud, grasp_offsets)
grasps = filter(lambda g: g.bottom[1] < 0.55, grasps)
rviz_node.all_grasps_pub.publish(plot.createGraspsMarkerArray(grasps, rgba=[1, 0, 0, 0.5]))
rviz_node.grasp_space_pub.publish(plot.createGraspsPosCube(grasp_space))

array = []
for grasp in grasps:
    test_hand_des = HandDescriptor(HandDescriptor.T_from_approach_axis_center(grasp.approach, grasp.axis,
                                                                              (grasp.bottom+grasp.top)/2))
    test_hand_des.generate_depth_image(cloud)
    array.append(test_hand_des.image)
array = np.stack(array, 0)
f = os.path.dirname(__file__) + '/../data/train_test_edge_1.npy'
# old = np.loadtxt(f)
try:
    old = np.load(f)
    if old.shape[0] != 0:
        old = old.reshape([old.shape[0], 3, 60, 60])
        array = np.append(old, array)
finally:
    # np.savetxt(f, array.reshape([array.shape[0], -1]))
    np.save(f, array.reshape([array.shape[0], -1]))

aaa = 1