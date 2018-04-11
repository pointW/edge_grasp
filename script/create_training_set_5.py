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


# cloud | single object
workspace = [(-0.5, 1.0), (0, 1.5), (-0.50, 0.50)]
p = [-0.09, 0.61, -0.15]
grasp_space = [(p[0]-0.05, p[0]+0.05), (p[1]-0.05, p[1]+0.05), (p[2]-0.08, p[2]+0.05)]


grasp_offsets = [0.23, 0, 0.16]
grasp_offsets[1] = grasp_offsets[2] + (grasp_offsets[0]-grasp_offsets[2])/2.0

rospy.init_node("edge_grasp")
cloud_proxy = CloudProxy()
grasp_proxy = GraspProxy()
rviz_node = RvizNode()
plot_rviz = PlotRviz()
plot_rviz.initRos()

cloud = point_cloud.LoadPcd(os.path.dirname(__file__) + '/../data/pcd/cloud.pcd')
cloud = point_cloud_util.filter_workspace(workspace, cloud)


def create_data():
    grasps = grasp_proxy.detect_grasps_whole_cloud(cloud, grasp_offsets)
    # cloud 2
    # grasps = filter(lambda g: (grasp_space[0][0] < g.bottom[0] < grasp_space[0][1] and
    #                            grasp_space[1][0] < g.bottom[1] < grasp_space[1][1] and
    #                            grasp_space[2][0] < g.bottom[2] < grasp_space[2][1]),
    #                 grasps)

    print 'grasps after filter: ' + str(len(grasps))
    rviz_node.cloud_pub.publish(cloud_proxy.convert_to_point_cloud2(cloud))
    rviz_node.all_grasps_pub.publish(plot.createGraspsMarkerArray(grasps, rgba=[1, 0, 0, 0.5]))
    # rviz_node.grasp_space_pub.publish(plot.createGraspsPosCube(grasp_space))

    image_3d = []
    image_2d = []
    for grasp in grasps:
        hand_des_3d = HandDescriptor(HandDescriptor.T_from_approach_axis_center(grasp.approach, grasp.axis,
                                                                                (grasp.bottom+grasp.top)/2),
                                     32)
        hand_des_3d.generate_3d_binary_image(cloud)
        image_3d.append(hand_des_3d.image)

        hand_des_2d = HandDescriptor(HandDescriptor.T_from_approach_axis_center(grasp.approach, grasp.axis,
                                                                                (grasp.bottom+grasp.top)/2))
        hand_des_2d.generate_depth_image(cloud)
        image_2d.append(hand_des_2d.image)

    image_3d = np.stack(image_3d, 0)
    image_2d = np.stack(image_2d, 0)

    f_3d = os.path.dirname(__file__) + '/../data/edge_3d.npy'
    f_2d = os.path.dirname(__file__) + '/../data/edge_2d.npy'
    try:
        old_3d = np.load(f_3d)
        if old_3d.shape[0] != 0:
            image_3d = np.vstack([image_3d, old_3d])
    except:
        pass
    try:
        old_2d = np.load(f_2d)
        if old_2d.shape[0] != 0:
            image_2d = np.vstack([image_2d, old_2d])
    finally:
        raw_input('write?')
        np.save(f_3d, image_3d)
        print 'current 3d data: ' + str(image_3d.shape[0])
        np.save(f_2d, image_2d)
        print 'current 2d data: ' + str(image_2d.shape[0])


while True:
    create_data()