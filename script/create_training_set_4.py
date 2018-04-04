# python
import os

# ros
import rospy

# numpy
import numpy as np

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


grasp_offsets = [0.23, 0, 0.16]
grasp_offsets[1] = grasp_offsets[2] + (grasp_offsets[0]-grasp_offsets[2])/2.0

rospy.init_node("edge_grasp")
cloud_proxy = CloudProxy()
grasp_proxy = GraspProxy()
rviz_node = RvizNode()
plot_rviz = PlotRviz()
plot_rviz.initRos()

cloud = point_cloud.LoadPcd(os.path.dirname(__file__) + '/../data/pcd/cloud4.pcd')


def create_data():
    grasps = grasp_proxy.detect_grasps_whole_cloud(cloud, grasp_offsets)

    print 'grasps after filter: ' + str(len(grasps))
    rviz_node.cloud_pub.publish(cloud_proxy.convert_to_point_cloud2(cloud))
    rviz_node.all_grasps_pub.publish(plot.createGraspsMarkerArray(grasps, rgba=[1, 0, 0, 0.5]))

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

    f_3d = os.path.dirname(__file__) + '/../data/npy/edge_table2_3d.npy'
    f_2d = os.path.dirname(__file__) + '/../data/npy/edge_table2_2d.npy'
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
    except:
        pass
    finally:
        i = raw_input('write?')
        if i == 'n':
            return
        np.save(f_3d, image_3d)
        print 'current 3d data: ' + str(image_3d.shape[0])
        np.save(f_2d, image_2d)
        print 'current 2d data: ' + str(image_2d.shape[0])


while True:
    create_data()