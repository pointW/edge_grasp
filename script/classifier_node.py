import torch, cv2

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
from cnn import CNN

# point_cloud
import point_cloud

# pytorch
from torch.autograd import Variable


cloud_name = 'cloud.pcd'
cnn = torch.load(os.path.dirname(__file__) + '/../model/cnn')

workspace = [(-0.5, 1.0), (0, 1.5), (-0.50, 0.50)]

grasp_offsets = [0.23, 0, 0.16]
grasp_offsets[1] = grasp_offsets[2] + (grasp_offsets[0]-grasp_offsets[2])/2.0

rospy.init_node("edge_grasp")
cloud_proxy = CloudProxy()
grasp_proxy = GraspProxy()
rviz_node = RvizNode()
plot_rviz = PlotRviz()
plot_rviz.initRos()

cloud = point_cloud.LoadPcd(os.path.dirname(__file__) + '/../data/pcd/' + cloud_name)
cloud = point_cloud_util.filter_workspace(workspace, cloud)

grasps = grasp_proxy.detect_grasps_whole_cloud(cloud, grasp_offsets)
image_2d = []

for grasp in grasps:
    hand_des_2d = HandDescriptor(HandDescriptor.T_from_approach_axis_center(grasp.approach, grasp.axis,
                                                                            (grasp.bottom + grasp.top) / 2))
    hand_des_2d.generate_depth_image(cloud)
    image_2d.append(hand_des_2d.image)

cnn.eval()
image_2d = np.stack(image_2d, 0)
inputs = torch.FloatTensor(image_2d).cuda()
inputs = Variable(inputs, volatile=True)
outputs = cnn(inputs)
prediction = outputs.data.max(1, keepdim=True)[1]
prediction = prediction.squeeze().tolist()
edge_grasp = []
non_edge_grasp = []
for i in range(len(grasps)):
    if prediction[i] == 1:
        edge_grasp.append(grasps[i])
    else:
        non_edge_grasp.append(grasps[i])
rviz_node.edge_grasp_pub.publish(plot.createGraspsMarkerArray(edge_grasp, rgba=[1, 0, 0, 0.5]))
rviz_node.non_edge_grasp_pub.publish(plot.createGraspsMarkerArray(non_edge_grasp, rgba=[0, 0, 1, 0.5]))
rviz_node.cloud_pub.publish(cloud_proxy.convert_to_point_cloud2(cloud))
