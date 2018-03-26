import numpy as np
from scipy.ndimage.morphology import grey_dilation
from scipy.spatial import cKDTree
import unittest

# self
import point_cloud_util


class HandDescriptor:
    def __init__(self, T):
        self.T = T

        self.depth = 0.06
        self.width = 0.105
        self.height = 0.02

        self.image = None

        self.image_pixel = 60
        self.image_depth = 0.1
        self.image_width = 0.1
        self.image_height = 0.1

        self.approach = T[0:3, 0]
        self.binormal = T[0:3, 1]
        self.axis = T[0:3, 2]
        self.center = T[0:3, 3]
        self.bottom = self.center - 0.5 * self.depth * self.approach
        self.top = self.center + 0.5 * self.depth * self.approach

    def compute_height_map(self, cloud, v_axis):
        """
        compute the height map over the given axis
        :param cloud: point cloud on hand frame, filtered
        :param v_axis: axis that orthogonal to height map
        :return: self.image_pixel x self.image_pixel height map
        """
        if v_axis == 0:
            xy_axis = (1, 2)
        elif v_axis == 1:
            xy_axis = (0, 2)
        elif v_axis == 2:
            xy_axis = (0, 1)
        else:
            raise Exception('unknown value axis, {}'.format(v_axis))

        d_max = max(self.image_depth, self.image_width, self.image_height)
        im = np.zeros((self.image_pixel, self.image_pixel), dtype='float32')

        coords = (cloud[:, xy_axis] + (d_max / 2.0)) * ((self.image_pixel - 1) / d_max)
        coords[coords < 0] = 0
        coords[coords > (self.image_pixel - 1)] = self.image_pixel - 1

        values = 1 - ((cloud[:, v_axis] + (d_max / 2.0)) / d_max)

        self.set_image_max_value(im, coords, values)
        return im

    def get_hand_points(self, cloud):
        """
        transform cloud into hand frame,
        then get points within the box range of self.(image_depth, image_width, image_height)
        :param cloud: original nx3 point cloud
        :return: point cloud after transformation and filtering
        """
        cloud_tree = cKDTree(cloud)
        search_radius = np.linalg.norm(np.array([self.image_depth, self.image_width, self.image_height]) / 2.0)
        indices = cloud_tree.query_ball_point(self.center, search_radius)
        points = cloud[indices, :]

        hTb = np.linalg.inv(self.T)
        workspace = [(-self.image_height/2, self.image_height/2),
                     (-self.image_width/2, self.image_width/2),
                     (-self.image_depth/2, self.image_depth/2)]
        points = point_cloud_util.transform(hTb, points)
        points = point_cloud_util.filter_workspace(workspace, points)

        return points

    def generate_depth_image(self, cloud):
        """
        compute the height map over 3 axises, do grey_dilation, assign depth image to self.image
        :param cloud: original nx3 point cloud
        """
        points = self.get_hand_points(cloud)

        im1 = self.compute_height_map(points, 2)
        im2 = self.compute_height_map(points, 1)
        im3 = self.compute_height_map(points, 0)

        im1 = grey_dilation(im1, size=3)
        im2 = grey_dilation(im2, size=3)
        im3 = grey_dilation(im3, size=3)

        self.image = np.stack((im1, im2, im3), 0)

    @staticmethod
    def T_from_approach_axis_center(approach, axis, center):
        """
        generate transformation matrix from base frame to hand frame bTh,
        based on approach, axis, and center of grasp
        :param approach: approach vector of grasp
        :param axis: axis vector of grasp
        :param center: (bottom + top) / 2
        :return: transformation matrix
        """
        T = np.eye(4)
        T[0:3, 0] = approach
        T[0:3, 1] = np.cross(approach, axis)
        T[0:3, 2] = axis
        T[0:3, 3] = center
        return T

    @staticmethod
    def set_image_max_value(image, coordinates, values):
        """
        set image based on value at coordinates, take the max if multiple values at same coordinate
        :param image: pxp np array
        :param coordinates: nx2 np array
        :param values: nx1 np array
        """
        for i in range(len(values)):
            row = int(coordinates[i][0])
            col = int(coordinates[i][1])
            image[row, col] = max(image[row, col], values[i])


def test():
    T = np.eye(4)
    descriptor = HandDescriptor(T)
    # cloud = np.random.rand(100, 3)
    cloud = np.array([[0, 0, 0], [0.1, 0, 0.1], [0.05, 0.05, 0.05]])
    im = descriptor.compute_height_map(cloud, 0)
    raw_input()


if __name__ == '__main__':
    test()