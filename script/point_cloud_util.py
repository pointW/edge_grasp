import numpy as np


def transform(T, cloud):
    """
    applies homogeneous transform T to the cloud: y = Tx for each x in cloud
    :param T: 4x4 transformation matrix
    :param cloud: nx3 point cloud
    :return: nx3 point cloud after transformation
    """
    X = np.vstack((cloud.T, np.ones(cloud.shape[0])))
    X = np.dot(T, X).T
    X = X[:, 0:3]
    return X


def filter_workspace(workspace, cloud):
    """
    remove points outside workspace
    :param workspace: [(minX, maxX), (minY, maxY), (minZ, maxZ)]
    :param cloud: nx3 point cloud
    :return: mx3 point cloud after removal
    """
    mask = (((((cloud[:, 0] >= workspace[0][0]) & (cloud[:, 0] <= workspace[0][1]))
              & (cloud[:, 1] >= workspace[1][0])) & (cloud[:, 1] <= workspace[1][1]))
            & (cloud[:, 2] >= workspace[2][0])) & (cloud[:, 2] <= workspace[2][1])
    cloud = cloud[mask, :]
    return cloud
