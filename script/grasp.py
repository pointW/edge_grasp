import numpy as np


class Grasp:
    def __init__(self, top, bottom, axis, approach, width, offsets, binormal=None, score=None):
        self.top = top
        self.bottom = bottom
        self.axis = axis
        self.approach = approach
        self.width = width
        self.offsets = offsets

        self.binormal = np.cross(approach, axis) if binormal is None else binormal
        self.score = 0 if score is None else score
        self.finger_normals = (self.binormal, -self.binormal)

        self.poses = []
        for offset in offsets:
            T = np.eye(4)
            T[0:3, 0] = approach
            T[0:3, 1] = binormal
            T[0:3, 2] = axis
            T[0:3, 3] = top - offset * approach
            self.poses.append(T)