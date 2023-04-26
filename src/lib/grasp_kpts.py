# Copyright (c) 2021 NVIDIA Corporation. All rights reserved.
# This work is licensed under the NVIDIA Source Code License - Non-commercial.
# Full text can be found in LICENSE.md

# Code modified from: https://github.com/NVlabs/CenterPose/blob/main/src/lib/utils/pnp/cuboid_objectron.py

from enum import IntEnum
from itertools import combinations

import cv2
import numpy as np
from math import sqrt


# Related to the object's local coordinate system
# @unique
class HedronVertexType(IntEnum):
    Left = 0
    Right = 1
    TopFront = 2
    TopBack = 3
    TotalCornerVertexCount = 4  # Corner vertexes doesn't include the center point
    TotalVertexCount = 4

class BoxVertexType(IntEnum):
    Left = 0
    Right = 1
    TopLeft = 2
    TopRight = 3
    TotalCornerVertexCount = 4  # Corner vertexes doesn't include the center point
    TotalVertexCount = 4

class TailVertexType(IntEnum):
    Left = 0
    Right = 1
    Middle = 2
    Tail = 3
    TotalCornerVertexCount = 4  # Corner vertexes doesn't include the center point
    TotalVertexCount = 4

def generate_kpt_links(kpts_type):
    """ Generate a list of kpt index pairs indicating the links between two keypoints.
    Returns:
        links (list of the shape (N_link, 2))
    """

    if kpts_type == "hedron":
        kpts_link = list(combinations(range(4),2))
    elif kpts_type == "tail":
        kpts_link = [
            # [TailVertexType.Left, TailVertexType.Right],
            [TailVertexType.Left, TailVertexType.Middle],
            [TailVertexType.Right, TailVertexType.Middle],
            [TailVertexType.Middle, TailVertexType.Tail]
        ]
    elif kpts_type == "box":
        kpts_link = [
            # [BoxVertexType.Left, BoxVertexType.Right],
            [BoxVertexType.Left, BoxVertexType.TopLeft],
            [BoxVertexType.Right, BoxVertexType.TopRight],
            [BoxVertexType.TopLeft, BoxVertexType.TopRight]
        ]
    else:
        raise NotImplementedError
    
    return kpts_link
    


# ========================= GraspKpts3D =========================
class GraspKpts3d():
    '''This class contains the 3D grasp keypoints of a Grasp model.
    The local grasp coordinate system is defined as: 
        1. The Left gripper lives on the positive z axis, right on the negative z axis, and they are origin-symmetrical
        2. The reach direction pointing towards the positive x direction
    '''

    # the grasp kpt coordinates can be determined by the open width
    def __init__(self, open_width = 1, kpt_type="hedron"):
        # NOTE: This local coordinate system is similar
        # to the intrinsic transform matrix of a 3d object
        # self.center_location = [size3d[0]/2,size3d[1]/2,size3d[2]/2]
        self.open_width = open_width
        self.kpt_type = kpt_type
        self._vertices = None

        self.generate_local_vertices()

    def set_open_width(self, open_width, regenerate_vertices = True):
        self.open_width = open_width
        if regenerate_vertices:
            self.generate_local_vertices()

    def set_kpt_type(self, kpt_type, regenerate_vertices = True):
        self.kpt_type = kpt_type
        if regenerate_vertices:
            self.generate_local_vertices()

    def get_local_vertex(self, vertex_type):
        """Returns the location of a vertex.
        Args:
            vertex_type: enum of type any VertexType
        Returns:
            Numpy array(3) - Location of the vertex type in the grasp frame
        """
        return self._vertices[vertex_type]

    def get_local_vertices(self):
        """Get the grasp keypoint in the local frame

        Returns:
            vertices ((N_kpt, 3) array)
        """
        return self._vertices

    def generate_local_vertices(self):
        
        if self.kpt_type == "hedron":
            vertices = np.array([
                [0, 0, self.open_width/2],
                [0, 0, -self.open_width/2],
                [-sqrt(2)/2 * self.open_width, -self.open_width/2, 0],
                [-sqrt(2)/2 * self.open_width, self.open_width/2, 0] 
            ])
        elif self.kpt_type == "tail":
            vertices = np.array([
                [0, 0, self.open_width/2],
                [0, 0, -self.open_width/2],
                [-sqrt(3)/2 * self.open_width, 0, 0],
                [-(sqrt(2)/2 + 1) * self.open_width, 0, 0]
            ])
        elif self.kpt_type == "box":
            vertices = np.array([
                [0, 0, self.open_width/2],
                [0, 0, -self.open_width/2],
                [-self.open_width, 0, self.open_width/2],
                [-self.open_width, 0, -self.open_width/2]
            ])
        else:
            raise NotImplementedError("The kpt type: {} is not supported".format(self.kpt_type))

        # set the vertices
        self._vertices = vertices