import numpy as np
from numpy import pi, sin, cos
from pyassimp import *
# Basic Configuration
import os


import sys, os
root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if root_dir not in sys.path:
    sys.path.append(root_dir)

import trimesh
from data_generation import Base
from utils.transform import create_homog_matrix, create_rot_mat_axisAlign

class Cylinder(Base):
    """The Cylinder class
    The inner and outer radius ratio if fixed to be 1.15 following the PSCNN
    The rotation axis of the cylinder aligns with the z axis
    """
    def __init__(self, r_in, height, mode="cylinder", color=np.random.choice(range(256), size=3).astype(np.uint8), \
        pose=None, trl_sample_num=5, ang_sample_num=11) -> None:
        """Generate a Cylinder instance

        Args:
            r_in (float): The inner radius in m
            height (float): The height of the cylinder in m
            mode (str): "cylinder" or "stick" or "ring". Will influence on the grasp family generation
            color (array (3,)): The color of the object. Defaults to random color
            pose (array (4,4), optional): The initial 4-by-4 homogenous transformation matrix of cuboid-to-world. Defaults to None (align with the world frame)
        """

        # set the size cm to m
        self.r_in = r_in
        self.r_out = self.r_in * 1.15
        self.height = height

        # store the mode
        self.mode = mode

        # create the instance
        super().__init__(pose=pose, color=color, trl_sample_num=trl_sample_num, ang_sample_num=ang_sample_num)
    
    def generate_mesh(self):
        """The PSCNN code (gneerate_mesh_vertices) seems not creating the inner surface, making the mesh incomplete.
        So temporarily overwrite it.
        """
        if self.mode == "stick":
            obj_mesh = trimesh.creation.cylinder(self.r_in, self.height)
        else:
            obj_mesh = trimesh.creation.annulus(self.r_in, self.r_out, self.height)
        
        return obj_mesh
    
    def generate_grasp_family(self):

        # parameters
        ang_sample_num = self.ang_sample_num
        trl_sample_num = self.trl_sample_num
        sink_in_dist = 0.005

        # result storage
        grasp_poses = []
        open_widths = []

        # grasp family from top & bottom
        if self.mode != "stick":
            grasp_poses_TB, open_width_TB = self._generate_grasp_top_bottom(ang_sample_num=ang_sample_num, sink_in_dist=sink_in_dist)
            grasp_poses = grasp_poses + grasp_poses_TB
            open_widths = open_widths + [open_width_TB] * len(grasp_poses_TB)

        # grasp family from the side
        if self.mode != "ring":
            grasp_poses_side, open_width_side = self._generate_grasp_side(ang_sample_num=ang_sample_num, trl_sample_num=trl_sample_num)
            grasp_poses = grasp_poses + grasp_poses_side
            open_widths = open_widths + [open_width_side] * len(grasp_poses_side)

        return np.array(grasp_poses), np.array(open_widths)
    
    def _generate_grasp_top_bottom(self, ang_sample_num=11, sink_in_dist=0.02):
        """The grasp family from top and bottom
        Free parameter is the rotation along the z axis (height direction)

        Return:
            grasp_poses (list).  A list of 4-by-4 grasp poses
            open_width (float).  The open width. All the grasp poses in this category share the same open width
        """
        open_width = 5 * (self.r_out - self.r_in)

        # transformation to the top & bottom center
        R_top = create_rot_mat_axisAlign([-3, 2, 1])
        T_top = [0, 0, self.height/2 - sink_in_dist]
        trf_top = create_homog_matrix(R_top, T_top)
        R_bottom = create_rot_mat_axisAlign([3, -2, 1])
        T_bottom = [0, 0, - self.height/2 + sink_in_dist]
        trf_bottom = create_homog_matrix(R_bottom, T_bottom)
        
        # sample the rotational free parameter
        grasp_poses = []
        angle_samples = np.linspace(0, 2*pi, ang_sample_num)
        for trf in [trf_top, trf_bottom]:
            # first translate then rotate along z axis
            trl = [(self.r_in + self.r_out)/2, 0, 0]
            trl_homog = create_homog_matrix(T_vec=trl)
            for theta in angle_samples:
                rot =   [[cos(theta), -sin(theta), 0],
                        [sin(theta), cos(theta), 0],
                        [0, 0, 1]]
                rot_homog = create_homog_matrix(R_mat=rot) 
                grasp_poses.append(rot_homog@trl_homog@trf)

        return grasp_poses, open_width
    
    def _generate_grasp_side(self, ang_sample_num=11, trl_sample_num=5):
        """The grasp family on the side.
        Free parameters involve the rotation and translation along the z axis (height direction)

        Return:
            grasp_poses (list).  A list of 4-by-4 grasp poses
            open_width (float).  The open width. All the grasp poses in this category share the same open width
        """
        open_width = 1.2 * (2 * self.r_out)

        # transform to side 
        R = create_rot_mat_axisAlign([1, -3, 2])
        trf = create_homog_matrix(R)

        # sample the free parameter
        grasp_poses = []
        angle_samples = np.linspace(0, 2*pi, ang_sample_num)
        trl_samples = np.linspace(-self.height/2, self.height/2, trl_sample_num+2)
        trl_samples = trl_samples[1:-1] # remove the first & last element
        for theta in angle_samples:
            rot =   [[cos(theta), -sin(theta), 0],
                    [sin(theta), cos(theta), 0],
                    [0, 0, 1]]
            for trl in trl_samples:
                trl_vec = [0, 0, trl]
                trf_sample = create_homog_matrix(rot, trl_vec)
                grasp_poses.append(trf_sample @ trf)
        
        return grasp_poses, open_width

    
    def get_obj_type(self):
        return self.mode


    def get_obj_dims(self):
        """
        (r_in, height)
        """
        return np.array([self.r_in, self.height])
    

    @staticmethod
    def construct_obj_info(obj_dims, obj_pose, mode, trl_sample_num=5, rot_sample_num=11, **kwargs):
        r_in = obj_dims[0]
        h = obj_dims[1]

        assert mode in ["cylinder", "stick", "ring"]

        return Cylinder(r_in=r_in, height=h, mode=mode, pose=obj_pose, 
                        ang_sample_num=rot_sample_num, trl_sample_num=trl_sample_num, **kwargs)


if __name__ == "__main__":
    np.random.seed(10)
    cylinder = Cylinder(0.05, 0.08, color=np.random.choice(range(256), size=3).astype(np.uint8))
    cylinder.vis(True, False, False, False, distinct_grasp_color=True)

    ring = Cylinder(0.07, 0.03, mode="ring", color=np.random.choice(range(256), size=3).astype(np.uint8))
    ring.vis(True, False, False, False, distinct_grasp_color=True)

    stick = Cylinder(0.01, 0.20, mode="stick", color=[0.5, 0.5, 0.5, 0.5])
    # create a random pose
    stick.set_pose(create_homog_matrix(
        R_mat=np.eye(3),
        T_vec=np.array([0.2, 0.1, 0])
    ))
    stick.vis(True, False, False, False, distinct_grasp_color=True)
