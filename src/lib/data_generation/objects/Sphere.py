import numpy as np
from numpy import pi, sin, cos
from pyassimp import *
import os
from torch import dist

import trimesh
import sys, os

root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if root_dir not in sys.path:
    sys.path.append(root_dir)

from data_generation import Base
from utils.transform import create_homog_matrix, create_rot_mat_axisAlign

class Sphere(Base):
    """The Sphere class
    The inner and outer radius ratio if fixed to be 1.15 following the PSCNN
    The cylinder frame origin 
    """
    def __init__(self, r, semiSphere=False, pose=None, color=np.random.choice(range(256), size=3).astype(np.uint8), \
        ang_sample_num=11):
        """Generate a Cuboid instance

        Args:
            r (float): The radius of the sphere in m. If semiSphere, then this is the outer radius
            semiSphere (bool, option): The option to generate a semi-sphere instead of a sphere. Defaults to false
            pose (array (4,4), optional): The initial 4-by-4 homogenous transformation matrix of cuboid-to-world. Defaults to None (align with the world frame)
        """

        # set the size cm to m
        self.r = r    # this is the r_out for the semiSphere

        # store the option
        self.semiSphere = semiSphere
        if self.semiSphere:
            self.r_in = self.r / 1.15

        # create the instance
        super().__init__(pose=pose, color=color, ang_sample_num=ang_sample_num)

    def generate_mesh(self):
        """The PSCNN code (generate_mesh_vertices) seems not creating the inner surface, making the mesh incomplete.
        So temporarily overwrite it.
        """
        if not self.semiSphere:
            # The whole sphere is okay
            obj_mesh = trimesh.primitives.Sphere(radius=self.r, center=[0, 0, 0])
        else:
            # use the revolve to generate hollow semi-sphere
            # create a 2d shape (on the y-z plane) to be revolved along the z axis.
            # NOTE: it needs to be COUNTERCLOCK-WISE according the the doc of revolve
            N = 32
            linestring = np.zeros((2*N+1, 2))
            idx = 0
            # outer
            for theta in np.linspace(-np.pi/2, 0, N):
                linestring[idx, :] = [self.r * cos(theta), self.r * sin(theta)]
                idx += 1
            # inner
            for theta in np.linspace(0, -np.pi/2, N):
                linestring[idx, :] = [self.r_in * cos(theta), self.r_in * sin(theta)]
                idx += 1
            # seal
            assert idx == 2*N
            linestring[-1, :] = linestring[0, :]
            # revolve
            obj_mesh = trimesh.creation.revolve(linestring=linestring)

        return obj_mesh

    def generate_grasp_family(self):
        ang_sample_num = self.ang_sample_num
        if not self.semiSphere:
            grasp_poses, open_widths = self._generate_grasp_sphere(ang_sample_num=ang_sample_num)
        else:
            grasp_poses, open_widths = self._generate_grasp_semi(ang_sample_num=ang_sample_num)
        return np.array(grasp_poses), np.array(open_widths) 
    
    def _generate_grasp_sphere(self, ang_sample_num=11):
        """Generate the grasp poses for the sphere

        Args:
            ang_sample_num (int, optional): [description]. Defaults to 11.

        Returns:
            [type]: [description]
        """
        grasp_poses = []
        open_width = 1.2 * 2 * self.r   # all grasps share the same open width
        # from top
        rot_top = create_rot_mat_axisAlign([-3, 2, 1])
        trf_top = create_homog_matrix(R_mat=rot_top)

        # from side
        rot_side = create_rot_mat_axisAlign([1, -3, 2])
        trf_side = create_homog_matrix(R_mat = rot_side)

        # sample the rotational parameters
        angles_sample = np.linspace(0, 2*pi, ang_sample_num)
        for trf in [trf_top, trf_side]:
            for theta in angles_sample:
                rot =   [[cos(theta), -sin(theta), 0],
                        [sin(theta), cos(theta), 0],
                        [0, 0, 1]]
                trf_sample = create_homog_matrix(R_mat=rot)
                grasp_poses.append(trf_sample @ trf)

        open_widths = [open_width] * len(grasp_poses)
    
        return grasp_poses, open_widths

    def _generate_grasp_semi(self, ang_sample_num=11, sink_in_dist=0.002):
        """
        Generate the grasp family for the semi-sphere
        """
        grasp_poses = []
        open_width = 5 * (self.r - self.r_in)

        # from top
        rot_top = create_rot_mat_axisAlign([-3, 2, 1])
        trl_top = [0, 0, -sink_in_dist]
        trf_top = create_homog_matrix(R_mat=rot_top, T_vec=trl_top)

        # sample the rotational parameters. Translate then rotate
        angles_sample = np.linspace(0, 2*pi, ang_sample_num)
        trl_homog = create_homog_matrix(T_vec=[(self.r + self.r_in)/2, 0, 0])
        for theta in angles_sample:
            rot =   [[cos(theta), -sin(theta), 0],
                    [sin(theta), cos(theta), 0],
                    [0, 0, 1]]
            rot_sample = create_homog_matrix(R_mat=rot)
            grasp_poses.append(rot_sample @ trl_homog @ trf_top)

        open_widths = [open_width] * len(grasp_poses)

        return grasp_poses, open_widths

    def get_obj_type(self):
        if self.semiSphere:
            obj_type = "semi_sphere"
        else:
            obj_type = "sphere"

        return obj_type
    
    def get_obj_dims(self):
        """
        The sphere or semisphere is quantified by a single parameter outer radius r
        The inner radius is r times a fixed scaling factor.
        """
        return self.r
    
    @staticmethod
    def construct_obj_info(obj_dims, obj_pose, mode, rot_sample_num=11, **kwargs):
        r = obj_dims

        if mode == "sphere":
            semi_sphere = False
        elif mode == "semi_sphere":
            semi_sphere = True
        else:
            raise NotImplementedError

        return Sphere(r=r, semiSphere=semi_sphere, pose=obj_pose, ang_sample_num=rot_sample_num, **kwargs)



if __name__ == "__main__":
    np.random.seed(10)
    sphere = Sphere(0.05, False, color=np.random.choice(range(256), size=3).astype(np.uint8))
    sphere.vis(True, False, distinct_grasp_color=True)

    semi_sphere = Sphere(0.1, True, color=np.random.choice(range(256), size=3).astype(np.uint8))
    semi_sphere.vis(True, False, distinct_grasp_color=True)