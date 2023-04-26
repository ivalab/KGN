import numpy as np
from copy import deepcopy

from pyassimp import *
import trimesh

import sys, os
root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if root_dir not in sys.path:
    sys.path.append(root_dir)

from utils.transform import create_homog_matrix 
from data_generation import Grasp

class Base():
    """The base class for the primitive shapes.

    Define the shared API names and functions
    """

    def __init__(self, color=np.random.choice(range(256), size=3).astype(np.uint8), pose=None,
        trl_sample_num=5, ang_sample_num=11) -> None:
        self.color = color

        # The file name to save out the model
        self.file_path = os.path.dirname(os.path.abspath(__file__))

        # create the mesh
        self.obj_mesh = self.generate_mesh()
        self.obj_mesh.visual.face_colors = self.color

        # store the pose
        self.pose=None
        self.set_pose(pose)

        # the grasp sample numbers
        self.trl_sample_num = trl_sample_num
        self.ang_sample_num = ang_sample_num

        # for storing the grasp infos
        self.grasp_poses = None
        self.open_widths = None
    
    def set_trl_sample_num(self, trl_sample_num):
        self.trl_sample_num = trl_sample_num
        # reset the grasp infos so that they will be regenerated if needed
        self.grasp_poses = None
        self.open_widths = None
    
    def set_ang_sample_num(self, ang_sample_num):
        self.ang_sample_num = ang_sample_num
        # reset the grasp infos so that they will be regenerated if needed
        self.grasp_poses = None
        self.open_widths = None

    def generate_mesh(self):
        raise NotImplementedError
        return obj_mesh
    
    def export_stl_obj(self, dir_name, file_name):
        """Export the mesh as a .stl and an .obj file

        Args:
            dir_name (str): The directory name to export the files
            file_name (str): The file name without the extenstion
        """
        stl_file = os.path.join(dir_name, file_name + ".stl")
        obj_file = os.path.join(dir_name, file_name + ".obj")

        # save out the stl
        self.obj_mesh.export(stl_file)

        # save out the obj
        scene=load(stl_file)
        export(scene, obj_file, file_type='obj')
        os.remove(obj_file+".mtl")


    def generate_grasp_family(self):
        """Generate the grasp family in the object frame. To be overwritten
        Returns:
            grasp_poses (N, 4, 4): an array of grasp homogeneous transformation matrices of the shape (N, 4, 4)
            open_widths (N, ): an array of grasp open width
        """
        raise NotImplementedError

    def set_pose(self, pose):
        """
        Set the pose of the grasp in the world frame

        Args:
            pose (array, (4,4)): A 4-by-4 homogeneous matrix. If set to None, will create identical transform matrix, which 
                align the gripper frame with the world frame
        """
        if pose is None:
           R = np.eye(3)
           T = np.zeros(3)
           pose = create_homog_matrix(R, T)
        
        self.pose = pose

        # reset the grasp info (especially grasp poses) so that they will be regenerated if needed
        self.grasp_poses = None
        self.open_widths = None
    
    def get_grasp_meshes(self, gripper_frame=False, distinct_color=False):
        """Get the gripper mesh in the world frame (with the set pose)

        Args:
            gripper_frame (bool, optional): Get the mesh for the gripper frame or not. Defaults to False.
        
        Returns:
            grasp_mesh_list (list): The mesh list for the grasp
        """
        grasp_mesh_list = []
        if self.grasp_poses is None:
            self.grasp_poses, self.open_widths = self.generate_grasp_family()

        for i in range(self.grasp_poses.shape[0]):
            T = self.grasp_poses[i, :, :]
            open_width = self.open_widths[i]
            if distinct_color:
                color = np.random.choice(range(256), size=4).astype(np.float32)
                color[-1] = 200.
                color = color / 255
            else:
                color = [0, 255, 0]
            grasp = Grasp(open_width=open_width, pose=(self.pose @ T), color=color)
            grasp_mesh = grasp.get_mesh(kpts=False, gripper_frame=gripper_frame)
            grasp_mesh_list = grasp_mesh_list + grasp_mesh
        
        return grasp_mesh_list
    
    def get_mesh(self, grasp=True, obj_frame=False, world_frame=False, gripper_frame=False, distinct_grasp_color=False):
        """Get the meshes for the object as a list, including the object and optionally 
        the, object_frame, or the world frame.

        Args:
            grasp (bool, optional): Option to get the grasp meshes. Defaults to True
            obj_frame (bool, optional): Option to get the object frame mesh. Defaults to False.
            world_frame (bool, optional): Option to get the world frame mesh. Defaults to False.

        Return:
            mesh_list (list): The list of the meshes
        """
        mesh_list = [self._get_obj_mesh_in_world()]

        # grasp meshes
        if (grasp is True):
            mesh_list = mesh_list + self.get_grasp_meshes(gripper_frame=gripper_frame, distinct_color=distinct_grasp_color)

        # world frame
        if world_frame:
            world_frame_mesh = trimesh.creation.axis(origin_size=0.002, axis_radius=0.002, axis_length=0.02)
            mesh_list.append(world_frame_mesh)

        # gripper frame
        if obj_frame:
            obj_frame_mesh = trimesh.creation.axis(transform=self.pose, origin_size=0.002, axis_radius=0.002, axis_length=0.01)
            mesh_list.append(obj_frame_mesh)


        return mesh_list
    
    def _get_obj_mesh_in_world(self):
        """Get the object mesh in world by applying the stored pose.

        The trimesh apply_transform will directly change the instance stored mesh pose, which means if called for multiple times the mesh
        would also change multiple times. 
        This function will not change the stored mesh.

        Returns:
            mesh_in_world:  The mesh in the world frame
        """ 
        mesh = deepcopy(self.obj_mesh)
        mesh.apply_transform(self.get_pose())
        return mesh
    
    def get_pose(self):
        """Get the obect pose in the world frame
        Args:
            pose (array, (4,4)): A 4-by-4 homogeneous matrix. 
        """
        return self.pose

    def get_grasp_infos(self, frame="world"):
        """Get the grasp poses and the open_width

        Args
            frame (str): The grasp pose in which frame. "world" or "object". Defaults to "world"
        
        Returns:
            grasp_poses (array. (N, 4, 4)): The homogeneous transformation of the grasp poses
            open_widths (array. (N,)): The open width
        """
        if self.grasp_poses is None:
            self.grasp_poses, self.open_widths = self.generate_grasp_family()

        if frame == "world":
            grasp_poses = self.pose[np.newaxis, :, :] @ self.grasp_poses
        elif frame == "object":
            grasp_poses = self.grasp_poses

        return grasp_poses, self.open_widths
    
    def get_obj_type(self):
        raise NotImplementedError
    
    def get_obj_dims(self):
        raise NotImplementedError

    def vis(self, grasp=True, obj_frame=False, world_frame=False, gripper_frame=False, distinct_grasp_color=False):
        """
        Visualize the grasp in the world frame
        Args:
            grasp (bool, optional): Option to visualize the grasp meshes. Defaults to True
            kpts (bool, optional): Visualize the keypoints or not. Defaults to False
            gripper_frame (bool, optional): Visualize the gripper frame or not. Defaults to False
            world_frame (bool, optional): Visualize the world frame or not. Defaults to False
        """
        # get the mesh
        mesh_list = self.get_mesh(grasp=grasp, obj_frame=obj_frame, world_frame=world_frame, gripper_frame=gripper_frame, distinct_grasp_color=distinct_grasp_color)
        mesh = trimesh.util.concatenate(mesh_list)
        
        # create the scene
        scene = trimesh.Scene(mesh)
        scene.show()