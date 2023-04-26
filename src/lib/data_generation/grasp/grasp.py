"""

The grasp class

"""

from copy import deepcopy
from math import sqrt
from itertools import combinations
import numpy as np
import trimesh

if __package__ is None:
    import sys, os
    root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    if root_dir not in sys.path:
        sys.path.append(root_dir)

from utils.transform import create_homog_matrix, apply_homog_transform
from grasp_kpts import GraspKpts3d, HedronVertexType, BoxVertexType, TailVertexType

class Grasp():
    """
    The grasp class
    """
    def __init__(self, open_width=0.08, color=[0, 255, 0], pose=None, kpts_option="hedron", kpts_color=[0, 0, 255], kpts_dist=0.1,
        gripper_tube_radius=0.0015) -> None:
        """Create a grasp instance, which stores the attributes of the grasp & gripper

        Args:
            open_width (float, optional): [description]. Defaults to 0.08.
            color (list, optional): [description]. Defaults to [0, 0, 255].
            pose (array, (4,4), optional): The initial pose. A 4-by-4 homogeneous matrix. Defaults to None, 
                which will align the gripper frame with the world frame
            kpts_option: (str, optional): The keypoint representation strategy. ["hedron", "lf_tail", "box"], defaults to "hedron"
            gripper_tube_radius (float, optional):      The tube radius of the gripper mesh. Defaults to 0.001
        """
        # parameter of the grasp
        self.set_pose(pose)
        self.open_width = open_width
        self.color = color
        self.kpts_color = kpts_color
        self.kpts_option = kpts_option
        self.kpts_dist = kpts_dist

        # representations
        self.tube_radius=gripper_tube_radius
        self.kpt_generator = GraspKpts3d(open_width=self.kpts_dist, kpt_type=self.kpts_option)
        self.kpts_3d = self.generate_3d_kpts()
        self.kpts_mesh = None

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

    def generate_3d_kpts(self):
        """
        Generate the location of the 3d keypoints in the gripper frame

        Args:
            option ("hedron", "box", "tail"): The keypoint strategy option
        Returns:
            kpts_3d (array, (N, 3)). The keypoint coordinate in the grasp frame 
        """
        kpts_3d = self.kpt_generator.get_local_vertices()
        return kpts_3d
        
    def generate_kpts_mesh(self, kpts_link=False) -> trimesh.Trimesh:
        """Generate the keypoints and the necessary keypoint connection mesh in the gripper frame

        Returns:
            kpts_mesh: The mesh for the keypoints
        """
        mesh_list = []
        # the keypoint meshes
        kpts_3d = self.get_kpts(frame="gripper")
        N, _ = kpts_3d.shape
        for i in range(N):
            trf_mat = create_homog_matrix(T_vec=kpts_3d[i, :])
            kpt_mesh = trimesh.creation.uv_sphere(
                radius=self.open_width/20,
                count=[10, 10]
            ).apply_transform(trf_mat)
            kpt_mesh.visual.face_colors = self.kpts_color
            mesh_list.append(kpt_mesh)

        # the connection mehses
        if kpts_link:
            assert N == 4
            if self.kpts_option == "hedron":
                kpts_link = list(combinations(range(4),2))
            elif self.kpts_option == "tail":
                kpts_link = [
                    [TailVertexType.Left, TailVertexType.Right],
                    [TailVertexType.Left, TailVertexType.Middle],
                    [TailVertexType.Right, TailVertexType.Middle],
                    [TailVertexType.Middle, TailVertexType.Tail]
                ]
            elif self.kpts_option == "box":
                kpts_link = [
                    [BoxVertexType.Left, BoxVertexType.Right],
                    [BoxVertexType.Left, BoxVertexType.TopLeft],
                    [BoxVertexType.Right, BoxVertexType.TopRight],
                    [BoxVertexType.TopLeft, BoxVertexType.TopRight]
                ]
            else:
                raise NotImplementedError
        
            for idx1, idx2 in kpts_link:
                mesh = trimesh.creation.cylinder(
                    radius = 0.001, 
                    sections=6,
                    segment=[
                        kpts_3d[idx1, :],
                        kpts_3d[idx2, :]
                    ]
                )
                mesh.visual.face_colors = self.kpts_color
                mesh_list.append(mesh)

        kpts_mesh = trimesh.util.concatenate(mesh_list) 
        return kpts_mesh


    def generate_gripper_mesh(self, open_width=0.08, color=[0, 0, 255], tube_radius=0.001, sections=6) -> trimesh.Trimesh:
        """Create a 3D mesh visualizing a parallel yaw gripper. 
        In the gripper frame it aligns with the X-Z plane and points towards the positive-X direction
        It consists of four cylinders.
        The code modified from: https://github.com/NVlabs/acronym/blob/d78431f06de3965de1b229510102d118ea8d9de1/acronym_tools/acronym.py#L404-L443 

        Args:
            color (list, optional): RGB values of marker. Defaults to [0, 0, 255].
            tube_radius (float, optional): Radius of cylinders. Defaults to 0.001.
            sections (int, optional): Number of sections of each cylinder. Defaults to 6.
        Returns:
            trimesh.Trimesh: A mesh that represents a simple parallel yaw gripper.
        """

        height = 0.2

        cfl = trimesh.creation.cylinder(
            radius = tube_radius,
            sections = sections,
            segment=[
                [0, 0, open_width/2],
                [-height/2, 0, open_width/2]
            ]
        )
        cfr = trimesh.creation.cylinder(
            radius = tube_radius,
            sections = sections,
            segment=[
                [0, 0, -open_width/2],
                [-height/2, 0, -open_width/2]
            ]
        )
        cb1 = trimesh.creation.cylinder(
            radius = tube_radius,
            sections = sections,
            segment=[
                [-height/2, 0, open_width/2],
                [-height/2, 0, -open_width/2]
            ]
        )
        cb2 = trimesh.creation.cylinder(
            radius = tube_radius,
            sections = sections,
            segment=[
                [-height, 0, 0],
                [-height/2, 0, 0]
            ]
        )

        tmp = trimesh.util.concatenate([cb1, cb2, cfr, cfl])
        tmp.visual.face_colors = color

        return tmp

    def get_pose(self):
        """Get the pose of the grasp in the world frame

        Return:
            pose (array, (4,4)). The 4-by-4 homogeneous transformation matrix
        """
        return self.pose 
    
    def get_mesh(self, kpts=False, kpts_link=False, gripper_frame=False, world_frame=False):
        """Get the meshes for the grasp as a list, including the gripper and optionally 
        the keypoints, gripper_frame, or the world frame.

        Args:
            kpts (bool, optional): Option to get the keypoint mesh. Defaults to False.
            gripper_frame (bool, optional): Option to get the gripper frame mesh. Defaults to False.
            world_frame (bool, optional): Option to get the world frame mesh. Defaults to False.
        
        Returns:
            mesh_list (List[trimesh.mesh]):     A list of meshes that can include the following depending on the arguments:
                                                    1.gripper mesh
                                                    2.kpts mesh
                                                    3.world frame mesh
                                                    4. gripper frame mesh 
        """
        gripper_mesh = self.generate_gripper_mesh(open_width = self.open_width, color= self.color, tube_radius=self.tube_radius)
        mesh_list = [gripper_mesh.apply_transform(self.pose)]

        # kpts
        if kpts:
            kpts_mesh = self.generate_kpts_mesh(kpts_link=kpts_link)
            mesh_list.append(kpts_mesh.apply_transform(self.pose))

        # world frame
        if world_frame:
            world_frame_mesh = trimesh.creation.axis(origin_size=0.002, axis_radius=0.001, axis_length=0.02)
            mesh_list.append(world_frame_mesh)

        # gripper frame
        if gripper_frame:
            gripper_frame_mesh = trimesh.creation.axis(transform=self.pose, origin_size=0.002, axis_radius=0.001, axis_length=0.02)
            mesh_list.append(gripper_frame_mesh)


        return mesh_list

    def get_kpts(self, frame="gripper"):
        """Get the 3d keypoints of the grasp in the world frame

        Args:
            frame (str, optional):  "gripper" or "world". The keypoint coordinates in which frame. Default to the gripper

        Returns:
            kpts (array (N, 3)). Each row is a keypoint 3d coordinate
        """
        if frame == "gripper":
            return self.kpts_3d
        elif frame == "world":
            return apply_homog_transform(self.kpts_3d, trf_mat=self.get_pose())
        else:
            raise NotImplementedError


    def vis(self, kpts=False, kpts_link=False, gripper_frame=False, world_frame=False):
        """
        Visualize the grasp in the world frame
        Args:
            kpts (bool, optional): Visualize the keypoints or not. Defaults to False
            gripper_frame (bool, optional): Visualize the gripper frame or not. Defaults to False
            world_frame (bool, optional): Visualize the world frame or not. Defaults to False
        """
        # get the mesh
        mesh_list = self.get_mesh(kpts=kpts, gripper_frame=gripper_frame, world_frame=world_frame, kpts_link=kpts_link)
        mesh = trimesh.util.concatenate(mesh_list)
        
        # create the scene
        scene = trimesh.Scene(mesh)
        scene.show()

if __name__ == "__main__":
    from utils.transform import create_rot_mat_axisAlign

    # visualization
    grasp = Grasp(open_width = 0.06, color=[0, 255, 0], pose=None, kpts_option="hedron", kpts_color=[255, 0, 255], kpts_dist=0.1)
    print(grasp.get_pose())
    grasp.vis(kpts=True, kpts_link=False, gripper_frame=True, world_frame=True)
    exit()

    # apply a transform - Move along positive x axis
    T1 = [0.2, 0 ,0]
    R1 = np.eye(3)
    transform = create_homog_matrix(R1, T1)
    grasp.set_pose(transform)
    print(grasp.get_pose())
    grasp.vis(kpts=True, gripper_frame=True, world_frame=True)

    # apply a second transform - Rotate along the z axis for 90 degrees then translate
    T2 = [0., 0.0 ,0]
    R2 = create_rot_mat_axisAlign([2, -1, 3])
    transform2 = create_homog_matrix(R2, T2)
    grasp.set_pose(transform @ transform2)
    print(grasp.get_pose())
    grasp.vis(kpts=True, gripper_frame=True, world_frame=True)
