"""

The class to create the scene with objects, table, and the camera.
It seems that the trimesh camera won't be able to render the image, so will use the pyrender scene for the data rendering

"""
from typing import List
from copy import deepcopy
import numpy as np
import pyrender
from torch import sin
import trimesh
from scipy.spatial.transform import Rotation as R

import sys, os

root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if root_dir not in sys.path:
    sys.path.append(root_dir)

from utils.transform import create_homog_matrix 
from utils.utils import NoStablePoseError
from data_generation import Grasp, Base, Cuboid, Cylinder, Sphere

class Scene():
    def __init__(self, table_size=1, table_thickness=0.04, table_color=[0.5, 0.5, 0.5, 1], grasp_color=None)-> None:
        """Create an on-the-table scene.
        The table is a large short cuboid who lives in the z<0 region and whose support surface aligns with the z=0 plane

        Args:
            table_size (int, optional): The square table side length in meter. Defaults to 1.
            table_thickness (int, optional): The table thickness in meter. Defaults to 0.04.
            table_color (list, optional): [description]. Defaults to [0.5, 0.5, 0.5, 1].
            grasp_color (array (3)): The color for the grasp. If None, will set the non-colliding grasps to green, and colliding grasps to red
        """
        self.table_size = table_size
        self.table_thickness = table_thickness
        self.table_color = table_color
        self.table = self.create_table()

        self.objects = []

        self.grasp_poses = np.array([])
        self.grasp_widths = np.array([])
        self.grasp_collide = np.array([])
        self.grasp_analyzed = False
        self.grasp_color = grasp_color
        self.collision_manager_obj = trimesh.collision.CollisionManager()
        self.collision_manager_table = trimesh.collision.CollisionManager()
        self.collision_manager_table.add_object(
           name = "table",
           mesh = trimesh.util.concatenate(
               self.table.get_mesh(False, False, False, False)
           ),
           transform=None
        )


    def create_table(self):
        """
        Will create a cuboid table
        """
        x_size = self.table_size
        y_size = self.table_size
        z_size = self.table_thickness  # thickness
        return Cuboid(x_size, y_size, z_size, \
            color=np.array(self.table_color),
            pose=create_homog_matrix(T_vec=[0, 0, -z_size/2]))  # To make the upper surface align with the world x-y frame

    def add_objs(self, objs: List[Base], sample_pose=False, resample_xy_loc=False):
        """Add multiple objects. Offer options to sample the pose or only sample the 
        x-y locations. If both set to False, will use the pose stored in the obj instances.

        Args:
            objs (List[Base]): [description]
            sample_pose (bool or List[bool], optional): [description]. Defaults to False.
            resample_xy_loc (bool or List[bool], optional): [description]. Defaults to False.
        """
        N_obj = len(objs)
        if isinstance(sample_pose, bool):
            sample_pose = [sample_pose] * N_obj
        if isinstance(resample_xy_loc, bool):
            resample_xy_loc = [resample_xy_loc] * N_obj

        for obj, pose_sample, xy_resample in zip(objs, sample_pose, resample_xy_loc):
            try:
                self.add_obj(obj, sample_pose=pose_sample, resample_xy_loc=xy_resample)
            except NoStablePoseError:
                continue
    
    def add_obj(self, object:Base, sample_pose=False, resample_xy_loc=False):
        """Add an object to the scene

        Args:
            object (Base): The object instance
            sample_pose (bool, optional): If True, will sample a pose for this object, meaning the stored pose in the object will be overwritten. \
                Else will use the object's pose. Defaults to False.
                Note that is sample_pose is False, the collision-check will not be performed.
            resample_xy_loc (bool, optional): If True, will only re-sample the x-y location and keep the original orientation. \
                will be ignored if the sample_pose is set to True
        
        Raises
            RuntimeError: no successful non-collision pose can be sampled. 
        """

        # sample the pose
        pose, succ = self._sample_obj_pose(object, sample_pose=sample_pose, resample_xy_loc=resample_xy_loc)
        if not succ and not (succ is None):
            raise NoStablePoseError("No stable pose can be generated for this object.")
        
        object.set_pose(pose)

        if object.get_obj_type() == "stick":
            object = self._rectify_stick_pose(object)

        # add to the list
        self.objects.append(object)

        # add to the collision manager, named by "obj_" + index number
        self.collision_manager_obj.add_object(
            name = "obj_"+str(len(self.objects) - 1),
            mesh = trimesh.util.concatenate(object.get_mesh(grasp=False, obj_frame=False, world_frame=False, gripper_frame=False)),
            transform = None
        )

        # any new object added means the grasp needs to be re-analyzed
        self.grasp_analyzed = False
    
    def _rectify_stick_pose(self, stick:Cylinder, vis=False):
        """If the stick lies on the table, then rectify the pose so that one of the discritized grasp family
        is perpendicular to the table

        Args:
            stick: stick type object
        Returns:
            stick: stick type object with the rectified pose
        """
        stick_pose = stick.get_pose()
        R_orig = stick_pose[:3, :3]
        T_orig = stick_pose[:3, 3]
        stick_grasp_poses, _ = stick.get_grasp_infos()
        # get the transformed z direction
        x_dir = stick_pose[:3, 0]
        y_dir = stick_pose[:3, 1]
        z_dir = stick_pose[:3, 2]
        obj_x_tar = np.array([0, 0, 1])

        if np.abs(z_dir @ np.array([0, 0, 1])) < 0.1:
            # Lying on the table. Need to rectify the pose
            # Just align the object x direction to the negative z given how the grasp is generated 
            grasp_pose_target = stick_grasp_poses[0, :, :]
            grasp_x_cur = grasp_pose_target[:3, 0]
            grasp_y_cur = grasp_pose_target[:3, 1]
            grasp_x_tar = np.array([0, 0, -1])
            #y_tar = - np.cross(x_tar, z_dir)
            # calculate the rectify transformation matrix
            cos_theta = np.dot(grasp_x_tar, grasp_x_cur)
            sin_theta = np.dot(grasp_x_tar, grasp_y_cur)

            cos_theta = np.dot(obj_x_tar, x_dir)
            sin_theta = np.dot(obj_x_tar, y_dir)
            R_rectify = np.array([
                [cos_theta, -sin_theta, 0], 
                [sin_theta, cos_theta, 0], 
                [0, 0, 1]]
            )
            stick_pose_new = create_homog_matrix(
                R_mat = R_orig @ R_rectify,
                T_vec = T_orig
            )
        else:
            stick_pose_new = stick_pose

        stick.set_pose(stick_pose_new)
        stick_grasp_poses, _ = stick.get_grasp_infos()
        grasp_x_new = stick_grasp_poses[0, :3, 0]

        # print("The old object x direction: {}".format(x_dir))
        # print("The new object x direction: {}".format(stick.get_pose()[:3, 0]))

        if vis:
            stick.vis(grasp=True, world_frame=True)
        return stick
    
    def _sample_obj_pose(self, object:Base, sample_pose, resample_xy_loc, max_iter=20, table_dist_tol=1e-3):
        """Use the trimesh compute_stable_poses function to get a stable pose. 
        The pose is supposed to (1) Be on the table (2) Not colliding with other objects
        The sampling process first sample an init pose at the origin (on the table), then sample a location (translation)

        Args:
            object (Base): The object for which to sample a pose
            max_iter (int): The maximum number for the pose sampling
            table_dist_tol: The tolerance for an object to "sink" in the table due to the numerical reason. Ideally they should be zero
        
        Returns:
            pose (array, (4,4)): The sampled pose in teh homogenous form
            succ (bool): The success flag of whether a pose has been successfully sampled. If not sampling, then return None
        """
        if not sample_pose and not resample_xy_loc:
            return object.get_pose(), None

        obj_tmp = deepcopy(object)
        pose_original = obj_tmp.get_pose()
        obj_tmp.set_pose(None)  # clear the pose
        mesh = obj_tmp.get_mesh(False, False, False, False)
        mesh = trimesh.util.concatenate(mesh)
        # The init pose. If sample_pose, then will compute a probability, else will keep the object original one
        if sample_pose:
            # Get the random pose on the center of the table
            poses, poses_probs = mesh.compute_stable_poses(sigma=0, n_samples=1, threshold=0)
        else:
            poses = [pose_original]
            poses_probs = [1]

        count = 0
        while (count <  max_iter):
            # Get a random pose
            pose = self._get_random_stable_pose(poses, poses_probs)
            # Get a random translation
            loc =  self._sample_loc() 
            if (abs(loc[0]) > self.table_size/2) or (abs(loc[1]) > self.table_size/2):
                continue

            # the pose
            pose = create_homog_matrix(T_vec=[loc[0], loc[1], 0]) @ pose

            # collison with other objects & table
            mesh_tmp = deepcopy(mesh)
            mesh_tmp.apply_transform(pose)
            # scene = trimesh.Scene()
            # scene.add_geometry(mesh_tmp)
            # scene.add_geometry(
            #     trimesh.creation.axis(origin_size=0.002, axis_radius=0.002, axis_length=0.02)
            # )
            # scene.show()
            collision_obj = self.collision_manager_obj.in_collision_single(mesh_tmp)
            table_dist = self.collision_manager_table.min_distance_single(mesh_tmp) 
            collision_table = ( abs(table_dist) > table_dist_tol)
            
            succ = (not collision_obj) #and (not collision_table)
            if succ:
                break
            count = count + 1
            

        return pose, succ
    
    def _sample_loc(self):
        # let the table edge be the four sigma, make it concentrated
        loc = np.random.multivariate_normal(mean=[0, 0], cov=np.eye(2)*((self.table_size/16)**2)) 
        return loc

    
    def _get_random_stable_pose(self, stable_poses, stable_poses_probs):
        """Return a stable pose according to their likelihood.

        Args:
            stable_poses (list[np.ndarray]): List of stable poses as 4x4 matrices.
            stable_poses_probs (list[float]): List of probabilities.

        Returns:
            np.ndarray: homogeneous 4x4 matrix
        """
        index = np.random.choice(len(stable_poses), p=stable_poses_probs)
        inplane_rot = trimesh.transformations.rotation_matrix(
            angle=np.random.uniform(0, 2.0 * np.pi), direction=[0, 0, 1]
        )
        return inplane_rot.dot(stable_poses[index])


    def clear_objs(self):
        self.objects = []
        self.collision_manager_obj = trimesh.collision.CollisionManager()
    
    def analyze_grasps(self, height=0.1, width_expand=0.02, thickness=0.02):
        """Analyze all the grasps in the scene

        It models the gripper as a cuboid, with predefined height, width_expand(in addition to the grasp openwidth), and thickness.
        The meaning of the parameters see the diagram below (thickness is along the other direction)

                            __________________  
                            |                 |  
                            |                 |  
                            |        |        |  height
                            |    ____|____    |  
                            |____|       |____|
                            |<-->|   
                            width_expand

        Args:
            height (float). The gripper cuboid model height in m
            width_expand (float). The gripper cuboid model width expansion upon the openwidth in m
            thichness (float). The gripper cuboid model thickness in m
        """
        if self.grasp_analyzed:
            return

        grasp_poses = []
        open_widths = []
        grasp_collide = []
        for idx, obj in enumerate(self.objects):
            poses, widths =  obj.get_grasp_infos(frame="world")
            grasp_poses.append(np.array(poses))
            open_widths.append(np.array(widths))
            grasp_collide_this = []
            # analyze the grasps collision, firstly remove the target object
            self.collision_manager_obj.remove_object("obj_"+str(idx))

            # for each grasp, add to the manager and analyze
            for i in range(poses.shape[0]):
                grasp_pose = poses[i, :, :]
                open_width = widths[i]
                gripper_cuboid = trimesh.creation.box(
                    [height, thickness, open_width + 2*width_expand], 
                    transform=grasp_pose @ create_homog_matrix(T_vec=[-height/2, 0, 0])
                )

                ## verify by visualization - PASS
                #grasp_this = Grasp(open_width, pose=grasp_pose)
                #scene = trimesh.Scene()
                #gripper_cuboid.visual.face_colors = [0.2, 0.2, 0.2, 0.5]
                #scene.add_geometry(gripper_cuboid)
                #scene.add_geometry(trimesh.util.concatenate(grasp_this.get_mesh()))
                #scene.add_geometry(self.table.get_mesh(False, False, False, False))
                #scene.show()

                # collision check
                collide_flag_obj = self.collision_manager_obj.in_collision_single(gripper_cuboid)
                collide_flag_table = self.collision_manager_table.in_collision_single(gripper_cuboid)
                collide_flag = collide_flag_obj or collide_flag_table 
                grasp_collide_this.append(collide_flag)
            grasp_collide.append(np.array(grasp_collide_this))

            # add back the object
            self.collision_manager_obj.add_object(
                name = "obj_"+str(idx),
                mesh = trimesh.util.concatenate(obj.get_mesh(grasp=False, obj_frame=False, world_frame=False, gripper_frame=False)),
                transform = None
            )

        # collision analysis. Any grasp must not collide with the table and the other objects
        self.grasp_poses = grasp_poses
        self.grasp_widths = open_widths 
        self.grasp_collide = grasp_collide

        # change the flag
        self.grasp_analyzed = True

    def get_meshes(self, grasp_mode=0, world_frame=False, obj_frame=False, gripper_frame=False):
        """Get the meshes for the scene, all as a list

        Args
            grasp_mode (int): The options to obtain the grasp meshes.
                            0: all grasps. Green for non-colliding, red for colliding
                            1: only the non-colliding grasp (green)
                            2: only the colliding grasp (red)
                            -1: for no grasp
            world/obj/gripper_frame (bool, optional). Get the world/obj/grasp frames or not. Default to False.

        Returns:
            obj_meshes (list):  The meshes for the objects. If obj_frame, then will include the obj frame mesh
            grasp_meshes (list): The meshes for the grasps. If grasp_frame, then will include the grasp frame mesh.\
                If the grasp caused no collision, will be set to green. Or will be red.
                If no grasp, will be an empty list.
            table_mesh  (list): The mesh for the table. If world_frame, will include the world frame mesh
        """


        obj_meshes = []
        for obj in self.objects:
            obj_meshes = obj_meshes + obj.get_mesh(grasp=False, obj_frame=obj_frame)

        grasp_meshes = []
        if grasp_mode >= 0:
            if not self.grasp_analyzed:
                self.analyze_grasps() 
            for i in range(len(self.grasp_poses)):
                for pose, width, collision_flag in zip(self.grasp_poses[i], self.grasp_widths[i], self.grasp_collide[i]):
                    if self.grasp_color is not None: 
                        color = self.grasp_color
                    elif collision_flag:
                        color = [255, 0, 0]
                    else:
                        color = [0, 255, 0]
                    
                    # conditions for adding the grasp
                    flag = (grasp_mode == 0) or (grasp_mode == 1 and not collision_flag) or (grasp_mode == 2 and collision_flag)
                    if flag:
                        grasp = Grasp(open_width=width, pose=pose, color=color)
                        grasp_meshes.append(trimesh.util.concatenate(grasp.get_mesh(False, gripper_frame, False)))
        
        self.table_mesh = self.table.get_mesh(grasp=False, world_frame=world_frame)
        return obj_meshes, grasp_meshes, self.table_mesh

    def get_grasp_infos(self):
        """Get all the grasp poses (in the world frame as the homogeneous transformation) and the open widths
        Will analyze the grasps first
        
        Returns:
            grasp_poses (list of array. (N_obj) of (N_grasps, 4, 4)). The grasp poses in the world frame
            open_widths (list of array. (N_obj) of (N_grasps)). The open widths
            collide     (list of array. (N_obj) of (N_grasps)).  The collision status
        """
        if not self.grasp_analyzed:
            self.analyze_grasps() 
        return self.grasp_poses, self.grasp_widths, self.grasp_collide
    
    def get_obj_infos(self):
        """Get the object information

        Returns:
            obj_types    (list, (N_obj,) )
            obj_dims     (list, (N_obj, ))
            obj_poses    (list, (N_obj, 4, 4)   The object poses in the world frame
        """
        obj_types = [obj.get_obj_type() for obj in self.objects]
        obj_dims = [obj.get_obj_dims() for obj in self.objects]
        obj_poses = [obj.get_pose() for obj in self.objects]

        return obj_types, obj_dims, obj_poses
    
    def to_pyrender_scene(self, grasp_mode=0, world_frame=False, obj_frame=False, gripper_frame=False):
        """Convert to the pyrender scene. 
        The mesh for each table, object, and grasps are added separately, and are named "table", "obj_#", and "grasp_#" separately, \
            where # is the index number starting from zero.
            e.g. obj_0, obj_1, grasp_0, grasp_1, grasp_2,...

        Args:
            grasp_mode (int): The options to obtain the grasp meshes.
                            0: all grasps. Green for non-colliding, red for colliding
                            1: only the non-colliding grasp (green)
                            2: only the colliding grasp (red)
                            -1: for no grasp
            world/obj/gripper_frame (bool, optional). Add the world/obj/grasp frames to the scene or not. Default to False.

        Return:
            scene (pyrender.Scene),
        """
        # create the empty scene & node list
        scene = pyrender.Scene()

        # get the meshes
        obj_meshes, grasp_meshes, table_meshes = self.get_meshes(grasp_mode=grasp_mode, world_frame=world_frame, obj_frame=obj_frame, gripper_frame=gripper_frame)

        # add the object meshes to the scene
        # NOTE: since all the meshes got are already transformed according to the assigned pose, the pose here is no longer needed
        for idx, mesh in enumerate(obj_meshes):
            name = "obj_{}".format(str(idx))
            scene.add(pyrender.Mesh.from_trimesh(mesh, smooth=False), name=name,pose=None)
        
        # add the table mesh to the scene
        table_node = scene.add(
            pyrender.Mesh.from_trimesh(
                trimesh.util.concatenate(table_meshes), smooth=False
            ), 
            name="table",
            pose=None
        )

        # add the grasp meshes to the scene
        for idx, mesh in enumerate(grasp_meshes):
            name = "grasp_{}".format(str(idx))
            scene.add(pyrender.Mesh.from_trimesh(mesh, smooth=False), name=name, pose=None)

        return scene
    
    def vis_scene(self, grasp_mode=0, world_frame=False, obj_frame=False, gripper_frame=False):
        """Visualize the scene in pyrender.

        Args:
            grasp_mode (int): The options to obtain the grasp meshes.
                            0: all grasps. Green for non-colliding, red for colliding
                            1: only the non-colliding grasp (green)
                            2: only the colliding grasp (red)
                            -1: for no grasp
            world_frame (bool, optional): _description_. Defaults to False.
            obj_frame (bool, optional): _description_. Defaults to False.
            gripper_frame (bool, optional): _description_. Defaults to False.
        """
        scene = self.to_pyrender_scene(grasp_mode=grasp_mode, world_frame=world_frame, obj_frame=obj_frame, gripper_frame=gripper_frame)
        pyrender.Viewer(scene,  use_raymond_lighting=True)
    

    @staticmethod
    def construct_scene_obj_info(obj_types, obj_dims, obj_poses, \
        table_size=1, table_thickness=0.04, table_color=[0.5, 0.5, 0.5, 1]):
        """Construct a scene from the object infos.
        The input arguments match the output of the get_obj_info

        Args:
            obj_types (list[str], (N_obj,)): The object types. 
            obj_dims (list[array], (N_obj, ), (-1, )): The object dimensions.
            obj_poses (list[array], (N_obj, ), (4, 4)): The object poses
        
        Returns:
            scene: The reconstructed scene containing all the objects
        """
        s = Scene(table_size, table_thickness, table_color)
        # objects
        objs = []
        for obj_type, obj_dim, obj_pose in zip(obj_types, obj_dims, obj_poses):
            if obj_type == "cuboid":
                obj = Cuboid.construct_obj_info(obj_dim, obj_pose)
            elif obj_type in ["cylinder", "stick", "ring"]:
                obj = Cylinder.construct_obj_info(obj_dim, obj_pose, mode=obj_type)
            elif obj_type in ["sphere", "semi_sphere"]:
                obj = Sphere.construct_obj_info(obj_dim, obj_pose, mode=obj_type)

            objs.append(obj)
        
        s.add_objs(objs)
        return s


if __name__ == "__main__":
    from data_generation import Cuboid, Sphere
    s = Scene(table_size=1, table_thickness=0.04)
    cuboid1 = Cuboid(0.06, 0.06, 0.08, pose=create_homog_matrix(T_vec=[-0.05, -0.05, 0.04]), color=np.random.choice(range(256), 3).astype(np.uint8))
    cuboid2 = Cuboid(0.04, 0.04, 0.04, pose=create_homog_matrix(T_vec=[-0.1, 0.05, 0.02]), color=np.random.choice(range(256), 3).astype(np.uint8))
    bowl = Sphere(0.08, semiSphere=True, pose=create_homog_matrix(T_vec=[0.08, 0.1, 0.08]), color=np.random.choice(range(256), 3).astype(np.uint8))
    print("Adding the first object")
    s.add_obj(cuboid1, sample_pose=False)
    print("Adding the second object")
    s.add_obj(cuboid2, sample_pose=False)
    print("Adding the third object")
    s.add_obj(bowl, sample_pose=False)
    s.analyze_grasps()
    s.vis_scene(grasp_mode=1, world_frame=False)

    # the grasp info
    grasp_poses, grasp_widths, grasp_collides = s.get_grasp_infos()
    
    # test the reconstruction of the scene
    obj_types, obj_dims, obj_poses = s.get_obj_infos()
    s_re = Scene.construct_scene_obj_info(obj_types, obj_dims, obj_poses)
    s_re.vis_scene(world_frame=False)