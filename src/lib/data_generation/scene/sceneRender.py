"""

Build upon the Scene class to add the camera to render the image data

"""
from typing import List

import numpy as np
from numpy import pi
import pyrender
import trimesh

import sys, os
root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if root_dir not in sys.path:
    sys.path.append(root_dir)

from utils.transform import create_homog_matrix , create_rot_mat_axisAlign, cam_pose_convert
from data_generation import Scene
from data_generation import Grasp, Base, Cuboid, Cylinder, Sphere

class SceneRender(Scene):
    """The scene renderer allows adding camera(s) and 

    """

    def __init__(self, table_size=1, table_thickness=0.04, table_color=[0.5, 0.5, 0.5, 1], \
        camera="realsense", camera_num=4, cam_width=None, cam_height=None, radius_range=[1, 1.2], latitude_range=[pi/6, pi/2]) -> None:
        """Create a scene with the camera image rendering function.
        The camera will be sampled on a semi-sphere covering the scene.

        Args:
            table_size (int, optional): The size of the table in cm. Defaults to 100cm.
            camera (str/3-by-3 array, optional): The camera name or the camera intrinsic. If set to a name then will use the pre-saved intrinsic.
                If set to a 3-by-3 array, then will use it as the intrinsic. Defaults to "realsense".
        """
        super().__init__(table_size, table_thickness, table_color)

        self.camera_num = camera_num

        # The camera intrinsic matrix
        if camera == "realsense":
            self.intrinsic = np.array(
                [[616.36529541, 0, 310.25881958], 
                [0, 616.20294189, 236.59980774], 
                [0, 0, 1]]
            )
            self.width=640
            self.height=480
        elif camera == "kinect":
            raise NotImplementedError
        elif not isinstance(camera, str):
            assert cam_width is not None and cam_height is not None, "Requires also the image width and height as input"
            # if not a str name, then the camera should be the intrinsic matrix
            self.intrinsic = np.array(camera)
            self.width = cam_width
            self.height = cam_height
        
        # parse the camera parameter
        self._parse_cam_params_intrinsic()
        self._create_camera()

        # camera extrinsic poses
        self.r_range = radius_range
        self.latitude_range = latitude_range
        self.camera_poses = np.array([])

        # init camera poses
        self.resample_camera_poses()
        
    
    def _parse_cam_params_intrinsic(self):
        """Parse the camera parameters from the intrinsic matrix
        """
        self.fx = self.intrinsic[0, 0]
        self.fy = self.intrinsic[1, 1]
        self.cx = self.intrinsic[0, 2]
        self.cy = self.intrinsic[1, 2]
    
    def set_camera_poses(self, camera_poses):
        """Set the camera poses

        Args:
            camera_poses (List[array], (N_cam) of (4, 4)):      A list of 4x4 camera poses in the world frame
        """
        self.camera_poses = camera_poses
    
    def resample_camera_poses(self):
        """Generate a list of camera poses.
        Will overwrite the previous poses if any
        """
        self.camera_poses = []
        for _ in range(self.camera_num):
            # sample a radius, latitude, and longitude
            r = self.r_range[0] + np.random.rand() * (self.r_range[1] - self.r_range[0])
            la = self.latitude_range[0] + np.random.rand() * (self.latitude_range[1] - self.latitude_range[0])
            lo = np.random.rand() * 2 * pi

            # create the pose
            pose = create_homog_matrix(T_vec=[0, r, 0], R_mat=create_rot_mat_axisAlign([-1, 3, 2])) # the R_mat so that the camera is still pointing to the table center
            trf1 = trimesh.transformations.euler_matrix(0, la, 0, axes='rzxz')
            trf2 = trimesh.transformations.euler_matrix(lo, 0, 0, axes='rzxz')
            pose = trf2 @ trf1 @ pose

            self.camera_poses.append(pose)

            ## a dumb pose for debug
            #from utils.transform import create_rot_mat_axisAlign
            #self.camera_poses.append(create_homog_matrix(
            #    #R_mat=create_rot_mat_axisAlign([1, -2, -3]),
            #    T_vec=[0, 0, 1]
            #))
    
    def _create_camera(self):
        """Create the pyrender&trimesh camera
        pyrender camera is for actually render the camera data
        trimesh camera is for vsiualize the scene as well as help determine the camera pose
        """
        z_near = 0.04
        z_far = 20
        self.camera_pyrd = pyrender.IntrinsicsCamera(self.fx, self.fy, self.cx, self.cy, znear=z_near, zfar=z_far)
        self.camera_trimesh = trimesh.scene.cameras.Camera(
            resolution=(self.height, self.width),
            focal=(self.fx, self.fy),
            z_near = z_near,
            z_far = z_far
        )

        self.proj = self.camera_pyrd.get_projection_matrix(width=self.width, height=self.height)

    def render_imgs(self, instance_masks=False, grasp_mode=-1):
        """Render images

        Args:
            instance_masks (bool):  Option to render the instance masks. Defaults to True
            grasp_mode (int): The options to obtain the grasp meshes.
                            0: all grasps. Green for non-colliding, red for colliding
                            1: only the non-colliding grasp (green)
                            2: only the colliding grasp (red)
                            -1: for no grasp


        Returns:
            colors (array of the size (N_cam, H, W, 3)): The images. N_cam is the number of the cameras
            depths (array of the size (N_cam, H, W)): The depth maps. N_cam is the number of the cameras
            ins_masks (array of the shape (N_cam, H, W)): The instance masks. N_cam is the camera numbers.  \
                The pixels corresponding to the i^th added object is labeled i. 0 is the background
                Only return is instance_masks is True
        """
        colors = []
        depths = []
        ins_masks = []

        scene = self.to_pyrender_scene(grasp_mode=grasp_mode, world_frame=False, obj_frame=False, gripper_frame=False)

        # add the camera to the scene, and prepare the render
        cam_node = scene.add(self.camera_pyrd, pose=np.eye(4), name="camera")
        img_renderer = pyrender.OffscreenRenderer(viewport_height=self.height, viewport_width=self.width,point_size=1)

        # add the light
        light = pyrender.SpotLight(
            color=np.ones(4),
            intensity=3.0,
            innerConeAngle=np.pi / 16,
            outerConeAngle=np.pi / 3.0,
        )

        # render images for all the camera poses
        ln = None
        for pose in self.camera_poses:
            if ln is not None:
                scene.remove_node(ln)
            scene.set_pose(cam_node, pose)
            ln = scene.add(light, pose=pose, name="light")
            color, depth = img_renderer.render(scene)
            colors.append(color)
            depths.append(depth)

            if instance_masks:
                ins_masks_this_cam = self._render_instance_masks(scene, img_renderer, depth) 
                ins_masks.append(ins_masks_this_cam)

        # end 
        img_renderer.delete()
        colors = np.array(colors)
        depths = np.array(depths)
        ins_masks = np.array(ins_masks)

        if instance_masks:
            return colors, depths, ins_masks
        else:
            return colors, depths

    def _render_instance_masks(self, scene:pyrender.Scene, renderer:pyrender.OffscreenRenderer, full_depth):
        """Render the instance mask for each object in the object adding order

        NOTE: The code is adopted from the issue: https://github.com/mmatl/pyrender/issues/15, 
        but it is not a very efficient method. 
        
        Returns:
            instance_mask (array (H, W)): The instance mask. The pixel value i means it belong's to the i'th object. 0 means background or table
        """
        height = renderer.viewport_height
        width = renderer.viewport_width
        instance_mask = np.zeros((height, width), dtype=int)

        flags = pyrender.RenderFlags.DEPTH_ONLY

        # Iterate through the object nodes, disable each one of them and set the pixel with large depth change to be the pixel of this object
        for node in scene.mesh_nodes:
            if "obj" in node.name:
                _, idx = node.name.split("_")
                idx = int(idx)
            else:
                # skip non-object mesh
                continue

            # disable, render depth, and compare    
            node.mesh.is_visible = False
            depth = renderer.render(scene, flags=flags)
            mask = np.logical_and(
                (np.abs(depth - full_depth) > 1e-6), np.abs(full_depth) > 0
            )

            instance_mask[mask] = idx + 1

            # add back
            node.mesh.is_visible = True
        
        return instance_mask

    def get_camera_infos(self, style="OpenCV"):
        """Get the camera infos

        Args:
            style (str):    The camera pose in the OpenCV or the OpenGL style. The camera frame definition is different, \
                hence the extrinsic matrix would be different.

        Returns:
            intrinsic (array (3,3)). The camera intrinsics. 
            camera_poses (array (N_cam, 4, 4)). The camera poses. \
                Note that this is the frame transformation from the World to the camera, not the coordinate transformation.\
                If want the extrinsic (world coordinate to camera coordinate), take the inverse of this pose
            proj (array, (4, 4)). The pyrender perspective camera projection matrix
        """
        if style == "OpenCV":
            trf = create_rot_mat_axisAlign([1, -2, -3])
            trf = create_homog_matrix(R_mat=trf)
            camera_poses = self.camera_poses @ trf
        elif style == "OpenGL":
            camera_poses = self.camera_poses

        return self.intrinsic, camera_poses, self.proj

    def to_trimesh_scene(self, grasp_mode=0, camera_marker=False, world_frame=False, obj_frame=False, gripper_frame=False):
        """Convert to the pyrender scene. Overwrite the Scene Class funciton to add the camera

        Args:
            grasp_mode (int): The options to obtain the grasp meshes.
                            0: all grasps. Green for non-colliding, red for colliding
                            1: only the non-colliding grasp (green)
                            2: only the colliding grasp (red)
                            -1: for no grasp
            camera_mesh: (bool, optional).  Add the camera marker meshes into the scene or not. Defaults to False
            world/obj/gripper_frame (bool, optional). Add the world/obj/grasp frames to the scene or not. Default to False.

        Return:
            scene (pyrender.Scene)
        """
        # create the empty scene
        scene = trimesh.scene.Scene()

        # get the meshes
        obj_meshes, grasp_meshes, table_meshes = self.get_meshes(grasp_mode=grasp_mode, world_frame=world_frame, obj_frame=obj_frame, gripper_frame=gripper_frame)
        if len(grasp_meshes) > 0:
            mesh_list = obj_meshes + grasp_meshes + table_meshes
        else:
            mesh_list = obj_meshes + table_meshes

        
        # add to the scene
        scene.add_geometry(mesh_list)

        # camera markers
        if camera_marker:
            for pose in self.camera_poses:
                # get the camera marker mesh
                camera_marker_list = trimesh.creation.camera_marker(self.camera_trimesh)
                # NOTE: the trimesh and pyrender has different camera frame definition. \
                # The trimesh camera is looking at the +z, whereas the pyrender at the -z
                # So reverse the direction here. It will also reverse the y axis, but it is okay since it is just a marker
                reverse_direction = create_homog_matrix(create_rot_mat_axisAlign([1, -2, -3]))
                scene.add_geometry(camera_marker_list, transform=pose@reverse_direction)

        return scene
    
    def vis_scene(self, mode="trimesh", grasp_mode=0, cameras=True, world_frame=False, obj_frame=False, gripper_frame=False):
        """visualize the scene, either use trimehs or the pyrender
        NOTE: If want to visualize the cameras, then please use the trimesh.\
            The pyrender scene will visualize the scene from the camera's view if any is added, which means it will not visualize the scene WITH the camera.\
            also the trimesh's camera marker includes non-mesh, so not able to add to the pyrender's scene

        Args:
            mode (str, optional): Use the trimesh or the pyrender for the visualization
            grasp_mode (int): The options to obtain the grasp meshes.
                            0: all grasps. Green for non-colliding, red for colliding
                            1: only the non-colliding grasp (green)
                            2: only the colliding grasp (red)
                            -1: for no grasp
            cameras (bool, optional): [description]. Defaults to True.
            world_frame (bool, optional): [description]. Defaults to False.
            obj_frame (bool, optional): [description]. Defaults to False.
            gripper_frame (bool, optional): [description]. Defaults to False.
        """
        if mode == "pyrender":
            if cameras:
                Warning("The pyrender visualization CANNOT visualize the camera, please consider change to the trimesh mode")
            scene = self.to_pyrender_scene(grasp_mode=grasp_mode, camera_marker=cameras, world_frame=world_frame, obj_frame=obj_frame, gripper_frame=gripper_frame)
            pyrender.Viewer(scene,  use_raymond_lighting=True)
        elif mode == "trimesh":
            scene = self.to_trimesh_scene(grasp_mode=grasp_mode, camera_marker=cameras, world_frame=world_frame, obj_frame=obj_frame, gripper_frame=gripper_frame)
            scene.show()
        else:
            raise NotImplementedError
    
    @staticmethod
    def construct_scene_obj_info(obj_types, obj_dims, obj_poses, \
        camera_poses, \
        obj_colors=None,
        trl_sample_num=5, rot_sample_num=11, \
        table_size=1, table_thickness=0.04, table_color=[0.5, 0.5, 0.5, 1]):
        """Construct a scene from the object infos.
        The input arguments match the output of the get_obj_info

        NOTE: For now assumes all the objects share the same sample number for both rotation and translation.

        Args:
            obj_types (list[str], (N_obj,)): The object types. 
            obj_dims (list[array], (N_obj, ), (-1, )): The object dimensions.
            obj_poses (list[array], (N_obj, ), (4, 4)): The object poses
            camera_poses (list[array], (N_cam, ), (4, 4)): The camera poses in the OpenCV style
            trl_sample_num (int, optional): The number of translation samples. Defaults to 5.
            rot_sample_num (int, optional): The number of rotation samples. Defaults to 11.
        
        Returns:
            scene: The reconstructed scene containing all the objects
        """
        s = SceneRender(table_size, table_thickness, table_color)
        # objects
        if obj_colors is None:
            obj_colors = [ np.random.choice(range(256), size=3) for _ in range(len(obj_types))]
        objs = []
        for obj_type, obj_dim, obj_pose, obj_color in zip(obj_types, obj_dims, obj_poses, obj_colors):
            if obj_type == "cuboid":
                obj = Cuboid.construct_obj_info(obj_dim, obj_pose, trl_sample_num=trl_sample_num, color=obj_color)
            elif obj_type in ["cylinder", "stick", "ring"]:
                obj = Cylinder.construct_obj_info(obj_dim, obj_pose, mode=obj_type, trl_sample_num=trl_sample_num, rot_sample_num=rot_sample_num, color=obj_color)
            elif obj_type in ["sphere", "semi_sphere"]:
                obj = Sphere.construct_obj_info(obj_dim, obj_pose, mode=obj_type, rot_sample_num=rot_sample_num, color=obj_color)

            objs.append(obj)
        
        s.add_objs(objs)

        # camera
        camera_poses = np.array(camera_poses)
        s.camera_poses = cam_pose_convert(camera_poses, mode="cv2gl")
        return s
        
if __name__ == "__main__":
    from data_generation import Sphere, Cuboid, Cylinder, Grasp
    from utils.keypoints import kpts_3d_to_2d, plot_grasps_kpts
    import trimesh.transformations as tra

    import matplotlib.pyplot as plt

    grasp_mode = 1

    # create the scene
    s = SceneRender(camera_num=5, table_color=[1.0, 0.0, 0.0, 1])
    cuboid1 = Cuboid(0.06, 0.06, 0.08, pose=create_homog_matrix(T_vec=[-0.05, -0.05, 0.04]), color=np.random.choice(range(256), 3).astype(np.uint8))
    #cuboid2 = Cuboid(4, 4, 4, pose=create_homog_matrix(T_vec=[-0.1, 0.05, 0.02]), color=np.random.choice(range(256), 3).astype(np.uint8))
    stick = Cylinder(0.01, 0.07, mode="stick")
    bowl = Sphere(0.08, semiSphere=True, pose=create_homog_matrix(T_vec=[0.08, 0.1, 0.08]), color=np.random.choice(range(256), 3).astype(np.uint8))
    try:
        print("Adding the first object")
        s.add_obj(cuboid1, sample_pose=True)
        print("Adding the second object")
        s.add_obj(stick, sample_pose=True)
        print("Adding the third object")
        s.add_obj(bowl, sample_pose=True)
        # NOTE: if the parameter is not designed correctly, some objects might not be able to added since the colllision is inevitable
    except RuntimeError:
        print("Some objects can not be added") 
    s.vis_scene(grasp_mode=grasp_mode, mode="trimesh", world_frame=True)

    # get the grasp poses and the camera poses
    grasp_poses, grasp_open_widths, grasp_collides = s.get_grasp_infos()
    intrinsic, camera_poses, proj_mats = s.get_camera_infos("OpenCV")

    grasp_poses = np.concatenate(grasp_poses, axis=0)
    grasp_open_widths = np.concatenate(grasp_open_widths, axis=0)
    grasp_collides = np.concatenate(grasp_collides, axis=0)

    # grasp keypoints mode
    grasp_kpts_mode = "hedron"

    # take the images
    ins_masks = None
    colors, depths, ins_masks = s.render_imgs(grasp_mode=-1,instance_masks=True)
    #colors, depths = s.render_imgs(instance_masks=False)
    for idx, (color, depth) in enumerate(zip(colors, depths)):
        camera_pose = camera_poses[idx] 
        if ins_masks is not None:
            ins_mask = ins_masks[idx]

        # plot the grasp keypoints - sample a few
        sample_num = int(grasp_poses.shape[0] / 5)
        sample_idxes = np.random.choice(np.arange(grasp_poses.shape[0]), sample_num, replace=False)
        # grasp kpts cache
        grasp_kpts = []
        for i in sample_idxes:
           grasp_pose = grasp_poses[i, :, :]
           grasp_width = grasp_open_widths[i]
           grasp_collide = grasp_collides[i]
           # skip invalid grasps
           if grasp_collide:
               continue

           grasp = Grasp(grasp_width, pose=grasp_pose, kpts_option=grasp_kpts_mode)
           kpts_coord = grasp.get_kpts(frame="world")  # the keypoint coordinate in the gripper frame, (N_kpt, 3)

           kpts_img = kpts_3d_to_2d(intrinsic, np.linalg.inv(camera_pose), kpts_coord)

           grasp_kpts.append(kpts_img)
        color = plot_grasps_kpts(color, grasp_kpts, kpts_mode=grasp_kpts_mode, size=4)

        if ins_masks is not None:
            f, axarr = plt.subplots(1, 3)
        else:
            f, axarr = plt.subplots(1, 2)
        im = axarr[0].imshow(color)
        f.colorbar(im, ax=axarr[0])
        im = axarr[1].imshow(depth)
        f.colorbar(im, ax=axarr[1])

        if ins_masks is not None:
            im = axarr[2].imshow(ins_mask)
            f.colorbar(im, ax=axarr[2])
    plt.show() 



