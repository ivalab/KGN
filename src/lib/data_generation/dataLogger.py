import os
import numpy as np
import cv2
import json


class DataLogger():
    """The logger to write out the data.
    The data will be saved out in the following format:
    
    - base_dir
        - scene1
            infos.json
            - color_images
                rgb_1.png
                rgb_2.png
            - depths
                depth_raw_1.npy
                depth_raw_2.npy
            - seglabels
                seglabels_1.png
                seglabels_2.png
        - scene2
    
    Upgrade to:
        1. camera intrinsic (4, 4). All camera is going to have the same camera intrinsics
        2. camera poses, (N_cam, 4, 4), where N_cam corresponds to the rgb_*
        3. objs
            3.1 obj_types (N_obj, )
            3.2 obj_dims (N_obj, )  a list of dimensions. The dimension size varies for different objs
            3.3 obj_poses (N_obj, 4, 4)
        4. grasps
            3.1 Poses, List[array]              (N_obj,) of (N_grasp_obj, 4, 4)
            3.2 Open_width, List[array]         (N_obj,) of  (N_grasp_obj, )
            3.3 grasp_collision: List[array]     (N_obj, ) of (N_grasp_obj)
    """

    def __init__(self, logging_directory, continue_logging=False):

        self.base_directory = logging_directory
        self.continue_logging = continue_logging

        if self.continue_logging:
            raise NotImplementedError
        
        # Create directory to save data
        if not os.path.exists(self.base_directory):
            os.makedirs(self.base_directory)
        
        # all sub dirs - will be populated later
        self.scene_subdir = None
        self.color_dir = None
        self.depth_raw_dir = None
        self.depth_img_dir = None
        self.seg_dir = None

    def save_scene_data(self, scene_idx, camera_mat, camera_poses, rgb_imgs, depths, seglabel_maps,\
            grasp_poses, grasp_widths, grasp_collision, \
            obj_types, obj_dims, obj_poses):
        """ Save scenen data

        Args:
            scene_idx (int): [description]
            camera_mat (array, (3, 3)): The camera intrinsic matrix
            camera_poses (array (N_cam, 4, 4)): The camera poses in the world frame
            rgb_imgs (array (N_cam, H, W, 3)): [description]
            depths (array (N_cam, H, W)): [description]
            seglabel_maps (N_cam, H, W): [description]
            grasp_poses (List[array], (N_obj,) of (N_grasp, H, W)): [description]
            grasp_widths (List[array], (N_obj) of (N_grasp,)): [description]
            grasp_collision (List[array], (N_obj) of (N_grasp ,)): [description]
            obj_types (List[str], (N_obj))
            obj_dims (List[array], (N_obj) of (D))
            obj_poses (List[array], (N_obj ) of (4, 4))
        """

        # scene subdirectory
        self._make_scene_subdirs(scene_idx)
        
        # save images
        self.save_colors(rgb_imgs)
        self.save_depths(depths)
        self.save_seg(seglabel_maps)
        self.save_info(camera_mat, camera_poses, grasp_poses, grasp_widths, grasp_collision, 
            obj_types=obj_types, obj_dims=obj_dims, obj_poses=obj_poses)
    
    def _make_scene_subdirs(self, scene_idx):
        """Make the scene subdirectories and store them as attributes of the logger.

        Args:
            scene_idx (int): The index of the scene
        """
        self.scene_subdir = os.path.join(self.base_directory, str(scene_idx))
        if not os.path.exists(self.scene_subdir):
            os.makedirs(self.scene_subdir)

        self.color_dir = os.path.join(self.scene_subdir, 'color_images')
        if not os.path.exists(self.color_dir):
            os.makedirs(self.color_dir)

        self.depth_raw_dir = os.path.join(self.scene_subdir, 'depth_raw')
        if not os.path.exists(self.depth_raw_dir):
            os.makedirs(self.depth_raw_dir)

        self.depth_img_dir = os.path.join(self.scene_subdir, 'depth_img')
        if not os.path.exists(self.depth_img_dir):
            os.makedirs(self.depth_img_dir)

        self.seg_dir =  os.path.join(self.scene_subdir, 'seg_labels')
        if not os.path.exists(self.seg_dir):
            os.makedirs(self.seg_dir)

    def save_colors(self, color_images):
        N_cam = color_images.shape[0]
        for i in range(N_cam):
            rgb = color_images[i, :, :, :]
            bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            cv2.imwrite(
                os.path.join(self.color_dir, 'color_image_{}.png'.format(i)), 
                bgr 
            )
    
    def save_depths(self, depths):
        N_cam = depths.shape[0]
        for i in range(N_cam):
            dep = depths[i, :, :]
            # save the raw depth map as the npy files (takes a lot of space)
            np.save(
                os.path.join(self.depth_raw_dir, 'depth_raw_{}.npy'.format(i)), 
                dep
            )

            # Save the depth image for visulization
            dep_image = np.round(dep * 10000).astype(np.uint16) # Save depth in 1e-4 meters
            dep_image = cv2.cvtColor(dep_image, cv2.COLOR_GRAY2BGR)
            cv2.imwrite(
                os.path.join(self.depth_img_dir, 'depth_image_{}.png'.format(i)), 
                dep_image 
            )

    def save_info(self, camera_mat, camera_poses, \
            grasp_poses, grasp_widths, grasp_collision, \
            obj_types, obj_dims, obj_poses):
        """ Save the infos as json """

        # the file path
        filepath = os.path.join(self.scene_subdir, 'scene_info.json')

        # convert to list
        grasp_poses_list = [pose.tolist() if isinstance(pose, np.ndarray) else pose for pose in grasp_poses]
        widths_list = [widths.tolist() if isinstance(widths, np.ndarray) else widths for widths in grasp_widths]
        collides_list = [collides.tolist() if isinstance(collides, np.ndarray) else collides for collides in grasp_collision]
        obj_dims_list = [dims.tolist() if isinstance(dims, np.ndarray) else dims for dims in obj_dims]
        obj_poses_list = [poses.tolist() if isinstance(poses, np.ndarray) else poses for poses in obj_poses]

        # data
        data = {
            "intrinsic": camera_mat.tolist(),
            "camera_poses": camera_poses.tolist(),
            "grasp_poses": grasp_poses_list,
            "grasp_widths": widths_list,
            "grasp_collision": collides_list,
            "obj_types": obj_types,
            "obj_dims": obj_dims_list,
            "obj_poses": obj_poses_list
        }

        # save out
        with open(filepath, "w") as f:
            json.dump(data, f)


    def save_seg(self, seg_label_masks):
        """ Method to save segmask image """

        N = seg_label_masks.shape[0]
        for i in range(N):
            seg_label_mask = seg_label_masks[i, :, :].astype(np.uint8)
            # write as single channel, so need to use jpg
            cv2.imwrite(os.path.join(self.seg_dir, 'segmask_label_{}.jpg'.format(i)), seg_label_mask)
