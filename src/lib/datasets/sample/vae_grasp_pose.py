from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from random import sample

import torch.utils.data as data
import numpy as np
import torch
import json
import cv2
import os

from utils.image import flip, color_aug
from utils.image import get_affine_transform, affine_transform
from utils.image import gaussian_radius, draw_umich_gaussian, draw_msra_gaussian
from utils.image import draw_dense_reg

from utils.keypoints import kpts_3d_to_2d, get_vanishing_points, get_ori_cls
from utils.ddd_utils import depth2pc

from graspnet_6dof.utils import utils
from utils.utils_graspnet import frame_trf_ps2graspnet

class VAEGraspPoseDataset(data.Dataset):
    """The data preparation for training the 6dof-graspnet
    Some functions adopted from the 6dof-graspnet implementation:
    https://github.com/jsll/pytorch_6dof-graspnet/blob/master/data/base_dataset.py

    Returns (N_gpo is the set grasp per object)
        meta(dict)                      A dictionary data with the following keys:
            pc (N_gpo, N_pc, 3):        The object-only point cloud in the camera frame. Duplicate N_gpo times
                                        N_pc is a preset number. Will sample the raw data to achieve that if can't
            grasp_rt (N_gpo, 16):       Vectorized homogeneous transformation matrix
            target_cps ():              The control points on the gripper
    """

    def apply_dropout(self, pc):
        if self.opt.occlusion_nclusters == 0 or self.opt.occlusion_dropout_rate == 0.:
            return np.copy(pc)

        labels = utils.farthest_points(pc, self.opt.occlusion_nclusters,
                                       utils.distance_by_translation_point)

        removed_labels = np.unique(labels)
        removed_labels = removed_labels[np.random.rand(removed_labels.shape[0])
                                        < self.opt.occlusion_dropout_rate]
        if removed_labels.shape[0] == 0:
            return np.copy(pc)
        mask = np.ones(labels.shape, labels.dtype)
        for l in removed_labels:
            mask = np.logical_and(mask, labels != l)
        return pc[mask]
    
    def grasp_sampling(self, grasp_poses, target_n):
        """Sample grasp poses
        The original implementation performs the stratified sampling, which requires clustering first.
        But here given that the GT grasps is not massive, directly sample the grasps

        Args:
            grasp_poses (N_grasps, 4, 4):   The grasp pose set to sample from
            target_n (int):                 The sampling number
        Returns:
            sampled_grasp_poses (target_n, 4, 4):       The sampled grasp poses
        """
        sampled_ids = np.random.choice(grasp_poses.shape[0], target_n)
        sample_grasp_poses = [grasp_poses[id] for id in sampled_ids]
        return np.array(sample_grasp_poses)
        

    def __getitem__(self, idx: int):


        scene_idx = self.scene_idxs[idx]
        cam_idx = self.camera_idxs[idx]

        color_path = self.images[idx]
        depth_path = self.depths[idx]
        seg_path = self.segs[idx]

        # get the depth map
        depth_raw = np.load(depth_path)

        # get the camera intrinsic
        intrinsic, camera_poses, obj_types, obj_dims, obj_poses, _, _, _,  = \
            self._get_scene_info(scene_idx)
        camera_pose = camera_poses[cam_idx]

        # get the point cloud
        pc = depth2pc(depth_raw, intrinsic=intrinsic, frame="camera", flatten=True, remove_zero_depth=True)
        pc_table = depth2pc(depth_raw, intrinsic=intrinsic, frame="tabletop", flatten=True, remove_zero_depth=True, extrinsic=np.linalg.inv(camera_pose))
        pc = pc[pc_table[:, 2] > 0.005]  # object only (and partial)
        pc = self.apply_dropout(pc)
        pc = utils.regularize_pc_point_count(pc, self.opt.npoints)


        # get the GT grasp poses (label in the tabletop frame), and transform to the camera pose
        grasp_poses, _, _ = self._get_gt_grasps(
            scene_idx=scene_idx, cam_idx=cam_idx, filter_collides=(not self.opt.no_collide_filter),
            #correct_rl=self.opt.correct_rl ## This seems not necessary
        )    
        grasp_poses = grasp_poses[0] # assume only one object
        grasp_poses = self.grasp_sampling(grasp_poses, self.opt.num_grasps_per_object)
        grasp_poses = np.linalg.inv(camera_pose)[None, :, :]@(grasp_poses)

        # grasp pose transform to the graspnet definition
        grasp_poses = frame_trf_ps2graspnet(grasp_poses)

        # centralize the point cloud
        pc_mean = np.mean(pc, 0, keepdims=True)
        pc[:, :3] -= pc_mean[:, :3]
        grasp_poses[:, :3, 3] -= pc_mean[0, :3]
        camera_pose[:3, 3] -= pc_mean[0, :3]

        # get the control points -TODO: it is in a different frame
        gt_control_points = utils.transform_control_points_numpy(
            np.array(grasp_poses), self.opt.num_grasps_per_object, mode='rt')
        target_cps = np.array(gt_control_points[:, :, :3])


        # ## visualization for debug. 
        # from mayavi import mlab
        # from graspnet_6dof.utils.visualization_utils import draw_scene
        # mlab.figure(bgcolor=(1, 1, 1))
        # draw_scene(
        #    pc,
        #    pc_color=None,
        #    grasps=grasp_poses,
        #    grasp_scores=None,
        # )
        # mlab.show()

        # store the data
        ret = {
            "pc": np.array([pc] * self.opt.num_grasps_per_object, dtype=np.float32)[:, :, :3],
            "grasp_rt": grasp_poses.reshape(self.opt.num_grasps_per_object, -1),
            "target_cps": target_cps
        } 

        return ret
    
  