from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from typing import OrderedDict

import pdb
import numpy as np
import json
from tqdm import tqdm
import os
import cv2
from torch import angle, dtype, functional
from scipy.spatial.transform import Rotation as R
import math

import torch.utils.data as data

from grasp_kpts import generate_kpt_links, BoxVertexType, HedronVertexType, TailVertexType

from utils.file import read_numList_from_file
from data_generation.grasp.grasp import Grasp
from data_generation.scene.sceneRender import SceneRender
from utils.keypoints import kpts_3d_to_2d, get_vanishing_points, get_ori_cls
from utils.transform import create_homog_matrix, create_rot_mat_axisAlign
from utils.ddd_utils import depth2pc

from utils.image import flip, color_aug
from utils.image import get_affine_transform, affine_transform
from utils.image import gaussian_radius, draw_umich_gaussian, draw_msra_gaussian
from utils.image import draw_dense_reg


from utils.metric import MetricCalculator


class PSGrasp(data.Dataset):
    num_classes = 1
    num_grasp_kpts = 4
    num_cams_per_scene = 5
    default_resolution = [512, 512]

    # Mean and stddev of the RGB-D
    means_rbgd = np.array([0.64326715, 0.64328622, 0.64328383, 0.7249166783727705], dtype=np.float32)
    std_rbgd = np.array([0.03159907, 0.03159791, 0.03159052, 0.06743566336832418], dtype=np.float32)


    def __init__(self, opt, split):
        super().__init__()

        self.split = split
        if self.split == "val":
            self.split = "train"

        self.opt = opt
        if self.opt.unitTest:
            self.data_dir = os.path.join(opt.data_dir, "ps_grasp_unitTest")
        else:
            if self.opt.ps_data_mode == "single":
                self.data_dir = os.path.join(opt.data_dir, "ps_grasp_single_1k")
            elif self.opt.ps_data_mode == "multi":
                self.data_dir = os.path.join(opt.data_dir, "ps_grasp_multi_1k")
            else:
                raise NotImplementedError


        # mean and std depending on the input modality
        if self.opt.input_mod == "RGD":
            self.mean = self.means_rbgd[[0, 1, 3]].reshape(1, 1, 3)
            self.std = self.std_rbgd[[0, 1, 3]].reshape(1, 1, 3)
        elif self.opt.input_mod == "RGBD":
            self.mean = self.means_rbgd.reshape(1, 1, 4)
            self.std = self.std_rbgd.reshape(1, 1, 4)
        elif self.opt.input_mod == "RGB":
            self.mean = self.means_rbgd[[0, 1, 2]].reshape(1, 1, 3)
            self.std = self.std_rbgd[[0, 1, 2]].reshape(1, 1, 3)
        elif self.opt.input_mod == "D":
            self.mean = self.means_rbgd[3].reshape(1, 1, 1)
            self.std = self.std_rbgd[3].reshape(1, 1, 1)


        # color-jitter.
        self._data_rng = np.random.RandomState(123)
        self._eig_val = np.array([0.2141788, 0.01817699, 0.00341571],
                                 dtype=np.float32)
        self._eig_vec = np.array([
            [-0.58752847, -0.69563484, 0.41340352],
            [-0.5832747, 0.00994535, -0.81221408],
            [-0.56089297, 0.71832671, 0.41158938]
        ], dtype=np.float32)

        # keypoint type
        self.kpt_type = self.opt.kpt_type 

        # flip idx..
        if self.kpt_type == "box":
            self.flip_idx = [
                [BoxVertexType.Left, BoxVertexType.Right], 
                [BoxVertexType.TopLeft, BoxVertexType.TopRight], 
            ]
        elif self.kpt_type == "hedron":
            self.flip_idx = [
                [HedronVertexType.Left, HedronVertexType.Right],
                [HedronVertexType.TopFront, HedronVertexType.TopBack]
            ]
        elif self.kpt_type == "tail":
            self.flip_idx = [
                [TailVertexType.Left, TailVertexType.Right]
            ]
        else:
            raise NotImplementedError

        # minimum open width & canonical open width
        self.min_open_width = self.opt.min_open_width
        self.canonical_open_width = self.opt.open_width_canonical

        # max grasp number per scene. For now, counted as:
        # 264 = 12*5(cuboid) + 7*12(cylinder) + 12*5(stick) + 12*2(ring) + 12*2 (sphere) + 12 (bowl)
        # Round to 300
        self.max_grasps = 300

        print('\n\n ==> initializing the Primitive Shape Grasp {} data.'.format(split))
        self.num_scene, self.scene_idxs, self.camera_idxs = self._load_scene_camera_idx()
        self.scenes, self.images, self.depths, self.segs = self._load_data_paths(
            self.scene_idxs, self.camera_idxs)
        self.num_samples = len(self.camera_idxs)
        print('Loaded {} {} scenes with {} samples'.format(
            split, self.num_scene, self.num_samples))

    def _load_scene_camera_idx(self):
        """Load the scene index and the camera index for each data
        If have 2 scenes both with 3 cameras, it would be:
            scene_idx: [0, 0, 0, 1, 1, 1]
            camera_idx: [0, 1, 2, 0, 1, 2]

        Returns:
            scene_idx
            camera_idx
        """
        camera_idx = []
        scene_idx = []

        # get the scene idx list
        scene_idx_list = read_numList_from_file(
            os.path.join(self.data_dir, self.split+".txt")
        )
        scene_num = len(scene_idx_list)

        # parse the scene index and camera index list
        for idx in scene_idx_list:
            _, color_folder, _, _ = self._get_scene_paths(idx)
            image_files = [file for file in os.listdir(color_folder)
                           if file.lower().endswith(('.png', '.jpg'))]
            num_data_this = len(image_files)
            # num_data_this = 1 # Fore debug
            camera_idx.extend(list(range(num_data_this)))
            scene_idx.extend([idx]*num_data_this)

        return scene_num, scene_idx, camera_idx

    def _load_data_paths(self, scene_idx, camera_idx):
        images = []
        depths = []
        segs = []
        scenes = []
        for s_id, c_id in zip(scene_idx, camera_idx):
            img_path, dep_path, seg_path = self._get_data_paths(s_id, c_id)
            scene_path, _, _, _ = self._get_scene_paths(s_id)
            images.append(img_path)
            depths.append(dep_path)
            segs.append(seg_path)
            scenes.append(scene_path)
        return scenes, images, depths, segs

    def _get_scene_paths(self, scene_idx):
        """
        Returns:
            scene root folder, 
            color images subfolder, 
            depth_raw subfolder, 
            seg_labesl subfolder path.
        """
        scene_path = os.path.join(self.data_dir, str(scene_idx))
        color_path = os.path.join(scene_path, "color_images")
        depth_path = os.path.join(scene_path, "depth_raw")
        seg_label_path = os.path.join(scene_path, "seg_labels")
        return scene_path, color_path, depth_path, seg_label_path

    def _get_data_paths(self, scene_idx, cam_idx):
        """Get the data paths given the scene index and the camera index

        Returns:
            color_img_path, depth_raw_path, seg_labels_path
        """
        # the scene paths
        _, color_path, depth_path, seg_path = self._get_scene_paths(scene_idx)
        color_img_path = os.path.join(
            color_path, "color_image_{}.png".format(cam_idx))
        depth_raw_path = os.path.join(
            depth_path, "depth_raw_{}.npy".format(cam_idx))
        seg_label_path = os.path.join(
            seg_path, "segmask_label_{}.jpg".format(cam_idx))

        return color_img_path, depth_raw_path, seg_label_path

    def _get_scene_info(self, scene_index):
        """Get the scene information

        Return:
            intrinsic, camera_poses, \
                obj_types, obj_dims, obj_poses \
                grasp_poses, grasp_widths, grasp_collisions
        """
        info_file = os.path.join(self.data_dir, str(
            scene_index) + "/scene_info.json")
        with open(info_file, "r") as f:
            infos = json.load(f)

        # camera info
        intrinsic = np.array(infos["intrinsic"])
        camera_poses = np.array(infos["camera_poses"])

        # grasp info
        grasp_poses_list = infos["grasp_poses"]
        grasp_poses = [np.array(poses) for poses in grasp_poses_list]
        grasp_widths_list = infos["grasp_widths"]
        grasp_widths = [np.array(widths) for widths in grasp_widths_list]
        grasp_collision_list = np.array(infos["grasp_collision"])
        grasp_collision = [np.array(collisions)
                           for collisions in grasp_collision_list]

        # object info
        obj_types = infos["obj_types"]
        obj_dims_list = infos["obj_dims"]
        obj_dims = [np.array(dims) for dims in obj_dims_list]
        obj_poses_list = infos["obj_poses"]
        obj_poses = [np.array(poses) for poses in obj_poses_list]

        return intrinsic, camera_poses, \
            obj_types, obj_dims, obj_poses, \
            grasp_poses, grasp_widths, grasp_collision



    def _get_border(self, border, size):
        """Shrink the border until 2*border > size"""
        i = 1
        while size - border // i <= border // i:
            i *= 2
        return border // i
    
    
    ###################################################
    # Evaluations
    ###################################################

    def convert_eval_format(self, all_bboxes):
        raise NotImplementedError
        # import pdb; pdb.set_trace()
        detections = []
        for image_id in all_bboxes:
            for cls_ind in all_bboxes[image_id]:
                category_id = 1
                for dets in all_bboxes[image_id][cls_ind]:
                    bbox = dets[:4]
                    bbox[2] -= bbox[0]
                    bbox[3] -= bbox[1]
                    score = dets[4]
                    bbox_out = list(map(self._to_float, bbox))
                    keypoints = np.concatenate([
                        np.array(dets[5:39], dtype=np.float32).reshape(-1, 2),
                        np.ones((17, 1), dtype=np.float32)], axis=1).reshape(51).tolist()
                    keypoints = list(map(self._to_float, keypoints))

                    detection = {
                        "image_id": int(image_id),
                        "category_id": int(category_id),
                        "bbox": bbox_out,
                        "score": float("{:.2f}".format(score)),
                        "keypoints": keypoints
                    }
                    detections.append(detection)
        return detections

    def __len__(self):
        return self.num_samples

    def save_results(self, results, save_dir):
        # TODO: finish the conver eval format and exit
        return
        json.dump(self.convert_eval_format(results),
                  open('{}/results.json'.format(save_dir), 'w'))

    def run_eval(self, results, save_dir, **kwargs):
        """Run the evaluation on the detection results of all the test data

        Args:
            results (dict):       The dictionary of the detection results[ind] = result, \
                where result is the dictionary contains: \
                    - locations (N_pred, 3) - The translation from 
                    - quaternions (N_pred, 4)
                    - widths (N_pred, ) \\
            save_dir (str):         The saving directory \\
            kwargs:\\
                angle_th (float):   The threshold to determine the alignment between two SO(3). \
                    Defaults to pi/4 degrees \\
                dist_th (float):    The threshold to determine the alignment between two R^3. \
                    Defaults to 0.02 meter \\
                rot_sample (int):   The number of the rotational samples of the GT grasps \\
                trl_sample (int):   The number of the translational samples of the GT grasps \\
                correct_lr (bool):  Whether to correct the left-right flip of the grasps \\
                pred_frame (str):   "camera" or "tabletop"
        Returns:
            eval_results(dict):       The dictionary of the evaluation results: results[ind] = result, \
                    where result is the dictionary contains: \
                        - pred_succ (N_pred, ) - Whether the prediction is successful \\
        """

        self.save_results(results, save_dir)

        eval_results = {}

        # parse args
        rot_sample = kwargs["rot_sample"] if "rot_sample" in kwargs else None
        trl_sample = kwargs["trl_sample"] if "trl_sample" in kwargs else None

        # the metrics calculator
        metric_calculators = OrderedDict()
        metric_calculators["all"] = MetricCalculator(self.opt)
        for obj_type in ["cuboid", "cylinder", "sphere", "semi_sphere", "stick", "ring"]:
            metric_calculators[obj_type] = MetricCalculator(self.opt)

        tqdm_bar = tqdm(total=len(results))
        print("\nEvaluating the results...")
        for ind, (data_id, predicts) in enumerate(results.items()):
            scene_idx = self.scene_idxs[data_id]
            cam_idx = self.camera_idxs[data_id]

            # get scene info
            _, camera_poses, obj_types, _, _, _, _, _ = \
                self._get_scene_info(scene_idx)
            camera_pose = camera_poses[cam_idx, :, :]

            # the GT info and the grasp results
            grasp_poses_gt, _, grasp_widths_gt = \
                self._get_gt_grasps(
                    scene_idx, cam_idx, filter_collides=(not self.opt.no_collide_filter),
                    trl_sample_num=trl_sample, rot_sample_num=rot_sample, 
                    # correct_rl=self.opt.correct_rl - not necessary since the evaluation will contain both
                )
            N_obj = len(obj_types)

            # recover the predicted grasp poses in the world frame
            locations = predicts["locations"]
            quaternions = predicts["quaternions"]
            grasp_widths_pred = predicts["widths"]
            N_grasps = locations.shape[0]
            grasp_poses_pred = np.zeros((N_grasps, 4, 4), dtype=float)
            mask = np.zeros((N_grasps,), dtype=bool)

            for i in range(N_grasps):
                try:
                    r = R.from_quat(quaternions[i, :])
                    pose_pred = create_homog_matrix(
                        R_mat=r.as_matrix(), T_vec=locations[i, :])
                    # if the prediction is in the camera frame
                    if (not "pred_frame" in kwargs) or (kwargs["pred_frame"]=="camera"):
                        grasp_pose_pred_this = camera_pose @ pose_pred
                    # if the prediction is in the tabletop frame
                    elif kwargs["pred_frame"] == "tabletop":
                        grasp_pose_pred_this = pose_pred
                    else:
                        raise NotImplementedError
                    grasp_poses_pred[i, :, :] = grasp_pose_pred_this
                    mask[i] = True
                except:
                    continue
            grasp_poses_pred = grasp_poses_pred[mask, :, :]

            # store the predict number
            for key, val in metric_calculators.items():
                metric_calculators[key].grasp_pred_num += N_grasps

            # iterate through the objects in the scene
            N_obj = len(obj_types)
            pred_succ = np.zeros((N_grasps,), dtype=bool)
            for i in range(N_obj):

                # the gt grasp poses in the world frame
                grasp_poses_gt_obj = grasp_poses_gt[i]
                grasp_widths_gt_obj = grasp_widths_gt[i]

                # evaluate on this object
                pred_num, pred_succ_num, gt_num, gt_cover_num, pred_succ_this, gt_covered = \
                    self._eval_grasps_with_gt(grasp_poses_pred, grasp_widths_pred, grasp_poses_gt_obj, grasp_widths_gt_obj,
                                            ignore_mask=pred_succ,
                                              **kwargs)
                pred_succ = np.logical_or(pred_succ, pred_succ_this)
                # pdb.set_trace()

                # skip if no GT grasp is available
                if gt_num == 0:
                    continue

                # parse results
                # import pdb; pdb.set_trace()
                metric_calculators["all"].update(0, gt_num, pred_succ_num, gt_cover_num)
                metric_calculators[obj_types[i]].update(0, gt_num, pred_succ_num, gt_cover_num)
            
            # store the eval results
            eval_results[data_id] = pred_succ

            # update tqdm
            tqdm_bar.update(1)

        # calculate the metrics and print out
        print("The grasp success rate / grasp coverage rate / object success rate:")
        for key, val in metric_calculators.items():
            # If multi-object scenario, grasp success rate for individual object is meaningless 
            # since we don't have information regarding which predicted grasp is targetting which object
            if self.opt.ps_data_mode == "multi" and key != "all":
                NA_gsr = True
            else:
                NA_gsr = False
            metric_calculators[key].print_results(key, NA_gsr=NA_gsr)
        
        return eval_results

    def _load_gt_grasps(self, scene_idx, trl_sample_num=-1, rot_sample_num=-1):
        """Load the GT grasps for a scene. Optionally can resample the grasps rather than loading from the annotation
        with the translational sample number and the rotational number to t

        Args:
            scene_idx (int): The scene index
            cam_idx (int): The camera index
            trl_sample_num (int): The number of translational samples
            rot_sample_num (int): The number of rotational samples
        Returns:
            grasp_poses_gt (N_obj, N_grasps, 4, 4): The GT grasp poses in the world frame
            grasp_widths_gt (N_obj, N_grasps): The GT grasp widths
            grasp_collisions_gt (N_obj, N_grasps): The GT grasp collisions
        """
        intrinsic, camera_poses, obj_types, obj_dims, obj_poses, grasp_poses_gt, grasp_widths_gt, grasp_collisions_gt = \
            self._get_scene_info(scene_idx)

        if (trl_sample_num > 0 ) or (rot_sample_num > 0):
            # reconstruct the scene to get the grasps
            scene = SceneRender.construct_scene_obj_info(
                obj_types, obj_dims, obj_poses, camera_poses, 
                trl_sample_num=trl_sample_num,
                rot_sample_num=rot_sample_num
            )

            # get the GT grasp poses
            grasp_poses_gt, grasp_widths_gt, grasp_collisions_gt = scene.get_grasp_infos()
        
        return grasp_poses_gt, grasp_widths_gt, grasp_collisions_gt

    def _get_gt_grasps(self, scene_idx, cam_idx, filter_collides=True, center_proj=False, trl_sample_num=-1, rot_sample_num=-1,
        correct_rl=False, get_both=False, 
        scale_kpts=False, scale_coeff_k=1,
        grasp_pose_frame="world"):
        """Get the GT grasp infos for a camera in a scene,
        including the grasp poses, keypoint projections, the open widths, and optionally the center projection

        Optionally, it can correct the grasp poses and the keypoint coordinates so that 
        the projection of the left keypoint is on the actually on the left of the image.

        Args:
            scene_idx (_type_): _description_
            cam_idx (_type_): _description_
            filter_collides (bool): Filter out the collided grasps
            center_proj (bool): The option to get the center projections.
            trl_sample_num (int): The number of translational samples
            rot_sample_num (int): The number of rotational samples
            correct_rl (bool): Only obtain the set of poses and the kpts s.t. the left keypoint is on the left of the image
            get_both (bool): Get both poses and keypoints for a grasp. Will overwrite the correct_rl option
            grasp_pose_frame (str):  The grasp pose frame. ["world", "camera"]. Defaults to "world"
        Returns:
            poses_all (List[array], (N_obj, ) of (N_grasps, 4, 4)): The GT grasp poses, in world frame by dataset design. \\
            kpts_all (List[array]. (N_obj, ) of (N_grasp, N_kpts, 2)) \\
            widths_all (List[array]. (N_obj, ) of (N_grasp, )) \\
            centers_all (List[array]. (N_obj, ) of (N_grasp, 2)). Only return if center_proj

            NOTE: the N_obj will be doubled if the get_both is True

            grasp_pose_pred_this = camera_pose @ pose_pred

        """
        # camera info
        intrinsic, camera_poses, obj_types, obj_dims, obj_poses, _, _, _ = \
            self._get_scene_info(scene_idx)
        cam_pose = camera_poses[cam_idx, :, :]

        # the GT grasp poses
        grasp_poses, grasp_widths, grasp_collisions = \
            self._load_gt_grasps(scene_idx, trl_sample_num=trl_sample_num, rot_sample_num=rot_sample_num)



        # get the GT keypoint projection
        N_objs = len(grasp_poses)
        poses_all = []
        kpts_all = []
        widths_all = []
        centers_all = []
        for i in range(N_objs):
            kpts_obj = []
            widths_obj = []
            centers_obj = []
            grasp_poses_obj = [] # for storing the non-colliding grasp poses
            grasp_poses_this = grasp_poses[i]
            grasp_widths_this = grasp_widths[i]
            grasp_collids_this = grasp_collisions[i]
            # grasp scales
            grasp_trls_this = np.stack(grasp_poses_this, axis=0)[:, :3, 3]
            grasp_scales_this = np.linalg.norm(grasp_trls_this, ord=2, axis=1) 
            for j in range(grasp_poses_this.shape[0]):
                grasp_pose = grasp_poses_this[j, :, :]
                grasp_width = grasp_widths_this[j]
                grasp_collide = grasp_collids_this[j]

                if filter_collides and grasp_collide:
                    continue

                if not scale_kpts:
                    if self.canonical_open_width is not None:
                        proj_width = self.canonical_open_width
                    elif self.min_open_width is not None:
                        proj_width = grasp_width if grasp_width > self.min_open_width \
                            else self.min_open_width
                    else:
                        proj_width =  grasp_width    

                if scale_kpts:
                    # get scale
                    scale = grasp_scales_this[j]
                    # proj width
                    proj_width = scale * grasp_width * scale_coeff_k

                grasp = Grasp(proj_width, pose=grasp_pose,
                            kpts_option=self.kpt_type)
                # the keypoint coordinate in the world frame, (N_kpt, 3)
                kpts_coord = grasp.get_kpts(frame="world")
                kpts_img = kpts_3d_to_2d(
                    intrinsic, np.linalg.inv(cam_pose), kpts_coord)
                
                # the centers projection
                if center_proj:
                    # (3, )
                    centers_coord = (kpts_coord[0] + kpts_coord[1]) / 2
                    # (2, )
                    centers_img = kpts_3d_to_2d(
                        intrinsic, np.linalg.inv(cam_pose), centers_coord.reshape(1, -1)
                    )[0]
                    centers_obj.append(centers_img)
                
                kpts_obj.append(kpts_img)
                widths_obj.append(grasp_width)
                if grasp_pose_frame == "world":
                    grasp_poses_obj.append(grasp_pose)
                elif grasp_pose_frame == "camera":
                    grasp_poses_obj.append(np.linalg.inv(cam_pose) @ grasp_pose)
                else:
                    return 

            if len(kpts_obj)  == 0:
                kpts_all.append(np.zeros(shape=(0, 4, 2)))
                widths_all.append(np.zeros(shape=(0,)))
                poses_all.append(np.zeros(shape=(0, 4, 4)))
                continue

            # correct the left and right s.t. the left keypoint is actually on the left side of the image
            kpts_obj = np.array(kpts_obj)
            widths_obj = np.array(widths_obj)
            grasp_poses_obj = np.array(grasp_poses_obj)

            if get_both:
                kpts_obj_flip, grasp_poses_obj_flip = self._get_grasp_doubleganger(kpts_obj, grasp_poses_obj)
                kpts_obj = np.concatenate([kpts_obj, kpts_obj_flip], axis=0)
                grasp_poses_obj = np.concatenate([grasp_poses_obj, grasp_poses_obj_flip], axis=0)
                widths_obj = np.concatenate([widths_obj, widths_obj], axis=0)
            elif correct_rl:
                kpts_obj, grasp_poses_obj = self._correct_lf_kpts(kpts_obj, grasp_poses_obj)

            kpts_all.append(np.array(kpts_obj))
            widths_all.append(np.array(widths_obj))
            poses_all.append(grasp_poses_obj)

            if center_proj:
                centers_all.append(np.array(centers_obj))

        if center_proj:
            return poses_all, kpts_all, widths_all, centers_all
        else:
            return poses_all, kpts_all, widths_all
        
    
    def _get_grasp_doubleganger(self, kpts_img, grasp_poses):
        """Get the flipping of the current pose and the keypoints

        Args:
            kpts_img (N_grasps, N_kpts, 2):         The current 2d keypoints
            grasp_pose (N_grasps, 4, 4):            The current grasp poses. Corresponds to the kpts_img
        Returns:
            kpts_img_flip (N_grasps, N_kpts, 2):    The flipped keypoints.
            grasp_pose (N_grasps, N_kpts, 2):       The flipped poses.
        """

        # flip the keypoints
        kpts_img_new = np.copy(kpts_img)
        for e in self.flip_idx:
            kpts_img_new[:, e[0], :], kpts_img_new[:, e[1], :] = kpts_img[:, e[1], :], kpts_img[:, e[0], :]

        # flip the grasp poses
        grasp_poses_new = self._rotate_poses_180_by_x(grasp_poses)

        return kpts_img_new, grasp_poses_new

    
    def _correct_lf_kpts(self, kpts_img, grasp_poses):
        """Correct the left keypoint s.t. the left keypoint is actually on the left side of the image.
        Only implemented for the box type for now

        Args:
            kpts_img (array): The keypoint coordinate in the image frame, (N_grasps, N_kpt, 2)
            grasp_pose (array): The grasp pose, (N_grasps, 4, 4)

        Returns:
            kpts_img_new (array): The corrected keypoint coordinate in the image frame, (N_grasps, N_kpt, 2)
            grasp_pose (array): The corrected grasp pose, (N_grasps, 4, 4)
        """

        # Get the flipped kpts and poses
        kpts_flipped, poses_flipped = self._get_grasp_doubleganger(kpts_img, grasp_poses)

        # switch mask - all keypoint types have the left and right with the same index
        kpts_l = kpts_img[:, BoxVertexType.Left, :]
        kpts_r = kpts_img[:, BoxVertexType.Right, :]
        switch_mask = kpts_l[:, 0] > kpts_r[:, 0]

        # switch
        kpts_img[switch_mask, :, :] = kpts_flipped[switch_mask, :, :]
        grasp_poses[switch_mask, :, :] = poses_flipped[switch_mask, :, :]

        return kpts_img, grasp_poses

    def _rotate_poses_180_by_x(self, poses):
        poses_new = np.copy(poses)
        # correct the pose correspondingly. Rotate along the x axis by 180 degrees
        M_rot = create_homog_matrix(
            R_mat=create_rot_mat_axisAlign([1, -2, -3]),
            T_vec=np.zeros((3, )) 
        )
        poses_new = poses_new @ M_rot 
        return poses_new


    def _eval_grasps_with_gt(self, grasp_poses_pred, grasp_widths_pred, grasp_poses_gt_obj, grasp_widths_gt_obj,
                            ignore_mask = None,
                             angle_th=np.pi/4, dist_th=0.02, **kwargs):
        """Evaluate a set of predicted grasps by comparing to a set of ground truth grasps

        Args:
            grasp_poses_pred (array, (N_pred, 4, 4)):   The predicted homogeneous grasp poses in the world frame. 
            grasp_widths_pred (array, (N_pred, )): The predicted grasp open widths
            grasp_poses_gt_obj (N_grasp_poses): The ground truth homogeneous grasp poses in the world frame
            grasp_widths_gt_obj (_type_):  The ground truth grasp open widths. 
            ignore_mask (array (N_pred)):   The mask for ignoring the predicted grasps during the pred_succ_num counting
            angle_th (float): The threshold for the angular difference
            dist_th (float): The threshold for the translation difference

        Returns:
            pred_num (int):             The number of the predicted grasps
            pred_succ_num (int):        The numebr of the successfully predicted grasps
            gt_num (int):               The number of the GT grasps
            gt_cover_num (int):         The numebr of the GT grasps that is covered by the predicted set
        """
        pred_rotations = grasp_poses_pred[:, :3, :3]  # (N_pred, 3, 3)
        pred_translates = grasp_poses_pred[:, :3, 3]  # (N_pred, 3)
        gt_rotations_1 = grasp_poses_gt_obj[:, :3, :3]  # (N_gt, 3, 3)
        gt_translates = grasp_poses_gt_obj[:, :3, 3]  # (N_gt, 3)
        grasp_poses_gt_obj_2 = self._rotate_poses_180_by_x(grasp_poses_gt_obj)
        gt_rotations_2 = grasp_poses_gt_obj_2[:, :3, :3]  # (N_gt, 3, 3)

        # the numbers
        pred_num = pred_rotations.shape[0]
        gt_num = gt_translates.shape[0]

        # SO(3) distances - minimum rotation angle
        angle_dist_1 = self._get_SO3_dist(pred_rotations, gt_rotations_1)
        angle_dist_2 = self._get_SO3_dist(pred_rotations, gt_rotations_2)
        

        # Translation distance
        translates_diff = pred_translates[:, np.newaxis,
                                          :] - gt_translates[np.newaxis, :, :]
        translates_dist = np.linalg.norm(
            translates_diff, axis=2)   # (N_pred, N_gt)

        # match matrix - (N_pred, N_gt)
        matches = np.logical_and(
            np.logical_or(angle_dist_1 < angle_th, angle_dist_2 < angle_th),
            translates_dist < dist_th
        )

        # get the success number
        pred_succ = np.any(matches, axis=1)
        gt_covered = np.any(matches, axis=0)
        assert pred_succ.size == pred_num
        assert gt_covered.size == gt_num

        if ignore_mask is None:
            ignore_mask = np.zeros(pred_num, dtype=bool)
        
        pred_succ_num = np.count_nonzero(pred_succ[np.logical_not(ignore_mask)])
        gt_cover_num = np.count_nonzero(gt_covered)

        # # debug
        # if np.any(~pred_succ) or np.any(~gt_covered):
        #     print("The angular distance: {}\n".format(angle_dist))
        #     print("\n The trace: {} \n".format(trace))

        #     print("The translate distance: {}".format(translates_dist))

        return pred_num, pred_succ_num, gt_num, gt_cover_num, pred_succ, gt_covered
    
    def _get_SO3_dist(self, rotMat1, rotMat2):
        """Get the SO3 distance from 2 set of rotation matrices

        Args:
            rotMat1 (N1, 3, 3)
            rotMat2 (N2, 3, 3)
        Returns:
            angle_dists (N1, N2)
        """
        rotMat1_inv = np.linalg.inv(rotMat1)[
            :, np.newaxis, :, :]  # (N1, 1, 3, 3)
        rotMat2_tmp = rotMat2[np.newaxis, :, :, :]  # (1, N2, 3, 3)
        # (N_pred, N_gt, 3, 3)
        mul_result = np.matmul(rotMat1_inv, rotMat2_tmp)
        # (N_pred, N_gt)
        trace = np.trace(mul_result, axis1=2, axis2=3)

        cos_angles = (trace - 1) / 2
        # improve the numerical stability
        cos_angles[cos_angles <= -1] = -1 + 1e-5
        # improve the numerical stability
        cos_angles[cos_angles >= 1] = 1 - 1e-5
        angle_dists = np.arccos(
            cos_angles
        )

        return angle_dists

    def _to_float(self, x):
        return float("{:.2f}".format(x))
    
    def get_pc(self, idx, frame="camera", flatten=True):
        """Get the point cloud

        Args:
            idx: The data index
            frame: In which frame. choices: ["camera", "tabletop"]
            flatten: If true, get (H*W,3). Else get (H, W, 3)
        """
        scene_idx = self.scene_idxs[idx]
        cam_idx = self.camera_idxs[idx]
        # get data paths
        depth_path = self.depths[idx]

        depth_raw = np.load(depth_path)

        # scene info - intrinsic and extrinsic
        intrinsic, camera_poses, _, _, _, _, _, _ = self._get_scene_info(scene_idx)
        extrinsic = np.linalg.inv(camera_poses[cam_idx, :, :])
        
        # get and return pc
        pc = depth2pc(depth_raw, intrinsic, frame=frame, extrinsic=extrinsic, flatten=flatten, remove_zero_depth=True)
        return pc

    
 


