from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

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

class GraspPoseDataset(data.Dataset):

    def _augment_depth(self, depth):
        """
        Augments depth map with z-noise and smoothing according to config
        Arguments:
            depth {np.ndarray} -- depth map
        Returns:
            np.ndarray -- augmented depth map
        Source: https://github.com/NVlabs/contact_graspnet/blob/main/contact_graspnet/data.py#L538-L560
        """

        if self.opt.depth_aug_sigma > 0:
            clip = self.opt.depth_aug_clip
            sigma = self.opt.depth_aug_sigma
            noise = np.clip(sigma*np.random.randn(*depth.shape), -clip, clip)
            depth += noise
        if self.opt.depth_aug_gaussian_kernel > 0:
            kernel = self.opt.depth_aug_gaussian_kernel 
            depth_copy = depth.copy()
            depth = cv2.GaussianBlur(depth,(kernel,kernel),0)
            depth[depth_copy==0] = depth_copy[depth_copy==0]
                
        return depth

    def __getitem__(self, idx: int):
        """ Get the input and the ground truth for the keypoint detection training 

        """
        scene_idx = self.scene_idxs[idx]
        cam_idx = self.camera_idxs[idx]
        # get data paths, color data, depth data, segmentation mask
        color_path = self.images[idx]
        depth_path = self.depths[idx]
        seg_path = self.segs[idx]

        img = cv2.imread(color_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        depth_raw = np.load(depth_path)
        seg_mask = cv2.imread(seg_path, cv2.IMREAD_GRAYSCALE)

        # augment depth
        depth = self._augment_depth(depth_raw)

        # the cropping center coordinate c
        height, width = img.shape[0], img.shape[1]
        c = np.array([img.shape[1] / 2., img.shape[0] / 2.], dtype=np.float32)
        # maximum dimension s
        s = max(img.shape[0], img.shape[1]) * 1.0
        rot = 0

        # The grasp_poses (N_grasps, 4, 4); 2d kpts (N_grasps, N_kpts, 2); grasp_widths (N_grasps, ); and centers (N_grasps, 2)
        grasp_poses, kpts_2d, grasp_widths, centers_2d = self._get_gt_grasps(
            scene_idx=scene_idx, cam_idx=cam_idx, filter_collides=(not self.opt.no_collide_filter),
            center_proj=True, 
            correct_rl=self.opt.correct_rl,
            scale_kpts=self.opt.scale_kpts_mode, 
            scale_coeff_k=self.opt.scale_coeff_k,
            grasp_pose_frame="camera"
        )
        grasp_poses = np.concatenate(grasp_poses, axis=0)
        kpts_2d = np.concatenate(kpts_2d, axis=0)
        grasp_widths = np.concatenate(grasp_widths, axis=0)
        centers_2d = np.concatenate(centers_2d, axis=0)
        num_grasps = min(kpts_2d.shape[0], self.max_grasps)

        # grasp scales
        grasp_trls = grasp_poses[:, :3, 3]
        grasp_scales = np.linalg.norm(grasp_trls, ord=2, axis=1)

        # input
        inp = img.astype(np.float32)
        if self.opt.input_mod == "RGD":
            inp[:, :, 2] = depth
        elif self.opt.input_mod == "RGBD":
            inp = np.concatenate([inp, depth[:,:, None]], axis=2) 
        elif self.opt.input_mod == "RGB":
            pass
        elif self.opt.input_mod == "D":
            inp = depth[:, :, None]
        else:
            raise NotImplementedError
        

        if self.opt.inspect_aug:
            cv2.imshow("original_image", inp)

        ##############################################
        # Transform the input image
        # Including: random crop, rotate, flip, color jittering
        ###############################################
        flipped = False
        if self.split == 'train':
            # random square cropping
            if not self.opt.not_rand_crop:
                s = s * np.random.choice(np.arange(0.6, 1.4, 0.1))
                # crop to 128, which is the output size with input_size = 512 and down_ratio=4
                w_border = self._get_border(128, img.shape[1])
                h_border = self._get_border(128, img.shape[0])
                c[0] = np.random.randint(
                    low=w_border, high=img.shape[1] - w_border)
                c[1] = np.random.randint(
                    low=h_border, high=img.shape[0] - h_border)

                # NOTE: fix the center for debug
                if self.opt.fix_crop:
                    s = max(img.shape[0], img.shape[1]) * 1.0
                    c = np.array([img.shape[1] / 2., img.shape[0] / 2.], dtype=np.float32)
            else:
                # If not random crop, apply random shift and scaling
                sf = self.opt.scale
                cf = self.opt.shift
                c[0] += s * np.clip(np.random.randn()*cf, -2*cf, 2*cf)
                c[1] += s * np.clip(np.random.randn()*cf, -2*cf, 2*cf)
                s = s * np.clip(np.random.randn()*sf + 1, 1 - sf, 1 + sf)

            # random rotation - NOTE: aug_rot is set to 0, so no rotation is applied
            if np.random.random() < self.opt.aug_rot:
                rf = self.opt.rotate
                rot = np.clip(np.random.randn()*rf, -rf*2, rf*2)

            # random flip
            if np.random.random() < self.opt.flip:
                flipped = True
                inp = inp[:, ::-1, :]
                c[0] = width - c[0] - 1

        # apply the geometric augmentation
        trans_input = get_affine_transform(
            c, s, rot, [self.opt.input_res, self.opt.input_res])

        inp = cv2.warpAffine(inp, trans_input,
                             (self.opt.input_res, self.opt.input_res),
                             flags=cv2.INTER_LINEAR)

        # Somehow warp affine will remove the single channel-number axis.
        if self.opt.input_mod == "D":
            inp = inp[:, :, None]

        # normalize the input - only the RG channel
        if self.opt.input_mod == "RGD":
            inp[:, :, :2] = inp[:, :, :2]  / 255.
        elif self.opt.input_mod == "RGBD" or self.opt.input_mod == "RGB":
            inp[:, :, :3] = inp[:, :, :3]  / 255.

        if self.opt.inspect_aug:
            cv2.imshow("Augmented image before color jitter",
                       (inp*255).astype(np.uint8))

        # color augmentation
        if self.split == 'train' and not self.opt.no_color_aug:
            if self.opt.input_mod == "D":
                pass
            elif self.opt.input_mod == "RGD":
                color_aug(self._data_rng, inp, self._eig_val, self._eig_vec)
            else:
                inp_color = inp[:, :, :3]
                color_aug(self._data_rng, inp_color, self._eig_val, self._eig_vec)
                inp[:,:,:3] = inp_color

        if self.opt.inspect_aug:
            cv2.imshow("Augmented image", (inp*255).astype(np.uint8))

        # standardize
        inp = (inp - self.mean) / self.std

        # (H, W, 3) to (3, H, W) for pytorch
        inp = inp.transpose(2, 0, 1)

        ######################################################################
        # Generate and Transform the labels
        ######################################################################

        output_res = self.opt.output_res
        num_kpts = self.num_grasp_kpts
        num_vpts = 2    # NOTE: hard code for now
        clf_num = self.opt.ori_num
        trans_output_rot = get_affine_transform(
            c, s, rot, [output_res, output_res])
        trans_output = get_affine_transform(c, s, 0, [output_res, output_res])

        hm = np.zeros((clf_num, output_res,
                      output_res), dtype=np.float32)
        hm_kpts = np.zeros((num_kpts, output_res, output_res),
                         dtype=np.float32)
        w = np.zeros((self.max_grasps, clf_num), dtype=np.float32)
        w_mask = np.zeros((self.max_grasps, clf_num), dtype=np.uint8)
        kpts_center_offset = np.zeros((self.max_grasps, clf_num * num_kpts * 2), dtype=np.float32)
        kpts_center_mask = np.zeros((self.max_grasps, clf_num * num_kpts * 2), dtype=np.uint8)
        reg = np.zeros((self.max_grasps, 2), dtype=np.float32)
        ind = np.zeros((self.max_grasps), dtype=np.int64)
        ori_clses = np.zeros((self.max_grasps, ), dtype=np.int64)
        reg_mask = np.zeros((self.max_grasps), dtype=np.uint8)
        kpts_offset = np.zeros((self.max_grasps * num_kpts, 2), dtype=np.float32)
        kpts_ind = np.zeros((self.max_grasps * num_kpts), dtype=np.int64)
        kpts_mask = np.zeros((self.max_grasps * num_kpts), dtype=np.int64)
        vpts = np.zeros((self.max_grasps * num_vpts, 2), dtype=np.float32) # The vanishing point. 2 points for each grasp
        vpts_fin_mask = np.zeros((self.max_grasps * num_vpts, 2), dtype=np.uint8)      # The mask for the vpts at the finite
        vpts_inf_mask = np.zeros((self.max_grasps * num_vpts, 2), dtype=np.uint8)      # The mask for the vpts at the infinite
        scales = np.zeros((self.max_grasps, clf_num), dtype=np.float32)
        scales_mask = np.zeros((self.max_grasps, clf_num), dtype=np.uint8)

        draw_gaussian = draw_msra_gaussian if self.opt.mse_loss else \
            draw_umich_gaussian

        gt_det = []
        ct_ori_cache = {}
        for k in range(num_grasps):

            # The keypoints for that grasp (N_kpt, 2), the grasp_width (1, ), and the center (2, )
            pts = kpts_2d[k]
            grasp_w = grasp_widths[k]
            ct = centers_2d[k]
            grasp_scale = grasp_scales[k]

            # the vanishing points and the mask for the grasp (N_vpts, 2)
            vpts_this, vpts_fin_mask_this = get_vanishing_points(pts.reshape(1, 8))
            vpts_this = vpts_this.reshape(num_vpts, 2)
            vpts_fin_mask_this = vpts_fin_mask_this.reshape(num_vpts, 2)

            # apply the flipping augmentation on kpts
            if flipped:
                # flip the y coordinates
                pts[:, 0] = width - pts[:, 0] - 1
                ct[0] = width - ct[0] - 1
                vpts_this[:, 0] = width - vpts_this[:, 0] - 1
                # for keypoints, flip the kpt idx since left become right and right becomes left
                for e in self.flip_idx:
                    pts[e[0]], pts[e[1]] = pts[e[1]].copy(), pts[e[0]].copy()
            
            # apply the affine transform (cropping). If out of the range, skip that grasp
            ct = affine_transform(ct, trans_output_rot)
            skip_flag = (
                (ct[0] < 0) or (ct[0] >= output_res) or \
                    (ct[1] < 0) or (ct[1] >= output_res)
            )
            for j in range(num_kpts):
                pts[j, :] = affine_transform(pts[j, :], trans_output_rot)
                skip_flag = skip_flag or (
                    (pts[j, 0] < 0) or (pts[j, 0] >= output_res) or \
                        (pts[j, 1] < 0) or (pts[j, 1] >= output_res)
                )
            if skip_flag:
                continue
            for j in range(num_vpts):
                vpts_this[j, :] = affine_transform(vpts_this[j, :], trans_output_rot)

            # The Gaussian radius - NOTE: hardcode for now. The original paper adaptively adjust the radius
            #radius = gaussian_radius((math.ceil(h), math.ceil(w)))
            radius = 1.5
            radius = self.opt.hm_gauss if self.opt.mse_loss else max(
                0, int(radius))

            ct_int = ct.astype(np.int32)
            ind_this = ct_int[1] * output_res + ct_int[0]
            
            # get the orientation
            ori_cls = get_ori_cls(pts, range_mode=0, total_cls_num=clf_num)[0]

            # keep only one grasp for each orientaion class for each pixl
            if ind_this not in ct_ori_cache.keys():
                ct_ori_cache[ind_this] = []
                ct_ori_cache[ind_this].append(ori_cls)
            elif ori_cls not in ct_ori_cache[ind_this]:
                ct_ori_cache[ind_this].append(ori_cls)
            else:
                continue

            # Fill in the GT widths w/ mask, center subpixel(reg) offset w/ mask, index, and the ori_cls
            w[k, ori_cls] = 1. * grasp_w
            w_mask[k, ori_cls] = 1
            reg[k] = ct - ct_int
            reg_mask[k] = 1
            ind[k] = ind_this
            ori_clses[k] = ori_cls

            # Fill in the scales
            scales[k, ori_cls] = grasp_scale
            scales_mask[k, ori_cls] = 1

            # Fill in the vpts
            for i in range(num_vpts):
                vpts[k * num_vpts + i] = vpts_this[i, :]
                vpts_fin_mask[k * num_vpts + i, :] = vpts_fin_mask_this[i, :].astype(np.uint8)
                vpts_inf_mask[k * num_vpts + i, :] = (~(vpts_fin_mask_this[i, :])).astype(np.uint8)

            # The kpts radius - NOTE: hardcode for now
            #kpts_radius = gaussian_radius((math.ceil(h), math.ceil(w)))
            kpts_radius = radius
            kpts_radius = self.opt.hm_gauss \
                if self.opt.mse_loss else max(0, int(kpts_radius))

            # The keypoint coordinates 
            for j in range(num_kpts):
                # fill in the kpts-center(int) offset
                kpts_center_offset[k, j * 2 + ori_cls*2*num_kpts: j * 2 + 2 + ori_cls*2*num_kpts] = pts[j, :2] - ct_int
                # set the indicator mask of the corresponding x,y element to 1
                kpts_center_mask[k, j * 2 + ori_cls*2*num_kpts: j * 2 + 2 + ori_cls*2*num_kpts] = 1

                # fill in the kpts offset, index, and mask
                pt_int = pts[j, :2].astype(np.int32)
                kpts_offset[k * num_kpts + j] = pts[j, :2] - pt_int
                kpts_ind[k * num_kpts + j] = pt_int[1] * \
                    output_res + pt_int[0]
                kpts_mask[k * num_kpts + j] = 1


                # draw the kpt heatmap
                draw_gaussian(hm_kpts[j], pt_int, kpts_radius)

            # draw the center heatmap
            draw_gaussian(hm[ori_cls], ct_int, radius)

            # gt detection.
            gt_det_this = [ct[0], ct[1]] + pts.reshape(num_kpts * 2).tolist() + [grasp_w, 1., ori_cls]
            if self.opt.sep_scale_branch:
                gt_det_this = gt_det_this + [grasp_scale]
            gt_det.append(gt_det_this)
        if rot != 0:
            raise NotImplementedError
        ret = {'input': inp, 'hm': hm, 'reg_mask': reg_mask, 'ind': ind, 'w': w,
               'kpts_center_offset': kpts_center_offset, 'kpts_center_mask': kpts_center_mask,
               'ori_clses': ori_clses, 'w_mask': w_mask}

        # NOTE: don't need to visualize the gt center heatmap here. It can be seen in the trainer debug function
        # if self.opt.inspect_aug:
        #     import matplotlib.pyplot as plt
        #     plt.imshow(
        #         hm.max(axis=0, keepdims=True).transpose(1, 2, 0)
        #     )
        #     plt.show()

        if self.opt.reg_offset:
            ret.update({'reg': reg})
        if self.opt.kpts_refine:
            ret.update({'hm_kpts': hm_kpts})
            ret.update({'kpts_offset': kpts_offset,
                       'kpts_ind': kpts_ind, 'kpts_mask': kpts_mask})
        if self.opt.debug > 0 or not self.split == 'train':
            gt_det = np.array(gt_det, dtype=np.float32) if len(gt_det) > 0 else \
                np.zeros((1, 13), dtype=np.float32)
            meta = {'c': c, 's': s, 'gt_det': gt_det, 'img_id': idx}
            ret['meta'] = meta

        if self.opt.inspect_aug:
            cv2.waitKey()
        
        if self.opt.sep_scale_branch:
            ret.update({"scales": scales, "scales_mask": scales_mask})

        return ret
    
  