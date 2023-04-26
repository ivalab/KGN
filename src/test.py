from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths

import pdb
import os
import cv2
import numpy as np
import time
from progress.bar import Bar
import torch

from opts import opts
from logger import Logger
from utils.keypoints import kpts_3d_to_2d, plot_grasps_kpts
from utils.utils import AverageMeter
from utils.vis import construct_scene_with_grasp_preds
from datasets.dataset_factory import dataset_factory
from detectors.detector_factory import detector_factory
from pose_recover.pnp_solver_factory import PnPSolverFactory
from keypoint_graspnet import KeypointGraspNet as KGN


class PrefetchDatasetGrasp(torch.utils.data.Dataset):
    def __init__(self, opt, dataset, pre_process_func):
        self.scene_idxs = dataset.scene_idxs
        self.camera_idxs = dataset.camera_idxs
        self.images = dataset.images
        self.depths = dataset.depths

        # contains lots of utility functions
        self.dataset = dataset

        # preprocess
        self.pre_process_func = pre_process_func

        self.opt = opt 

    def __getitem__(self, index):

        # get the scene and camera index
        scene_idx = self.scene_idxs[index]
        cam_idx = self.camera_idxs[index]
        
        # Load image data
        if not self.opt.test_oracle_kpts:
            img_path = self.images[index]
            dep_path = self.depths[index]

            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            depth_raw = np.load(dep_path)

            # add noise
            if opt.depth_noise_level > 0:
                noise = np.random.normal(0, opt.depth_noise_level, size=depth_raw.shape)
                depth_raw += noise

            # determin input
            image = img.astype(np.float32)
            if self.opt.input_mod == "RGD":
                image[:, :, 2] = depth_raw
            elif self.opt.input_mod == "RGBD":
                image = np.concatenate([image, depth_raw[:,:, None]], axis=2) 
            elif self.opt.input_mod == "RGB":
                pass
            elif self.opt.input_mod == "D":
                image = depth_raw[:, :, None]
            else:
                raise NotImplementedError

            images, meta = {}, {}
            for scale in opt.test_scales:
                images[scale], meta[scale] = self.pre_process_func(
                    image, scale)

            # append the scene, cam, and data index info for debug and visualization
            meta["scene_idx"] = scene_idx
            meta["camera_idx"] = cam_idx
            meta["img_idx"] = index
            return index, {'images': images, 'image': image, 'meta': meta, "rgb": img, "depth": depth_raw}
        # Load the oracle keypoints
        else:

            # get the camera intrinsic and the oracle open width
            intrinsic, camera_poses, obj_types, obj_dims, obj_poses, grasp_poses, grasp_widths, grasp_collisions,  = \
                self.dataset._get_scene_info(scene_idx)

            # Get the oracle keypoint projection - List(N_obj) of (N_grasp, N_kpts, 2)
            _, oracle_kpts_2d, oracle_widths = \
                self.dataset._get_gt_grasps(
                    scene_idx, cam_idx, filter_collides=True, 
                    center_proj = False, 
                    correct_rl = True 
                )

            # merge to one set the grasps for each object
            kpts_2d_pred = np.concatenate(oracle_kpts_2d, axis=0)       #(N_grasp_all_obj, N_kpts, 2)
            widths_pred = np.concatenate(oracle_widths, axis=0)   #(N_grasp_all_obj, )

            # perturb with the gaussian noise
            shape=kpts_2d_pred.shape
            noise = np.random.normal(0, opt.Gaussian_noise_stddev, size=shape)
            kpts_2d_pred = kpts_2d_pred + noise

            return index, {"oracle_kpts_2d_noise": kpts_2d_pred, "oracle_widths": widths_pred, "rgb": img, "depth": depth_raw}

    def __len__(self):
        return len(self.images)


def test(opt):
    os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpus_str

    Dataset = dataset_factory[opt.dataset]
    opt = opts().update_dataset_info_and_set_heads(opt, Dataset)
    print(opt)
    Logger(opt)

    # dataset
    split = 'val' if not opt.trainval else 'test'
    dataset = Dataset(opt, split)

    # detector
    Detector = detector_factory[opt.task]
    detector = Detector(opt)

    # PnP Solver
    PNPSolver = PnPSolverFactory[opt.pnp_type] 
    pnp_solver = PNPSolver(
        kpt_type=opt.kpt_type,
        use_center=opt.use_center
    )

    # KGN
    kgn = KGN(opt, detector, pnp_solver)
    
    # dataloader that utilizes the detector's preprocess function
    set_for_loader = PrefetchDatasetGrasp(opt, dataset, detector.pre_process)
        
    data_loader = torch.utils.data.DataLoader(
        set_for_loader,
        batch_size=1, shuffle=False, num_workers=1, pin_memory=True)


    results = {}
    num_iters = len(dataset)
    bar = Bar('{}'.format(opt.exp_id), max=num_iters)
    time_stats = ['tot', 'load', 'pre', 'net', 'dec', 'post', 'merge']
    avg_time_stats = {t: AverageMeter() for t in time_stats}
    total_avg_timer = AverageMeter()

    # shape counter
    shape_counter = {
        'cuboid': 0,
        'cylinder': 0,
        'stick': 0,
        'ring': 0,
        'sphere': 0,
        'semi_sphere': 0
    }

    #for ind in range(num_iters):
    for ind, (img_id, oracle_dets_or_pre_processed_images) in enumerate(data_loader):

        np.random.seed(img_id)      # for the consistent object color

        # debug: - For the scene 145
        #if not ind in [196, 197, 198, 199]:
        #     contin
        # if ind > 100:
        #     break

        # get the scene and camera index
        scene_idx = dataset.scene_idxs[img_id]
        cam_idx = dataset.camera_idxs[img_id]

        # get the camera intrinsic and the oracle open width
        intrinsic, camera_poses, obj_types, obj_dims, obj_poses, grasp_poses, grasp_widths, grasp_collisions,  = \
            dataset._get_scene_info(scene_idx)
        kgn.set_cam_intrinsic_mat(intrinsic)

        # # for debug:
        # if not "cuboid" in obj_types:
        #     continue
        
        # limit the testing number for debugging
        if opt.test_num_per_shape > 0:
            if all(shape_counter[obj_type] >= opt.test_num_per_shape for obj_type in ["cuboid", "cylinder", "stick", "ring", "sphere", "semi_sphere"]):
                break
            elif shape_counter[obj_types[0]] >= opt.test_num_per_shape:
                continue
            else:
                shape_counter[obj_types[0]] += 1
        
        # obtain the detection results
        time_start = time.time()

        # apply KGN
        quaternions, locations, widths, kpts_2d_pred, scores = kgn.generate(oracle_dets_or_pre_processed_images)
        

        Bar.suffix = '[{0}/{1}]|Tot: {total:} |ETA: {eta:} '.format(
            ind, num_iters, total=bar.elapsed_td, eta=bar.eta_td)
        # for t in avg_time_stats:
        #     avg_time_stats[t].update(ret[t])
        #     Bar.suffix = Bar.suffix + '|{} {tm.val:.3f}s ({tm.avg:.3f}s) '.format(
        #         t, tm=avg_time_stats[t])

        
        time_end = time.time()
        total_avg_timer.update(time_end - time_start, n=1)

        # save to the results
        results_this = {}
        results_this["locations"] = np.array(locations)
        results_this["quaternions"] = np.array(quaternions)
        results_this['widths'] = np.array(widths) #* 3
        results[img_id.numpy().astype(np.int32)[0]] = results_this

        # visualize results
        if opt.vis_results or opt.debug == 5:
            img_id = img_id.numpy().astype(np.int32)[0]

            # sample color
            obj_color = [np.random.choice(range(256), size=3) for _ in range(len(obj_types))]
            
            # evaluate this result
            results_tmp = {}
            results_tmp[img_id] = results_this
            # pred_succ = dataset.run_eval(results_tmp, opt.save_dir, angle_th=opt.angle_th, dist_th=opt.dist_th, \
            #     rot_sample=opt.rot_sample_num, trl_sample=opt.trl_sample_num)[img_id]

            # make the small widths bigger so that the visualization is clearer
            widths_show = results_this['widths']
            widths_show[widths_show < 0.02] = 3 * widths_show[widths_show < 0.02]
            s = construct_scene_with_grasp_preds(
                obj_types=obj_types,
                obj_dims=obj_dims,
                obj_poses=obj_poses,
                obj_colors= obj_color,
                camera_pose=camera_poses[cam_idx],
                grasp_results=results_this,
                grasp_color=[0, 0, 255],
                # grasp_succ=pred_succ,
            )

            # load image
            img_path = dataset.images[img_id]
            color = cv2.imread(img_path)
            color = cv2.cvtColor(color, cv2.COLOR_BGR2RGB)

            # plot the predicted keypoints
            img_kpts_pred = plot_grasps_kpts(color, kpts_2d_pred, kpts_mode=opt.kpt_type, size=5)

            # plot the GT keypoints
            poses_gt, kpts_2d_gt, grasp_widths_gt = \
                dataset._get_gt_grasps(
                    scene_idx, cam_idx, filter_collides=True, 
                    center_proj=False, 
                    correct_rl = True
                )
            kpts_2d_gt = np.concatenate(kpts_2d_gt, axis=0)
            img_kpts_gt = plot_grasps_kpts(color, kpts_2d_gt, kpts_mode=opt.kpt_type, size=5)

            # # see the GT
            grasp_widths_gt = np.concatenate(grasp_widths_gt, axis=0)
            grasp_widths_gt[grasp_widths_gt < 0.02] = 3 * grasp_widths_gt[grasp_widths_gt < 0.02]
            results_gt = {
                "poses": np.concatenate(poses_gt, axis=0), 
                "widths": grasp_widths_gt
            }
            s_gt = construct_scene_with_grasp_preds(
                obj_types=obj_types,
                obj_dims=obj_dims,
                obj_poses=obj_poses,
                obj_colors= obj_color,
                camera_pose=camera_poses[cam_idx],
                grasp_results=results_gt,
                grasp_color=[0, 0, 255],
            )


            if opt.debug == 5:
                # re-render image so that the color is consistent
                colors_pure, _ = s.render_imgs(instance_masks=False, grasp_mode=-1)
                # plot the predicted keypoints
                img_kpts_pred_save = plot_grasps_kpts(colors_pure[0], kpts_2d_pred, kpts_mode=opt.kpt_type, size=5)
                img_kpts_gt_save = plot_grasps_kpts(colors_pure[0], kpts_2d_gt, kpts_mode=opt.kpt_type, size=5)
                # save the img_kpts_pred, img_kpts_gt
                cv2.imwrite(
                    os.path.join(opt.debug_dir, '{}_kpts_pred.png'.format(img_id)), 
                    img_kpts_pred_save[:,:,::-1]
                )
                cv2.imwrite(
                    os.path.join(opt.debug_dir, '{}_kpts_gt.png'.format(img_id)), 
                    img_kpts_gt_save[:,:,::-1]
                )

                # save the 3d grasp
                colors, _ = s.render_imgs(instance_masks=False, grasp_mode=0)
                cv2.imwrite(
                    os.path.join(opt.debug_dir, '{}_grasp_pred.png'.format(img_id)),
                    colors[0][:,:,::-1]
                )

                # save the 3d grasp
                colors, _ = s.render_imgs(instance_masks=False, grasp_mode=0)
                cv2.imwrite(
                    os.path.join(opt.debug_dir, '{}_grasp_pred.png'.format(img_id)),
                    colors[0][:,:,::-1]
                )
                colors_gt, _ = s_gt.render_imgs(instance_masks=False, grasp_mode=0)
                cv2.imwrite(
                    os.path.join(opt.debug_dir, '{}_grasp_gt.png'.format(img_id)),
                    colors_gt[0][:,:,::-1]
                )

            if opt.vis_results:
                cv2.imshow("The detected keypoints", img_kpts_pred)
                cv2.imshow("The GT keypoints", img_kpts_gt)
                cv2.waitKey()

                # visualize the scene with the grasps
                # print("The GT grasps")
                # s_gt.vis_scene(mode = "trimesh")
                print("The predicted grasps")
                s.vis_scene("trimesh") 
         
        bar.next()
    bar.finish()
    print("\n The average time taken: {:4f} SPF ({:4f} FPS). ".format(total_avg_timer.avg, 1./total_avg_timer.avg))
    dataset.run_eval(results, opt.save_dir, angle_th=opt.angle_th, dist_th=opt.dist_th, \
        rot_sample=opt.rot_sample_num, trl_sample=opt.trl_sample_num)


if __name__ == '__main__':
    opt = opts().parse()
    test(opt)
