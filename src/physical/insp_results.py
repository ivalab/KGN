from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pdb
import os
import sys
import argparse


# NOTE: Must install before cv2
from mayavi import mlab

import cv2
import numpy as np
import matplotlib.pyplot as plt
# from scipy.spatial.transform import Rotation as R
import trimesh




# set the path
src_path = os.path.dirname(os.path.dirname(__file__))
sys.path.append(src_path)
import _init_paths

from data_generation.grasp.grasp import Grasp
from utils.ddd_utils import depth2pc
from physical.utils_physical import GraspPoseRefineScale, draw_scene


# globle setting
root_path = None


def load_results(rgb_path, dep_path, poses_path):
    rgb = cv2.imread(rgb_path)
    rgb = rgb[:, :, ::-1]
    dep = np.load(dep_path)
    with np.load(poses_path) as poses:
        intrinsic = poses["intrinsic"]
        grasp_poses_pred_cam = poses["grasp_pose_pred_cam"]
        widths = poses["widths"]
        select_idx = poses["select_idx"] if "select_idx" in poses.keys() else None
    
    return rgb, dep, intrinsic, grasp_poses_pred_cam, widths, select_idx

def get_paths(dir_name, id_target=None):
    ls = os.listdir(dir_name)
    rgb_paths = []
    dep_paths = []
    poses_paths = []

    for file_name in sorted(ls):
        ext = file_name[file_name.rfind('.') + 1:].lower()
        if ext in ["png"]:
            # get the id
            file_name_noExt = file_name[:file_name.rfind('.')]
            id = file_name_noExt[file_name_noExt.rfind('_')+1:]

            if id_target is not None and not int(id) in [id_target]: # selective visualization
                continue

            # get the other paths
            dep_name = "depth_raw_"+id+".npy"
            poses_name = "poses_"+id+".npz"

            # add to cache
            rgb_paths.append(os.path.join(dir_name, file_name))
            dep_paths.append(os.path.join(dir_name, dep_name))
            poses_paths.append(os.path.join(dir_name, poses_name))
        
    return rgb_paths, dep_paths, poses_paths
 


def main(args):
    # The directory and get the paths
    result_dir = os.path.join(root_path, args.exp_name)
    rgb_paths, dep_paths, poses_paths = get_paths(result_dir, id_target=args.trail_id)
    
    if args.refine_poses:
        pose_refiner = GraspPoseRefineScale(intrinsic=None)

    # check results one by one
    for rgb_path ,dep_path, poses_path in zip(rgb_paths, dep_paths, poses_paths):
        rgb, dep, intrinsic, grasp_poses_pred_cam, widths, select_idx = load_results(rgb_path,dep_path, poses_path)
        widths += 0.05

        # refine poses
        if args.refine_poses:
            pose_refiner.store_info(dep=dep, intrinsic=intrinsic)
            grasp_poses_pred_cam, refine_succ = pose_refiner.refine_poses(grasp_poses_pred_cam)
            grasp_poses_pred_cam = grasp_poses_pred_cam[refine_succ, :, :]
        
        # get the point cloud
        pcl_cam = depth2pc(dep, intrinsic, frame="camera", flatten=True)
        vis_mask = ~np.all(pcl_cam==0, axis=1)
        pcl_cam = pcl_cam[vis_mask]
        pc_color = rgb.reshape(-1, 3)[vis_mask]
        # draw_scene(pcl_cam, grasps=grasp_poses_pred_cam, pc_color=rgb.reshape(-1, 3))
        select_idx = None
        draw_scene(pcl_cam, grasps=grasp_poses_pred_cam, pc_color=pc_color, select_idx=select_idx, widths=widths)
        print("Close the window to see the next")
        mlab.show()

        # visualize

    return


def get_args():
    global root_path
    opt = argparse.ArgumentParser()
    opt.add_argument("--exp_type", type=str, default="single")
    opt.add_argument("--exp_name", type=str, default="test")
    opt.add_argument("--trail_id", type=int, default=-1)
    opt.add_argument("--refine_poses", action="store_true")

    args = opt.parse_args()

    if args.trail_id == -1:
        args.trail_id = None
    return args


if __name__=="__main__":
    args = get_args()
    root_path = os.path.join(os.path.dirname(__file__), f"kgnv2_phy_{args.exp_type}")
    main(args)
