import pdb

import os, sys
import os.path as osp
src_path = os.path.dirname(os.path.dirname(__file__))
sys.path.append(src_path)
import _init_paths

import numpy as np

from mayavi import mlab
import cv2


from opts import opts
from keypoint_graspnet import KeypointGraspNet as KGN
from utils.ddd_utils import depth2pc

from physical.insp_results import load_results, get_paths 
from physical.utils_physical import quad2homog, draw_scene

class OptDemo(opts):
    def __init__(self):
        super().__init__()
        self.parser.add_argument("--demo_rgb_path", type=str, default=None)
        self.parser.add_argument("--demo_depth_path", type=str, default=None)
        self.parser.add_argument("--demo_cam_path", type=str, default=None)
        self.parser.add_argument("--demo_data_folder", type=str, default=None)
    
    def parse(self, args=''):
        opt = super().parse(args)
        # dirname = osp.dirname(opt.rgb_path)
        # basename_noExt = osp.splitext(os.path.basename(opt.rgb_path))[0]

        if opt.demo_data_folder is None:
            assert opt.demo_rgb_path is not None and opt.demo_depth_path is not None and opt.demo_cam_path is not None, \
                "Please provide either a folder containing all demo data or individual data rgb, depth, and camera info path"

        return opt


def prepare_kgn(opt):
    kgn = KGN.buildFromArgs(opt)
    return kgn


def main(opt):

    # prepare kgn
    kgn = prepare_kgn(opt)

    # get data paths
    if opt.demo_data_folder is not None:
        rgb_paths, dep_paths, cam_info_paths = get_paths(opt.demo_data_folder)
    else:
        rgb_paths, dep_paths, cam_info_paths = [opt.demo_rgb_file], [opt.demo_depth_file], [opt.demo_cam_file]

    # run
    for rgb_path ,dep_path, poses_path in zip(rgb_paths, dep_paths, cam_info_paths):
        rgb, dep, intrinsic, _, _, _ = load_results(rgb_path,dep_path, poses_path)
        kgn.set_cam_intrinsic_mat(intrinsic)

        # run KGN
        input = np.concatenate([rgb.astype(np.float32), dep[:, :, None]], axis=2)
        quaternions, locations, widths, kpts_2d_pred, scores = kgn.generate(input)
        widths += 0.05  # increment the open width, as the labeled widths are too tight.
        grasp_poses_pred_cam = quad2homog(
            locations=locations, 
            quaternions=quaternions
        )

        # filter far-away grasps that is off the ROI
        trl_norm = np.linalg.norm(grasp_poses_pred_cam[:, :3, 3], axis=1)
        filtered_idx = trl_norm > 0.9
        grasp_poses_pred_cam = grasp_poses_pred_cam[~filtered_idx, :, :]
        widths = widths[~filtered_idx]

        # get the point cloud
        pcl_cam = depth2pc(dep, intrinsic, frame="camera", flatten=True)
        invalid_pcl_idx = np.all(pcl_cam == 0, axis=1)
        pcl_cam = pcl_cam[~invalid_pcl_idx, :]
        pc_color = rgb.reshape(-1, 3)[~invalid_pcl_idx, :]

        # visualize
        draw_scene(pcl_cam, grasps=grasp_poses_pred_cam, pc_color=pc_color, widths=widths)
        print("Close the window to see the next")
        mlab.show()

    return


if __name__=="__main__":
    args = OptDemo().init()
    print(args)

    main(args)