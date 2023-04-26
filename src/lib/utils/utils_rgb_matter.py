import pdb

import numpy as np
import open3d as o3d
import argparse
import torch
import torchvision.transforms as transforms
import time
import json
import matplotlib.pyplot as plt
import cv2
from PIL import Image


from utils.transform import create_rot_mat_axisAlign

from graspnetAPI.grasp import GraspGroup
from rgb_matter.rgbd_graspnet.net.rgb_normal_net import RGBNormalNet
from rgb_matter.rgbd_graspnet.data.utils.convert import convert_grasp_camera, get_workspace_mask
from rgb_matter.rgbd_graspnet.constant import GRASPNET_ROOT, LABEL_DIR
from rgb_matter.rgbd_graspnet.data.utils.collision import gen_cloud


def rgb_transform(rgb_array):
    resize = transforms.Resize((288, 384))
    totensor = transforms.ToTensor()
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )
    trans_list = [resize]
    trans_list += [totensor, normalize]
    rgb_transform = transforms.Compose(trans_list)

    rgb_transformed = rgb_transform(rgb_array)
    return rgb_transformed

def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")


class RGBMatter_Runner():
    """
    Args:
        depth_scale (float):                Convert unit of depth data in testing to mm. 
    """
    def __init__(self, opt, num_layers=50, use_normal=False, normal_only=False, depth_scale=1000,
                resume="checkpoints/kn_no_norm_76800.pth", cuda=True) -> None:

        self.depth_scale = depth_scale


        self.device = "cuda:0" if cuda else "cpu"
        weights_path = resume

        self.net = RGBNormalNet(
            num_layers=num_layers, use_normal=use_normal, normal_only=normal_only
        )
        state_dict = torch.load(weights_path)
        self.net.load_state_dict(state_dict["net"], strict=False)
        self.net = self.net.to(self.device)
        self.net.eval()
    
    def _convert_to_kgn_frame(self, gg:GraspGroup):
        """
        Convert pose for Graspnet1b frame definition to that of kgn frame.
        Graspnet1b defines: (see doc: https://graspnetapi.readthedocs.io/en/latest/grasp_format.html#grasp-and-graspgroup-transformation)
            0. Origin at the center, and the distance towards finger tips is param as depth
            1. +x points to reaching direction
            2. +y points to finger tip
            3. +z follows
        kgn defines:
            0. Origin at the midway between finger tips
            1. +x points to reaching direction
            2. +z points to finger tip
            3. +y follows
        """

        locations = gg.translations         # (N, 3)
        rot_mats = gg.rotation_matrices     # (N, 3, 3)
        widths = gg.widths                  # (N, )


        # first reach forward by detph so that the origin is at the midway between finger tips 
        # NOTE: RGBMatter hard code depth to be 0.04
        T_retract = np.array([0.04, 0, 0])
        locations = locations + rot_mats @ T_retract 

        # then switch y & z 
        R_align = create_rot_mat_axisAlign([1, 3, -2])
        rot_mats =  rot_mats @ R_align

        return locations, rot_mats
        


    def detect_grasp_from_ps_paths(self, rgb_image_path, depth_raw_path, seg_image_path, camera_info_path, vis=False):
        """
        Args:
            depth_raw_path:         Note the unit should correspond to the depth_scale 
            vis:                    Visualize detected grasps using open3d in camera frame
        """
        # load input
        rgb = Image.open(rgb_image_path)
        rgb_inp = rgb_transform(rgb)
        normal = torch.Tensor([])
        rgb_inp = rgb_inp.unsqueeze(0).to(self.device)
        normal = normal.unsqueeze(0).to(self.device)

        # the first time it will run very slowly.
        prob_map = self.net(rgb_inp, normal)

        tic = time.time()
        for _ in range(100):
            prob_map = self.net(rgb_inp, normal)
        toc = time.time()

        print("=" * 20)
        print("Net time:{}".format((toc - tic) / 100.0))
        print("=" * 20)

        pred_map = prob_map[0].to("cpu").clone().detach().numpy().astype(np.float32)

        rgb_image_path = rgb_image_path
        depth_raw_path = depth_raw_path 
        seg_path = seg_image_path
        f = open(camera_info_path, 'r')
        cam_info = json.load(f)
        intrinsic = cam_info["intrinsic"]
        depth_scale = self.depth_scale
        gg = convert_grasp_camera(
            label=pred_map,
            rgb_image_path=rgb_image_path,
            depth_image_path=depth_raw_path,
            seg_path=seg_path,
            intrinsic=intrinsic,
            factor_depth=depth_scale,
            top_in_grid=5,
            top_in_map=1000,
            top_sample=200,
            topK=30,
            approach_dist=0.05,#0.05,
            collision_thresh=0.001,#0.001,
            empty_thresh=0.10,#0.10,
            nms_t=0.04,
            nms_r=30,
            width_list=[0.1],
            delta_depth_list=[-0.02, 0, 0.02],
            flip=False,
            device="cuda:0",
        )

        gg.sort_by_score()
        # pdb.set_trace()

        if vis:
            color = np.array(Image.open(rgb_image_path))
            image_height, image_width = color.shape[:2]
            depth = np.load(depth_raw_path)
            depth = depth * depth_scale # convert to mm
            seg = np.array(
                Image.open(seg_path)
            )
            scene_points, colors = gen_cloud(
                    color,
                    seg,
                    depth,
                    image_width,
                    image_height,
                    intrinsic,
                    depth_scale,
                    False,
                    False,
                )
            pc = o3d.geometry.PointCloud()
            pc.points = o3d.utility.Vector3dVector(scene_points)
            pc.colors = o3d.utility.Vector3dVector(colors)

            gg_3d = gg.to_open3d_geometry_list()

            o3d.visualization.draw_geometries([pc, *gg_3d])

            plt.subplot(2, 2, 1)
            plt.title("Transformed image")
            rgbbp = rgb_inp.detach().cpu().numpy()[0]
            rgbbp = rgbbp / rgbbp.max()
            plt.imshow(np.transpose(rgbbp, (1, 2, 0)))

            plt.subplot(2, 2, 2)
            plt.title("Predicted sum of AVH")
            pred_heatmap = np.sum(pred_map, axis=0)
            pred_heatmap = pred_heatmap / pred_heatmap.max()
            plt.imshow(pred_heatmap)

            plt.show()


        # for return 
        locs, rot_mats = self._convert_to_kgn_frame(gg)

        return locs, rot_mats, gg.widths 
        
         


