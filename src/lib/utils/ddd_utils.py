from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import cv2


def depth2pc(depth_map, intrinsic, frame="camera", extrinsic=None, flatten=True, remove_zero_depth=False):
    """
    Recover the point cloud with the input depth map and the intrinsic matrix
    Assuming the 2d frame coordinate used by the intrinsic matrix set the upper left as (0, 0), right is x, down is y.
    This is true for the realsense cameras according to the github issue below:
    https://github.com/IntelRealSense/librealsense/issues/8221#issuecomment-765335642

    Args
        depth_map (H, W):   The depth map in meters
        intrinsic (3,3):    The camera intrinsic matrix
		frame (str):        point cloud in which frame. "camera" or "tabletop"
		extrinsic (4, 4):	The extrinsic. Requires if frame="tabletop"
        flatten (bool):     Flatten the point cloud or not.

    Returns:
        pc:                 Point cloud. (H, W, 3) if not flatten else (H*W, 3)
    """

    if frame == "tabletop":
        assert extrinsic is not None, "The extrinsic is required to obtain the tabletop frame point cloud"

    H, W = depth_map.shape[:2]
    uv_map = _get_uv(depth_map)

    # (H, W, 3). where 3 is (uz, vz, z). Make use of the numpy multiplication broadcast mechanism
    uvz_map = np.concatenate(
        (uv_map, np.ones_like(uv_map[:, :, :1])),
        axis=2
    ) * depth_map[:, :, None]

    # recover the camera-frame coordinates
    pc_Cam = np.linalg.inv(intrinsic) @ \
        uvz_map.reshape(-1, 3).T
    pc_Cam = pc_Cam.T	#(N, 3)

    if remove_zero_depth:
        pc_Cam = pc_Cam[pc_Cam[:, 2] != 0]

    if frame == "tabletop":
        pc_Cam_homog = np.concatenate(
            (pc_Cam, np.ones_like(pc_Cam[:, :1])),
            axis=1
        )
        pc_table = (np.linalg.inv(extrinsic) @ (pc_Cam_homog.T)).T 
        pc_table = pc_table[:, :3]
        pc = pc_table
    else:
        pc = pc_Cam
    
    if not flatten:
        pc = pc.reshape(H, W, 3)

    return pc


def _get_uv(img, vec=False):
    """
    Get the pixel coordinates of an input image.
    The origin (0, 0) is the upperleft corner, with right as u and down as vA

    @param[in]  img     The input image of the shape (H, W)
    @param[out] vec     Vectorize the outputs? Default is False
    @param[out] uv_map  The (u, v) coordinate of the image pixels. (H, W, 2), where 2 is (u, v)
                        If vec is True, then will be (H*W, 2)
    """
    H, W = img.shape[:2]
    rows, cols = np.indices((H, W))
    U = cols
    V = rows
    uv_map = np.concatenate(
        (U[:, :, None], V[:, :, None]),
        axis=2
    )

    # TODO: Vectorize the output instead as a map?
    if vec:
        uv_map = uv_map.reshape(-1, 2)

    return uv_map


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import os,sys
    import json
    import pcl
    import pcl.pcl_visualization as viewer

    # path - 1 is a bowl, 4 is a stick
    scene_path = "/home/cyy/Research/Grasp/3d_grasp/data/ps_grasp_single_1k/1/"
    depth_path = os.path.join(scene_path, "depth_raw/depth_raw_1.npy")
    scene_info_path = os.path.join(scene_path, "scene_info.json")

    # depth and camera info
    depth = np.load(depth_path)
    with open(scene_info_path, "r") as f:
        infos = json.load(f)
    intrinsic = np.array(infos["intrinsic"])
    camera_pose = np.array(infos["camera_poses"])[1, :, :]

    # get the point cloud and visualize
    pc = depth2pc(depth, intrinsic, frame="tabletop", extrinsic=np.linalg.inv(camera_pose), 
        flatten=True, remove_zero_depth=True)
    pc_obj = pc[pc[:, 2] > 0.002]

    # visualize the point cloud to confirm
    pc_pcl = pcl.PointCloud(pc_obj.astype(np.float32))
    vs=pcl.pcl_visualization.PCLVisualizering
    vss0=pcl.pcl_visualization.PCLVisualizering() 
    # color setting
    scene_color = pcl.pcl_visualization.PointCloudColorHandleringCustom(pc_pcl, 0, 0, 255)
    # grasp_color = pcl.pcl_visualization.PointCloudColorHandleringCustom(cloud, 0, 255, 0)
    # add the point cloud with label and the point size
    vs.AddPointCloud_ColorHandler(vss0,pc_pcl, scene_color, id=b'obj',viewport=0)
    vss0.SetPointCloudRenderingProperties(viewer.PCLVISUALIZER_POINT_SIZE, 5, b'obj')
    v = True
    while not vs.WasStopped(vss0):
        vs.Spin(vss0)    
