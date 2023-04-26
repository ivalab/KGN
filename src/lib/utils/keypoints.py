import enum
import numpy as np
import torch
import cv2
from itertools import combinations

from copy import deepcopy

import os,sys
sys.path.append(
    os.path.dirname(os.path.dirname(__file__))
)
from grasp_kpts import generate_kpt_links, BoxVertexType


def kpts_3d_to_2d(intrinsic, extrinsic, coord_3d, output_details = False):
    """Map the 3d keypoints to the 2d image plane

    Args:
        intrinsic (array, (3, 3)): The camera intrinsic matrix
        extrinsic (array, (4, 4)): The camera extrinsic pose matrix
        coords_3d (array, (N, 3)): The world 3d coordinates as column vector 
        output_details (bool):    Output the details, including:
                                1. camera-frame 3d coordinates
                                2. scales
    Returns:
        coord_2d (array, (N,2))
        p_c (array, (N, 3))     The camera frame 3d coordinates. Only output if output_details is True
        scales (array, (N, ))   The scales. Only output when output_scale is True. Only output if output_details is True
    """
    # homogeneous world frame coordinates
    coord_3d = np.array(coord_3d)
    N = coord_3d.shape[0]
    p_w_homog = np.concatenate(
        (np.array(coord_3d), np.ones((N, 1))), 
        axis=1
    )   #(N, 4)

    # the extrinsic matrix
    if extrinsic.shape == (4, 4):
        extrinsic = extrinsic[:3, :]
    elif extrinsic.shape == (3, 4):
        extrinsic = extrinsic
    else:
        raise ValueError("The extrinsic matrix must be either 4-by-4 or 3-by-4. Got {}".format(extrinsic.shape))
    
    # camera coordinate
    p_c = extrinsic @ p_w_homog.T   #(3, N)

    # image coordinate
    p_img = (intrinsic @ p_c).T     #(N, 3)
    coord_2d = p_img[:, :2] / p_img[:, 2:]  

    if not output_details:
        return coord_2d
    else:
        return coord_2d, p_c.T, p_img[:, 2:].squeeze()

def plot_single_grasp_kpts(img_rgb, grasp_kpts_2d, kpt_mode="hedron", color_rgb=[0, 0, 255], shape="square", size=8):
    """Plot the kpts for a single grasp

    Args:
        img_rgb (array): The rgb image
        kpt_mode (str): "hedron" or "box" or "tail"
        grasp_kpts_2d (array (n, 2)): The grasp keypoints' image coordinate as the column vector. \
            If "hedron", n=4. The order should be left (+z), right (-z), top (+y), bottom (-y).
        rgb (array, [3, ]): The rgb color
        shape (str): The shape of the keypoints. "circle" or "square"
        size (int): The size of the keypoints. circle diameter or the square side dimension
    
    Return:
        img_plot [np.ndarray]:  the rgb image plotted with the keypoints
    """
    img_plot = deepcopy(img_rgb[:, :, ::-1])
    color_bgr = np.array(color_rgb)[::-1].astype(int)   # opencv bgr
    color = [int(color_bgr[0]), int(color_bgr[1]), int(color_bgr[2])]

    # plot the links first to avoid overlapping the keypoints, which is more important
    links = generate_kpt_links(kpts_type=kpt_mode)
    for idx1, idx2 in links:
        img_plot = cv2.line(
            img_plot, 
            (int(grasp_kpts_2d[idx1, 0]), int(grasp_kpts_2d[idx1, 1])), 
            (int(grasp_kpts_2d[idx2, 0]), int(grasp_kpts_2d[idx2, 1])), 
            color=color, 
            thickness=1
        )  

    # plot the keypoints
    for i in range(grasp_kpts_2d.shape[0]):

        # plot the finger tips (left, right) as dark and bright blue, the other two as dark and bright red
        # Use the Box type index is okay, since all set the left and right as the first two.
        # bgr
        if i == BoxVertexType.Left:
            color_kpt = [125, 0, 0]   
        elif i == BoxVertexType.Right:
            color_kpt = [255, 0, 0]    
        elif i == BoxVertexType.TopLeft:
            color_kpt = [0, 0, 125]     
        elif i == BoxVertexType.TopRight:
            color_kpt = [0, 0, 255]     
        else:
            raise NotImplementedError

        if shape == "circle":
            img_plot = cv2.circle(img_plot, (int(grasp_kpts_2d[i, 0]), int(grasp_kpts_2d[i, 1])), radius=int(size/2), color=color_kpt, thickness=-1)
        elif shape == "square":
            img_plot = cv2.rectangle(
                img_plot,
                (int(grasp_kpts_2d[i, 0] - size/2), int(grasp_kpts_2d[i, 1] - size/2)),
                (int(grasp_kpts_2d[i, 0] + size/2), int(grasp_kpts_2d[i, 1] + size/2)),
                color=color_kpt,
                thickness=-1
            )
    

        
    return img_plot[:,:,::-1]



def plot_grasps_kpts(img, grasps_kpts_2d, kpts_mode="hedron", succ_flags=None, succ_rgb=[0, 255, 0], succ_shape="circle", \
    fail_rgb=[255, 0, 0], fail_shape="circle", size=8):
    """Plot the 2d keypoints for all the grasps

    Args:
        img (array): The rgb image
        kpt_mode (str): "hedron" or "box" or "tail"
        grasps_kpts_2d (array, (N, n, 2)): The grasp keypoints image coordinate. N is the grasp number. n is the keypoint number for each grasp.  \
            The coordinate should be in the OpenCV style.
            If "hedron", n=4. The order should be left (+z), right (-z), top (+y), bottom (-y).
        succ_flags (list of bool, (N, )): The grasp is success or not. Will distinguish them using the color and shape. \
            Defaults to None, which take all the grasps as success
        size (int): The size of the keypoints in pixel. circle diameter or the square side dimension. Defaults to 8

    Return:
        img_plot [np.ndarray]:  the image plotted with the keypoints
    """
    N = len(grasps_kpts_2d)

    # default settings
    if succ_flags is None:
        succ_flags = [True] * N

    # iterate through all grasps
    for i in range(N):
        grasp_kpts = grasps_kpts_2d[i]
        # determine the color and shapes
        if succ_flags[i]:
            rgb = succ_rgb
            shape = succ_shape
        else:
            rgb = fail_rgb
            shape = fail_shape

        # plot
        img = plot_single_grasp_kpts(img, grasp_kpts, kpt_mode=kpts_mode, color_rgb=rgb, shape=shape, size=size) 

    return img


def get_vanishing_points(corners, return_corners_ordered=False, tol=1e-3, mode="np"):
    """Calculate the vanishing points from the corners of the box-type keypoint.

    Args:
        corners (N, 8):             The corner coordinates. N is the grasp number, 8=2*4 is the coordinates

        return_corners_ordered:     Option to return the ordered corners from which the vpts are calculated.
                                    The ordered corners are the rearrangement of the corners s.t. the first two are on the one line
                                    while the next two are on the other.

        tol (float):                The tolerance for determine the infinity of the vpts with the denominator.
                                    Defaults to 1e-4
        
        mode (str):                 Numpy or torch (for pytorch)

    Returns:  
        vpts (N, 4):                The vanishing points coordinates. 4=2*2 is the coords for two vanishing points.
                                    The vanishing point at the infinity will have the coordinate (0, 0)

        fin_mask (N, 4):            The mask indicating the vanishing points at the finity.
                                    The 2 coordinates corresponding to the vpts at the infinity will be set to False,
                                    else True

        corners_ordered (N, 16)     The ordered corners. Only returned if return_corners_ordered is True
                                    The first 8 are the coordinates of the 4 kpts contributes to the first vpt (i.e. the first two values of the vpts),
                                    while the next 8 are for the second.
    """

    N, _ = corners.shape

    if mode == "np":
        corners = corners.astype(np.float32)
        vpts = np.zeros((N, 4), dtype=np.float32)
        fin_mask = np.ones((N, 4), dtype=bool)
        corners_ordered = np.empty((N, 16), dtype=np.float32) 
        abs_func = np.abs
    elif mode == "torch":
        corners = corners.float()
        # requires_grad will be overwritten by corners'
        vpts = torch.zeros((N, 4), device=corners.device)
        fin_mask = torch.ones((N, 4), dtype=bool, device=corners.device)
        corners_ordered = torch.empty((N, 16), device=corners.device) 
        abs_func = torch.abs
    else:
        raise NotImplementedError

    inds = [
        [BoxVertexType.Left, BoxVertexType.TopLeft, BoxVertexType.Right, BoxVertexType.TopRight],
        [BoxVertexType.Left, BoxVertexType.Right, BoxVertexType.TopLeft, BoxVertexType.TopRight]
    ]


    for vpt_ind, (ind1, ind2, ind3, ind4) in enumerate(inds):
        # the corner xys. All with the shape (N, )
        x1s = corners[:, 2*ind1]
        y1s = corners[:, 2*ind1+1]
        x2s = corners[:, 2*ind2]
        y2s = corners[:, 2*ind2+1]
        x3s = corners[:, 2*ind3]
        y3s = corners[:, 2*ind3+1]
        x4s = corners[:, 2*ind4]
        y4s = corners[:, 2*ind4+1]

        if mode == "np":
            corners_ordered[:, vpt_ind*8: (vpt_ind+1)*8] = np.stack(
                (x1s, y1s, x2s, y2s, x3s, y3s, x4s, y4s),
                axis=1
            )
        elif mode == "torch":
            corners_ordered[:, vpt_ind*8: (vpt_ind+1)*8] = torch.stack(
                (x1s, y1s, x2s, y2s, x3s, y3s, x4s, y4s),
                dim=1
            )

        # denominator (N, )
        denom = \
            - x1s * y3s + x2s * y3s + x1s * y4s  - x2s * y4s \
            - x3s * y2s + x4s * y2s + x3s * y1s - x4s * y1s
        

        # Finite mask - If the deminator is not zero
        fin_inds = (abs_func(denom) > tol)
        fin_mask[:, 2*vpt_ind] = fin_inds
        fin_mask[:, 2*vpt_ind + 1] = fin_inds

        # vpt coordinates
        vpts_x_num = \
            -x1s * x3s * y2s + x1s * x4s * y2s - x1s * x4s * y3s  + x1s * x3s * y4s  \
                - x2s * x3s * y4s + x2s * x4s * y3s + x2s * x3s * y1s - x2s * x4s * y1s
        vpts_y_num = \
            x2s * y1s * y3s - x2s * y1s * y4s - x4s * y1s * y3s + x3s * y1s * y4s \
                - x1s * y2s * y3s + x1s * y2s * y4s - x3s * y2s * y4s + x4s * y2s * y3s
    
        # fill in the vpts
        vpts[fin_inds, 2 * vpt_ind] = vpts_x_num[fin_inds] / denom[fin_inds]
        vpts[fin_inds, 2 * vpt_ind + 1] = vpts_y_num[fin_inds] / denom[fin_inds]

    if return_corners_ordered: 
        return vpts, fin_mask, corners_ordered
    else:
        return vpts, fin_mask 
    
def get_ori_cls(kpts, range_mode=0, total_cls_num=20):
    """Given the 2d keypoint coordinates, get the orientation class
    The orientation class is defined as the left-right vector direction.

    Args:
        kpts (N_grasps, N_kpts, 2) or (N_kpts, 2):       The keypoint coordinates
        range_mode (int):       The mode of the definition of the orientation angles
                                0: -90 to 90
                                1: -180 to 180
        total_cls_num (int):    The total class number

    Returns:
        ori_cls (N_grasps, ):      The class number
    """
    if len(kpts.shape) == 2:
        kpts = kpts[None, :, :]
    left = kpts[:, 0, :]
    right = kpts[:, 1, :]

    # The orientation
    delta_y = right[:, 1] - left[:, 1]
    delta_x = right[:, 0] - left[:, 0]
    ori = np.arctan2(delta_y, delta_x)    # The numpy range: (-180, 180)

    # The orientation range
    if range_mode == 0:
        ori_range = (-np.pi/2, np.pi/2)
    elif range_mode == 1:
        ori_range = (-np.pi, np.pi)
    
    # the orientation class
    interval = (ori_range[1] - ori_range[0]) / total_cls_num
    ori_cls = np.floor((ori - ori_range[0]) / interval)
    ori_cls[ori_cls >= total_cls_num] = total_cls_num - 1
    return ori_cls.astype(np.int)

def ori_cls_2_angle(ori_cls, range_mode=0, total_cls_num=20):
    """Given the orientation class, get the mean orientation angle, theta, and the absolute half interval, delta, in radians.
    The angles belong to this class should fall in the interval [theta - delta, theta + delta]
    The orientation class is defined as the left-right vector direction.

    Args:
        ori_cls (N, ):       The keypoint coordinates
        range_mode (int):       The mode of the definition of the orientation angles
                                0: -90 to 90
                                1: -180 to 180
        total_cls_num (int):    The total class number

    Returns:
        theta (float):      The mean orientation angle theta
        delta (float):      The absolute half interval delta 
    """
    # The orientation range
    if range_mode == 0:
        ori_range = (-np.pi/2, np.pi/2)
    elif range_mode == 1:
        ori_range = (-np.pi, np.pi)

    # the interval
    interval = (ori_range[1] - ori_range[0]) / total_cls_num
    delta = interval / 2 
    
    # the orientation class
    theta = ori_range[0] + ori_cls * interval + delta
    return theta, delta


if __name__ == "__main__":

    # test the orientation classification
    kpts = np.array([
        [
        [0, 0],
        [0, 10]
        ],
        [
        [0, 0],
        [10, -10]
        ]
    ])
    ori_cls = get_ori_cls(kpts=kpts, range_mode=0, total_cls_num=18)
    theta, delta = ori_cls_2_angle(ori_cls=ori_cls, range_mode=0, total_cls_num=18)
    print(ori_cls)
    print("angle range: [{}, {}]".format((theta - delta)*180/np.pi, (theta + delta)*180/np.pi))

    # test the vanishing points
    corners = torch.tensor(
        [
            [0, -1, 2, 0, 0, 1, 1, 2],
            [0, -1, 1, 0, 0, 1, 1, 2]
        ],
        dtype=float,
        requires_grad=True
    )

    # # test the vanishing points
    # corners = torch.tensor(
    #     [
    #         [85.9935, 77.0073, 75.0053, 71.0068, 94.9980, 68.9977, 83.9993, 65.0041]
    #     ],
    #     dtype=float,
    #     requires_grad=True
    # )

    
    # expect: (0, 4, -4, -3) for the first one, and all False for the second one
    vpts, mask, corners_ordered = get_vanishing_points(corners, return_corners_ordered=True, mode="torch")
    print("Torch test:")
    print(vpts)
    print(mask)
    print(vpts.requires_grad)
    print(mask.requires_grad)
    print(corners_ordered.requires_grad)


    # test the vanishing points with np
    corners = np.array(
        [
            [0, -1, 2, 0, 0, 1, 1, 2],
            [0, -1, 1, 0, 0, 1, 1, 2]
        ],
        dtype=float,
    )
    
    # expect: (0, 4, -4, -3) for the first one, and all False for the second one
    vpts, mask, corners_ordered = get_vanishing_points(corners, return_corners_ordered=True)
    print("\n Numpy test:")
    print(vpts)
    print(mask)
    