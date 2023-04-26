import os, sys
import numpy as np
from numpy import pi

import trimesh
import matplotlib.pyplot as plt
import cv2

path = os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__)
    )))
sys.path.append(
    path
)
import _init_paths

from utils.transform import create_homog_matrix , create_rot_mat_axisAlign
from scipy.spatial.transform import Rotation as R

from pose_recover.pnp_solver_factory import PnPSolverFactory

from pose_recover.unit_test import perturb_coords, get_grasp_2Dprojs


def get_camera_intrinsic():
    intrinsic = np.array(
        [[616.36529541, 0, 310.25881958], 
        [0, 616.20294189, 236.59980774], 
        [0, 0, 1]]
    )
    return intrinsic

def create_camera_pose(scale):
    # create the pose
    pose = create_homog_matrix(T_vec=[0, scale, 0], R_mat=create_rot_mat_axisAlign([-1, 3, 2])) # the R_mat so that the camera is still pointing to the table center
    trf = create_rot_mat_axisAlign([1, -2, -3])
    trf = create_homog_matrix(R_mat=trf)
    pose = pose @ trf
    return pose

def sample_grasp_poses(num):
    return trimesh.transformations.random_rotation_matrix(num=num)

def generate_kpts(intrinsic, cam_pose, grasp_poses, width):
    """
    Return: kpt (N, 4, 2)
    """
    return get_grasp_2Dprojs(grasp_poses, grasp_widths=[width]*grasp_poses.shape[0], intrinsic=intrinsic, camera_pose=cam_pose, kpt_type="box")

def plot(errors, scales, noise_level, ylabel="Error"):

    save_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    figsize = (8, 6)
    # figsize = (10, 6)
    axis_label_prop = {
        # "fontweight": "bold",
        "fontsize": 22,
    }
    ticks_prop = {
        "fontsize": 18
    }
  


    # configs
    line_prop = {
        "linewidth": 2,
        "markersize": 10
    }
    legend_prop = {
        "ncol": 1,
        "loc": "upper left",
        "fontsize": 18,
        "frameon": True,
    }


    # plot
    fig, ax = plt.subplots(figsize = figsize)
    xlabel = "2d Keypoint Noise - Standard Deviation"
    for idx, scale in enumerate(scales):
        errs = errors[idx, :]
        plt.plot(noise_level, errs, "-o", label=f"Scale={scale}m", **line_prop)

    # reverse the order, so that energy decrease towards right
    # ax.invert_xaxis() 

    # axes, labels, legends, etc.
    plt.yticks(None, **ticks_prop)
    plt.ylabel(ylabel, **axis_label_prop)
    # plt.ylim(50, 100)
    plt.xticks(noise_level, **ticks_prop)
    plt.xlabel(xlabel, **axis_label_prop)
    handles, labels = ax.get_legend_handles_labels()
    plt.legend(handles[::-1], labels[::-1], **legend_prop)
    # plt.show()


    # save
    # save_path = os.path.join(save_dir, "anlsEngyNoise.pdf")
    # fig.savefig(save_path, bbox_inches='tight', format="pdf")


def cal_pose_error(poses_pred, poses_gt):
    def _rotate_poses_180_by_x(poses):
        poses_new = np.copy(poses)
        # correct the pose correspondingly. Rotate along the x axis by 180 degrees
        M_rot = create_homog_matrix(
            R_mat=create_rot_mat_axisAlign([1, -2, -3]),
            T_vec=np.zeros((3, )) 
        )
        poses_new = poses_new @ M_rot 
        return poses_new

    
    def _get_SO3_dist(rotMat1, rotMat2):
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

    pred_rotations = poses_pred[:, :3, :3]  # (N_pred, 3, 3)
    pred_translates = poses_pred[:, :3, 3]  # (N_pred, 3)
    gt_rotations_1 = poses_gt[:, :3, :3]  # (N_gt, 3, 3)
    gt_translates = poses_gt[:, :3, 3]  # (N_gt, 3)
    # poses_gt_2 = _rotate_poses_180_by_x(poses_gt)
    # gt_rotations_2 = poses_gt_2[:, :3, :3]  # (N_gt, 3, 3)

    # the numbers
    pred_num = pred_rotations.shape[0]
    gt_num = gt_translates.shape[0]

    # SO(3) distances - minimum rotation angle
    angle_dist_1 = _get_SO3_dist(pred_rotations, gt_rotations_1)
    angle_dist = np.diag(angle_dist_1)

    # Translation distance
    translates_diff = pred_translates[:, np.newaxis,
                                      :] - gt_translates[np.newaxis, :, :]
    translates_dist = np.linalg.norm(
        translates_diff, axis=2)   # (N_pred, N_gt)
    trl_dist = np.diag(translates_dist)

    angle_dist_avg = np.average(angle_dist)
    trl_dist_avg = np.average(trl_dist)

    return angle_dist_avg, trl_dist_avg

def main(scale_list, noise_levels, grasp_num=100):
    grasp_poses = sample_grasp_poses(grasp_num)
    intrinsic = get_camera_intrinsic()
    width = 0.1
    # create the solver:
    solver = PnPSolverFactory["cvIPPE"](
        kpt_type = "box",
        camera_intrinsic_matrix = intrinsic,
    )
    solver.set_open_width(open_width=width)

    # result cache
    N_scale = len(scale_list)
    N_noise = len(noise_levels)
    rot_errs, trl_errs = np.zeros((N_scale, N_noise)), np.zeros((N_scale, N_noise))

    # begin
    for idx_s, s in enumerate(scale_list):
        cam_pose = create_camera_pose(s)

        # keypoints 
        kpts = generate_kpts(intrinsic, cam_pose, grasp_poses, width)

        # add noise
        for idx_n, noise in enumerate(noise_levels):
            kpts_noise = perturb_coords(kpts, "Gaussian", noise)

            # recover with pnp
            grasp_poses_pred = np.zeros_like(grasp_poses)
            valid_preds = np.ones((grasp_poses_pred.shape[0]), dtype=bool)
            for i in range(grasp_num):
                try:
                    location, quaternion, projected_points, reprojectionError = \
                        solver.solve_pnp(
                            kpts_noise[i, :, :]
                        )
                    grasp_pose_pred_cam = create_homog_matrix(
                        R_mat=R.from_quat(quaternion).as_matrix(),
                        T_vec=location
                    )
                    grasp_poses_pred[i, :, :] = cam_pose @ grasp_pose_pred_cam

                except:
                    valid_preds[i] = False

            # compute average error - rotation and translation
            rot_err_avg, trl_err_avg = cal_pose_error(grasp_poses_pred[valid_preds], grasp_poses[valid_preds])
            rot_errs[idx_s, idx_n] = rot_err_avg
            trl_errs[idx_s, idx_n] = trl_err_avg 
            print(f"[Scale = {s}; Noise = {noise}] Rot_error_avg={rot_err_avg};  Trl_error_avg = {trl_err_avg}")
    
    # plot
    plot(rot_errs, scales=scale_list, noise_level=noise_levels, ylabel="Translation Error (m)")
    plot(trl_errs, scales=scale_list, noise_level=noise_levels, ylabel="Rotational Error (rad)")
    plt.show()
    
if __name__ == "__main__":
    scale_list = [0.4, 0.6, 0.8, 1.0]
    noise_level = [0.5, 1, 3, 5]
    grasp_num = 100
    main(scale_list, noise_level, grasp_num)

