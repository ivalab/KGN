import pdb
import os
import numpy as np
import sys
from scipy.spatial.transform import Rotation as R
from scipy.signal import medfilt2d

import trimesh

# set the import path
src_path = os.path.dirname(os.path.dirname(__file__))
sys.path.append(src_path)
import _init_paths
from data_generation.grasp.grasp import Grasp
from utils.transform import create_homog_matrix
from utils.keypoints import plot_grasps_kpts


def quad2homog(locations, quaternions):
    locations = locations.reshape((-1, 3))
    quaternions = quaternions.reshape((-1, 4))
    N_grasps = locations.shape[0]
    poses = np.zeros((N_grasps, 4, 4), dtype=float)
    for i in range(N_grasps):
        r = R.from_quat(quaternions[i, :])
        pose = create_homog_matrix(R_mat=r.as_matrix(), T_vec=locations[i, :])
        poses[i, :, :] = pose 
    return poses


def pose_cam_2_robot(pose_cam, M_CL, M_BL):
    """
    Args:
        pose_cam: (N, 4, 4) or (4,4).           The transform from the camera frame to a target frame.
        M_CL: (4, 4)                            
        M_BL: (4, 4)
    Returns:
        pose_robot: (N, 4, 4) or (4,4). i.e.    The frame transform from robot to grasp
    """
    if len(pose_cam.shape) == 2:
        pose_robot = M_BL @ np.linalg.inv(M_CL) @ pose_cam
    elif len(pose_cam.shape) == 3:
        pose_robot = M_BL[None, :, :] @ np.linalg.inv(M_CL)[None, :, :] @ pose_cam
    return pose_robot



def draw_kpts(rgb, kpts_2d_pred, opt, sample_num = -1):
    N_grasps = kpts_2d_pred.shape[0]
    if N_grasps == 0:
        return rgb 
    if sample_num > 0:
        sample_num = min(sample_num, N_grasps)
        ids = np.random.choice(N_grasps, sample_num, replace=False)
        kpts_draw = kpts_2d_pred[ids, :, :]
    else:
        kpts_draw = kpts_2d_pred
    img_kpts_pred = plot_grasps_kpts(rgb, kpts_draw, kpts_mode=opt.kpt_type, size=5)

    return img_kpts_pred

def draw_scene(pc,
               grasps=[],
               widths = None,
               grasp_scores=None,
               grasp_color=None,
               gripper_color=(0, 1, 0),
               grasps_selection=None,
               select_idx = None,
               visualize_diverse_grasps=False,
               pc_color=None,
               plasma_coloring=False):
    """
    Draws the 3D scene for the object and the scene.
    Args:
      pc: point cloud of the object
      grasps: list of 4x4 numpy array indicating the transformation of the grasps.
        grasp_scores: grasps will be colored based on the scores. If left 
        empty, grasps are visualized in green.
      grasp_color: if it is a tuple, sets the color for all the grasps. If list
        is provided it is the list of tuple(r,g,b) for each grasp.
      mesh: If not None, shows the mesh of the object. Type should be trimesh 
         mesh.
      show_gripper_mesh: If True, shows the gripper mesh for each grasp. 
      grasp_selection: if provided, filters the grasps based on the value of 
        each selection. 1 means select ith grasp. 0 means exclude the grasp.
      select_idx:   The index of the grasp selected for the execution
      visualize_diverse_grasps: sorts the grasps based on score. Selects the 
        top score grasp to visualize and then choose grasps that are not within
        min_seperation_distance distance of any of the previously selected
        grasps. Only set it to True to declutter the grasps for better
        visualization.
      pc_color: if provided, should be a n x 3 numpy array for color of each 
        point in the point cloud pc. Each number should be between 0 and 1.
      plasma_coloring: If True, sets the plasma colormap for visualizting the 
        pc.
    """

    from mayavi import mlab

    max_grasps = 40
    grasps = np.array(grasps)

    if grasp_scores is not None:
        grasp_scores = np.array(grasp_scores)

    if len(grasps) > max_grasps:

        print('Downsampling grasps, there are too many')
        chosen_ones = np.random.randint(low=0,
                                        high=len(grasps),
                                        size=max_grasps)
        # add the selected one if not sampled
        if select_idx is not None and select_idx not in chosen_ones:
            chosen_ones = np.append(chosen_ones, select_idx)
        grasps = grasps[chosen_ones]
        if grasp_scores is not None:
            grasp_scores = grasp_scores[chosen_ones]
        if widths is not None:
            widths = widths[chosen_ones]
    else:
        chosen_ones = np.arange(len(grasps))

    if pc_color is None and pc is not None:
        if plasma_coloring:
            mlab.points3d(pc[:, 0],
                          pc[:, 1],
                          pc[:, 2],
                          pc[:, 2],
                          colormap='plasma')
        else:
            mlab.points3d(pc[:, 0],
                          pc[:, 1],
                          pc[:, 2],
                          color=(0.1, 0.1, 1),
                          scale_factor=0.01)
    elif pc is not None:
        if plasma_coloring:
            mlab.points3d(pc[:, 0],
                          pc[:, 1],
                          pc[:, 2],
                          pc_color[:, 0],
                          colormap='plasma')
        else:
            rgba = np.zeros((pc.shape[0], 4), dtype=np.uint8)
            rgba[:, :3] = np.asarray(pc_color)
            rgba[:, 3] = 255
            src = mlab.pipeline.scalar_scatter(pc[:, 0], pc[:, 1], pc[:, 2])
            src.add_attribute(rgba, 'colors')
            src.data.point_data.set_active_scalars('colors')
            g = mlab.pipeline.glyph(src)
            g.glyph.scale_mode = "data_scaling_off"
            g.glyph.glyph.scale_factor = 0.01

    if grasp_scores is not None:
        indexes = np.argsort(-np.asarray(grasp_scores))
    else:
        indexes = range(len(grasps))

    print('draw scene ', len(grasps))

    selected_grasps_so_far = []
    removed = 0

    if grasp_scores is not None:
        min_score = np.min(grasp_scores)
        max_score = np.max(grasp_scores)
        top5 = np.array(grasp_scores).argsort()[-5:][::-1]

    for ii in range(len(grasps)):
        i = indexes[ii]
        if grasps_selection is not None:
            if grasps_selection[i] == False:
                continue
        if widths is not None:
            w = widths[ii]
        else:
            w = 0.1

        if select_idx is None:
            gripper_color = (0.0, 1.0, 0.0)
            tube_radius = 0.0015
        else:
            if chosen_ones[ii] == select_idx:
                gripper_color = (0.0, 1.0, 0.0)
                tube_radius = 0.003
            else:
                gripper_color = (0.8, 0.0, 0.8) # magenta
                tube_radius = 0.0015
        
        g = grasps[i]
        gripper = Grasp(open_width=w, pose=g, gripper_tube_radius=tube_radius)
        gripper_meshes = gripper.get_mesh()
        gripper_mesh = trimesh.util.concatenate(gripper_meshes)
        # gripper_mesh = sample.Object(
            # 'gripper_models/panda_gripper.obj').mesh
        # gripper_mesh.apply_transform(g)
        mlab.triangular_mesh(
            gripper_mesh.vertices[:, 0],
            gripper_mesh.vertices[:, 1],
            gripper_mesh.vertices[:, 2],
            gripper_mesh.faces,
            color=gripper_color,
            opacity=1 if visualize_diverse_grasps else 0.5)

    print('removed {} similar grasps'.format(removed))




class GraspPoseRefineBase():
    def __init__(self) -> None:
        pass

    def store_info(self):
        """Different refinement approach might requires different materials (e.g. depth map)
        Store them here
        """
        raise NotImplementedError()

    def refine_poses(grasp_poses):
        raise NotImplementedError


class GraspPoseRefineScale(GraspPoseRefineBase):
    """
    Refine the grasp by scale the translation along the camera

    Args:
        intrinsic (3, 3):       The camera intrinsic matrix
    """
    def __init__(self, intrinsic) -> None:
        super().__init__()
        self.intrinsic = intrinsic

        # cache for the required materials
        self.dep = None
    
    def store_info(self, dep, intrinsic=None):
        self.dep = dep
        if intrinsic is not None:
            self.intrinsic = intrinsic

    def refine_poses(self, grasp_poses, verbose=False):
        """
        Refine the poses by scale the translation s.t. 
        the center between the tips aligns with the observed object align that direction 
        Args:
            grasp_poses (N, 4, 4):          The frame transformation from the camera to gripper. 
                                            Assume the gripper frame origin is the center between the tips
        """
        N = grasp_poses.shape[0]
        # preprocess
        dep = self.preprocess_dep(self.dep)

        refine_successed = np.zeros((N,), dtype=bool)
        grasp_poses_refined = np.zeros_like(grasp_poses)
        for i in range(N):
            pose = grasp_poses[i, :, :]
            trl = pose[:3, 3].reshape(-1)
            rot_mat = pose[:3, :3]

            # get the image coordinate
            img_coord = (self.intrinsic @ trl.reshape((-1, 1))).reshape(-1)
            img_coord = (img_coord / img_coord[-1])[:2]

            # get the depth - OpenCV coordinate
            depth_value = dep[int(img_coord[1]), int(img_coord[0])]

            # scale 
            if(np.linalg.norm(trl)==0):
                if verbose:
                    print("The pose has problem!")
            elif (depth_value == 0):
                if verbose:
                    print("Depth has problem")
            else:
                if verbose:
                    print("Refine success!")
                refine_successed[i] = True
            trl_refined = trl * depth_value / np.linalg.norm(trl)

            # assemble to get the new pose
            grasp_poses_refined[i, :3, 3] = trl_refined
            grasp_poses_refined[i, :3, :3] = rot_mat
            grasp_poses_refined[i, 3, 3] = 1.

        return grasp_poses_refined, refine_successed

    
    def preprocess_dep(self, dep):
        # median filter
        scale = 0.01
        dep_scaled = (dep / scale).astype(np.int32)
        dep_scaled = medfilt2d(dep_scaled, kernel_size=5)
        dep = dep_scaled.astype(np.float32) * scale

        return dep
      
