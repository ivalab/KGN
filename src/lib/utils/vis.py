import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.transform import Rotation as R

from data_generation.scene.sceneRender import SceneRender
from utils.transform import create_homog_matrix

def construct_scene_with_grasp_preds(
    obj_types,
    obj_dims,
    obj_poses,
    camera_pose,
    obj_colors=None,
    grasp_results=None,
    grasp_pred_frame="camera",
    grasp_succ=None,
    grasp_color=[0, 0, 255],
    table_size=1,
    table_thickness=0.04,
    table_color=[0.5, 0.5, 0.5, 1],
    max_grasp_num  = None
):
    """Visualize the scene with the grasp prediction from a single camera view

    Args:
        obj_types (_type_): _description_
        obj_dims (_type_): _description_
        obj_poses (_type_): _description_
        camera_pose (_type_): _description_
        grasp_results (_type_, optional): _description_. Defaults to None.
        grasp_succ (_type_, optional): _description_. Defaults to None.
        grasp_color (list, optional): _description_. Defaults to [0, 0, 255].
        table_size (int, optional): _description_. Defaults to 1.
        table_thickness (float, optional): _description_. Defaults to 0.04.
        table_color (list, optional): _description_. Defaults to [0.5, 0.5, 0.5, 1].
    
    Returns:
        s (SceneRender): The SceneRender instance
    """
    s = SceneRender.construct_scene_obj_info(
        obj_types=obj_types,
        obj_dims=obj_dims,
        obj_poses=obj_poses,
        obj_colors = obj_colors,
        camera_poses = [camera_pose],
        table_size=table_size,
        table_thickness=table_thickness,
        table_color=table_color
    )

    # grasps
    if grasp_results is None:
        s.analyze_grasps()
    else:
        # recover the predicted grasp poses in the world frame
        grasp_widths_pred = grasp_results["widths"]
        if not "poses" in grasp_results.keys():
            locations = grasp_results["locations"]
            quaternions = grasp_results["quaternions"]
            N_grasps = locations.shape[0]
            grasp_poses_pred = np.zeros((N_grasps, 4, 4), dtype=float)
            for i in range(N_grasps):
                r = R.from_quat(quaternions[i, :])
                grasp_pose_pred_this = create_homog_matrix(R_mat=r.as_matrix(), T_vec=locations[i, :])
                if grasp_pred_frame == "camera":
                    # cam_coords = (cam_pose)^-1 kpt_world_coords = (cam_pose)^-1 grasp_pose kpt_gripper_coords
                    grasp_pose_pred_this =  camera_pose @ grasp_pose_pred_this
                elif grasp_pred_frame == "tabletop":
                    pass
                else:
                    raise NotImplementedError
                grasp_poses_pred[i, :, :] = grasp_pose_pred_this
        else:
            grasp_poses_pred = grasp_results["poses"]

        if max_grasp_num is not None and grasp_poses_pred.shape[0] > max_grasp_num:
            print("Too many grasps. Sample {} of them".format(max_grasp_num))
            ids = np.random.choice(grasp_poses_pred.shape[0], max_grasp_num)
            grasp_poses_pred = grasp_poses_pred[ids, :, :]
        s.grasp_poses=[grasp_poses_pred]
        s.grasp_widths = [grasp_widths_pred]
        if grasp_succ is None:
            s.grasp_color = grasp_color
            s.grasp_collide = [np.ones_like(grasp_widths_pred)]
            s.grasp_analyzed = True
        else:
            s.grasp_color = None
            s.grasp_collide = [np.logical_not(grasp_succ)]
            s.grasp_analyzed = True
    
    return s


def vis_scene_grasps(
    obj_types,
    obj_dims,
    obj_poses,
    camera_pose,
    grasp_results=None,
    grasp_color=[0, 0, 255],
    table_size=1,
    table_thickness=0.04,
    table_color=[0.5, 0.5, 0.5, 1]
):
    """Visualize the scene with the grasp prediction from a single camera view

    Args:
        obj_types (_type_): _description_
        obj_dims (_type_): _description_
        obj_poses (_type_): _description_
        camera_pose (_type_): _description_
        grasp_results (_type_, optional): _description_. Defaults to None.
        grasp_color (list, optional): _description_. Defaults to [0, 0, 255].
        table_size (int, optional): _description_. Defaults to 1.
        table_thickness (float, optional): _description_. Defaults to 0.04.
        table_color (list, optional): _description_. Defaults to [0.5, 0.5, 0.5, 1].
    """
    s = construct_scene_with_grasp_preds(
        obj_types,
        obj_dims,
        obj_poses,
        camera_pose,
        grasp_results,
        grasp_color,
        table_size,
        table_thickness,
        table_color
    )

    colors, depths = s.render_imgs(instance_masks=False, grasp_mode=0)
    for color, depth in zip(colors, depths):
       f, axarr = plt.subplots(1, 2)
       im = axarr[0].imshow(color)
    #    f.colorbar(im, ax=axarr[0])
       im = axarr[1].imshow(depth)
    #    f.colorbar(im, ax=axarr[1])
    plt.show()
