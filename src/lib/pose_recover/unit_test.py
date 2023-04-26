from matplotlib import scale
import numpy as np
import sys, os 
import trimesh.transformations as tra
import matplotlib.pyplot as plt

path = os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__)
    )))
sys.path.append(
    path
)
import _init_paths

from data_generation.scene.sceneRender import SceneRender
from utils.transform import create_homog_matrix , create_rot_mat_axisAlign
from data_generation.scene.Scene import Scene
from data_generation import Sphere, Cuboid, Cylinder, Grasp
from utils.keypoints import kpts_3d_to_2d, plot_grasps_kpts
from main_data_generate import ObjSampler
from scipy.spatial.transform import Rotation as R

from pose_recover.cvPnPs import CVEPnPSolver
from pose_recover.pnp_solver_factory import PnPSolverFactory


#### Create the scene

def generate_scene(seed=None, obj_names = ["cuboid"]):
    if seed is not None:
        # fix the random seed
        np.random.seed(seed)
    obj_sampler = ObjSampler(
        cuboid_size_range= [0.02, 0.12],
        sphere_radius_range= [0.02, 0.05],
        bowl_radius_range= [0.02, 0.05],
        cylinder_rin_range=[0.03, 0.07],
        cylinder_h_range=[0.05, 0.1],
        #ring_rin_range=[0.02, 0.05],
        ring_rin_range=[0.02, 0.03],
        ring_h_range=[0.008, 0.024],
        stick_rin_range=[0.008, 0.015],
        stick_h_range=[0.04, 0.01]
    )
    scene_renderer = SceneRender(
        camera_num=1,
        table_color=[0.5, 0.5, 0.5, 0.3]  # Ignore collision for now
    )

    while(True):
        scene_renderer.clear_objs()
        scene_renderer.resample_camera_poses()
        obj_list = []
        sample_pose = []
        resample_loc = []
        for obj_name in obj_names:
            obj = obj_sampler.sample_obj(obj_name)
            obj_list.append(obj)
            # If sphere or semi-sphere, just sample the location instead of the orientation
            if obj_name == "sphere" or obj_name == "bowl":
                sample_pose.append(False)
                resample_loc.append(True)
            else:
                sample_pose.append(True)
                resample_loc.append(False)
        try:
            scene_renderer.add_objs(obj_list, sample_pose=sample_pose, resample_xy_loc=resample_loc)
        except RuntimeError:
            print("Failed to add the object. Retrying...")
            continue
        break

    return scene_renderer

def get_grasp_2Dprojs(grasp_poses, grasp_widths,intrinsic, camera_pose, kpt_type):
    """Get the grasp kpt projections
    Returns:
        kpts_2d: (N_grasp, N_kpts, 2) 
    """
    kpts_2d = []
    for i in range(grasp_poses.shape[0]):
        grasp_pose = grasp_poses[i, :, :]
        grasp_width = grasp_widths[i]   

        grasp = Grasp(grasp_width, pose=grasp_pose, kpts_option=kpt_type, kpts_dist=grasp_width)        # here we use the grasp width as the kpts dist, which is not the case in the final version
        kpts_coord = grasp.get_kpts(frame="world")  # the keypoint coordinate in the gripper frame, (N_kpt, 3)
        kpts_this = kpts_3d_to_2d(intrinsic, np.linalg.inv(camera_pose), kpts_coord)
        kpts_2d.append(kpts_this)
    return np.array(kpts_2d)

def perturb_coords(coords, perturb_mode=None, *args) :
    """Perturb the 2d coordinates
    Args:
        coords_2d (N, D)
        perturb_mode (str, optional).       The perturbation method.
                                            Options: None (no degenracy); round_off (round to the nearest integer); Gaussian (iid gaussian noise)
        *args:
            stddev (float):                    The standard deviation for the gaussian option
    """
    coords = np.array(coords)
    if perturb_mode is None:
        coords_p = coords
    elif perturb_mode == "round_off":
        coords_p = np.round(coords)
    elif perturb_mode == "Gaussian":
        stddev = args[0]   # requires a variance
        shape = coords.shape
        coords_p = coords + np.random.normal(0, stddev, size=shape)
    else:
        raise NotImplementedError
    
    return coords_p

if __name__ == "__main__":
    from copy import deepcopy

    #### settings
    kpt_type = "box"                # hedron, box, tail
    solver_name = "cvIPPE"          # cvEPnP, cvP3P, cvIPPE, planar
    #obj_names = ["cuboid", "bowl", "sphere", "cylinder", "ring", "stick"]
    obj_names = ["cuboid"]
    check_one_by_one = False        # whether visualize the predicted grasp one-by-one in the scene 
    perturb_mode =  None                # The perturbation method for the oracle 2d projections.
                                    # Options: None (no degenracy); round_off (round to the nearest integer); Gaussian
    stddev = 3                     # The standard deviation for the gaussian noise - 1, 2, 3, 5
    # thresholds
    reproj_error_th = None           # the reprojection error filter. If None, will not filter
    open_width_cano = None               # The canonical open width
    open_width_th = None            # The open width filter threshold. If the oracle open width is below this threshold, will set to this threshold.
                                    # If None, will not remap the open width

    #### Oracle info
    # scene creation
    scene_renderer = generate_scene(seed=1, obj_names=obj_names)
    for obj in scene_renderer.objects:
        obj.grasp_poses, obj.open_widths = obj.generate_grasp_family()
        if open_width_cano is not None:
            len = len(obj.open_widths)
            obj.open_widths = [open_width_cano] * len

    # visualization of the GT grasps
    scene_renderer.grasp_color = [0, 255, 0]
    scene_renderer.vis_scene(mode="trimesh", grasp_mode=1, cameras=False)
    scene_renderer.vis_scene(mode="trimesh", grasp_mode=-1, cameras=False)
    colors_orig, depths_orig = scene_renderer.render_imgs(instance_masks=False)
    colors_wg_orig, _ = scene_renderer.render_imgs(instance_masks=False, grasp_mode=1)
    # original rgb
    plt.figure()
    ax = plt.gca()
    ax.axis('off')
    im = ax.imshow(colors_orig[0])
    # original depth
    plt.figure()
    ax = plt.gca()
    ax.axis('off')
    im = ax.imshow(depths_orig[0])

    # rgb with grasps
    plt.figure()
    ax = plt.gca()
    ax.axis('off')
    im = ax.imshow(colors_wg_orig[0])

    plt.show()
    exit()

    color = colors_orig[0]

    # get the grasp 3d info
    grasp_poses_gt, grasp_open_widths, grasp_collides = scene_renderer.get_grasp_infos()
    grasp_poses_gt = np.concatenate(grasp_poses_gt, axis=0)
    grasp_open_widths = np.concatenate(grasp_open_widths, axis=0)
    grasp_collides = np.concatenate(grasp_collides, axis=0)
    grasp_poses_gt = grasp_poses_gt[~grasp_collides, :, :]
    grasp_open_widths = grasp_open_widths[~grasp_collides]

    # clip the open_width
    if open_width_th is not None:
        grasp_open_widths[grasp_open_widths<open_width_th] = open_width_th
        print(grasp_open_widths)

    # get the grasp 2d projections - only one camera
    intrinsic, camera_poses, proj_mats = scene_renderer.get_camera_infos("OpenCV")
    camera_pose = camera_poses[0]
    
    grasp_kpts_2d_oracle = get_grasp_2Dprojs(grasp_poses=grasp_poses_gt, grasp_widths=grasp_open_widths,\
        intrinsic=intrinsic, camera_pose=camera_pose, kpt_type=kpt_type)

    N_grasps = grasp_kpts_2d_oracle.shape[0]

    # perturb the 2d projections
    grasp_kpts_2d_oracle_copy = deepcopy(grasp_kpts_2d_oracle)
    for i in range(N_grasps):
        grasp_kpts_2d_oracle_this = perturb_coords(grasp_kpts_2d_oracle[i, :, :], perturb_mode, stddev)
        grasp_kpts_2d_oracle[i, :, :] = grasp_kpts_2d_oracle_this
    
    #### solve the 3d grasp pose with the oracle 2d projections and the oracle open_width

    # create the solver:
    solver = PnPSolverFactory[solver_name](
        kpt_type = kpt_type,
        camera_intrinsic_matrix = intrinsic
    )


    # Predict and plot the grasp poses
    scene_pred = scene_renderer.to_trimesh_scene(grasp_mode=1, camera_marker=False, world_frame=False, obj_frame=False, gripper_frame=False)
    scene_pred_all = scene_renderer.to_trimesh_scene(grasp_mode=1, camera_marker=False, world_frame=False, obj_frame=False, gripper_frame=False)

    # plot the 2d keypoint predictions
    for i in range(N_grasps):
        color_kpts = plot_grasps_kpts(color, grasp_kpts_2d_oracle, kpts_mode=kpt_type, size=2)
    plt.figure()
    plt.title("The round-off 2d keypoint projections")
    plt.imshow(color_kpts)
    plt.show()

    plt.ion()
    plt.show()
    # predict the grasps
    poses_pred = []
    coord_recover_errors = []
    coord_align_errors = []
    for i in range(N_grasps):

        grasp_kpts_2d_oracle_this = grasp_kpts_2d_oracle[i, :, :]
        # predict the grasp pose
        solver.set_open_width(open_width=grasp_open_widths[i])
        location, quaternion, projected_points, reprojectionError = \
            solver.solve_pnp(
                grasp_kpts_2d_oracle_this
            )
        
        # filter according to the reprojection error
        if reproj_error_th is None or reprojectionError < reproj_error_th:
            print("The reprojection error: {}. Kept".format(reprojectionError))
        else:
            print("The reprojection error: {}. Discarded".format(reprojectionError))
            continue

        r = R.from_quat(quaternion)
        cam_grasp_pose_pred = create_homog_matrix(R_mat=r.as_matrix(), T_vec=location)
        # cam_coords = (cam_pose)^-1 kpt_world_coords = (cam_pose)^-1 grasp_pose kpt_gripper_coords
        # cam_grasp_pose_pred = (cam_pose)^-1 @ grasp_pose
        grasp_pose_pred =  camera_pose @ cam_grasp_pose_pred
        poses_pred.append(grasp_pose_pred)

        # Add the predicted grasp for the 3d visualization
        pose = grasp_pose_pred
        grasp = Grasp(open_width=grasp_open_widths[i], pose = pose, color = [0, 0, 255])
        grasp_mesh = grasp.get_mesh()
        scene_pred.add_geometry(grasp_mesh, node_name="grasp_pred")
        scene_pred_all.add_geometry(deepcopy(grasp_mesh))

        # add the GT grasp for the 3d visualization
        pose = grasp_poses_gt[i]
        grasp = Grasp(open_width=grasp_open_widths[i], pose = pose, color = [0, 255, 0])
        grasp_mesh = grasp.get_mesh()
        scene_pred.add_geometry(grasp_mesh, node_name="grasp_gt")

        # store the GT info for the debug purpose
        grasp = Grasp(grasp_open_widths[i], pose=grasp_poses_gt[i], kpts_option=kpt_type)
        kpts_coord = grasp.get_kpts(frame="world")  # the keypoint coordinate in the world frame, (N_kpt, 3)
        coord_2d, p_c, scales = kpts_3d_to_2d(intrinsic, np.linalg.inv(camera_pose), kpts_coord, output_details=True)
        #coord_mid, p_mid, _ = kpts_3d_to_2d(intrinsic, np.linalg.inv(camera_pose), ((kpts_coord[0, :] + kpts_coord[1, :])/2).reshape(1, 3), output_details=True)


        solver.store_gt_info(
            scales,
            p_c 
        )

        ## debug - Recover the perfect scales and the camera frame coordinates
        if solver_name == "planar":
            solver.output_debug_info()
            coords_recover_error, coords_align_error = solver.get_debug_info(verbose=True)
            coord_recover_errors.append(coords_recover_error)
            coord_align_errors.append(coords_align_error)

        # visualize
        if check_one_by_one:
            color_this = plot_grasps_kpts(color, grasp_kpts_2d_oracle[i:i+1, :, :], kpts_mode=kpt_type, size=4)
            plt.imshow(color_this)
            plt.pause(0.001)
            scene_pred.show()
            plt.close('all')
        scene_pred.delete_geometry(names = ["grasp_pred", "grasp_gt"])
    plt.ioff()
    print("\n")

    # print the mean errors
    print("The average coordinate recover error to the scale prediction is: {} meters".format(np.mean(np.array(coord_recover_errors))))
    print("The average coordinate align error to the ICP is: {} meters".format(np.mean(np.array(coord_align_errors))))

    # add the predicted grasps to the scene, and show
    #scene_pred_all.show()
    scene_renderer.grasp_poses = [poses_pred]
    scene_renderer.grasp_color = [0, 0, 255]
    scene_renderer.vis_scene(mode="trimesh")

    # render the images
    colors, depths, ins_masks = scene_renderer.render_imgs(instance_masks=True, grasp_mode=0)
    plt.imshow(colors[0])
    plt.show()
    
    



