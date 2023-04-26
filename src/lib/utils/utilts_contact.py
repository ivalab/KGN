import numpy as np
from utils.transform import create_homog_matrix, create_rot_mat_axisAlign

import contactGraspNet.contact_graspnet.mesh_utils as mesh_utils 

GRIPPER = mesh_utils.create_gripper('panda')

def frame_trf_cgn2ps(trf_graspnet):
    """
    Args:
        trf_graspnet (array, (N, 4, 4))
    """
    
    # obtain the rotation. graspnet defines the reaching direction as +z, tip-to-tip as x.
    rot = create_rot_mat_axisAlign(align_order=[3, -2, 1])

    # obtain the translation. the graspnet frame is not defined at the center between the finger tips.
    # instead, it is on the tail. So the translation is the max +z value
    grasp_pc = GRIPPER.get_control_point_tensor(1, False, convex_hull=False).squeeze()
    z_translate = np.max(grasp_pc[:, 2])

    T01 = create_homog_matrix(T_vec=[0, 0, z_translate])
    T12 = create_homog_matrix(R_mat=rot)

    return trf_graspnet @ (T01@T12)

