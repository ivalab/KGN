from venv import create
import numpy as np
import os
import argparse

from utils.transform import create_homog_matrix, create_rot_mat_axisAlign
from opts import opts

from graspnet_6dof.utils.utils import get_control_point_tensor


def frame_trf_ps2graspnet(trf_ps):

    # obtain the rotation. graspnet defines the reaching direction as +z, tip-to-tip as x.
    rot = create_rot_mat_axisAlign(align_order=[3, -2, 1])

    # obtain the translation. the graspnet frame is not defined at the center between the finger tips.
    # instead, it is on the tail. So the translation is the max +z value
    grasp_pc = np.squeeze(get_control_point_tensor(1, False), 0)
    z_translate = np.max(grasp_pc[:, 2])

    T01 = create_homog_matrix(R_mat=rot)
    T12 = create_homog_matrix(T_vec=[0, 0, -z_translate])

    return trf_ps @ (T01@T12)


def frame_trf_graspnet2ps(trf_graspnet):
    """
    Args:
        trf_graspnet (array, (N, 4, 4))
    """
    
    # obtain the rotation. graspnet defines the reaching direction as +z, tip-to-tip as x.
    rot = create_rot_mat_axisAlign(align_order=[3, -2, 1])

    # obtain the translation. the graspnet frame is not defined at the center between the finger tips.
    # instead, it is on the tail. So the translation is the max +z value
    grasp_pc = np.squeeze(get_control_point_tensor(1, False), 0)
    z_translate = np.max(grasp_pc[:, 2])

    T01 = create_homog_matrix(T_vec=[0, 0, z_translate])
    T12 = create_homog_matrix(R_mat=rot)

    return trf_graspnet @ (T01@T12)

def addGraspNetOpts(opts:opts):
    """The arguments adopted from the graspnet.demo.main.make_parser() function"""

    opts.parser.add_argument('--grasp_sampler_folder',
                        type=str,
                        default='./graspnet_6dof/checkpoints/gan_pretrained/')
    opts.parser.add_argument('--grasp_evaluator_folder',
                        type=str,
                        default=None)
    opts.parser.add_argument('--refinement_method',
                        choices={"gradient", "sampling"},
                        default='sampling')
    opts.parser.add_argument('--refine_steps', type=int, default=25)

    opts.parser.add_argument('--npy_folder', type=str, default='./graspnet_6dof/demo/data/')
    opts.parser.add_argument(
        '--threshold',
        type=float,
        default=0.8,
        help=
        "When choose_fn is something else than all, all grasps with a score given by the evaluator notwork less than the threshold are removed"
    )
    opts.parser.add_argument(
        '--choose_fn',
        choices={
            "all", "better_than_threshold", "better_than_threshold_in_sequence"
        },
        default='better_than_threshold',
        help=
        "If all, no grasps are removed. If better than threshold, only the last refined grasps are considered while better_than_threshold_in_sequence consideres all refined grasps"
    )

    opts.parser.add_argument('--target_pc_size', type=int, default=1024)
    opts.parser.add_argument('--num_grasp_samples', type=int, default=100)
    opts.parser.add_argument(
        '--generate_dense_grasps',
        action='store_true',
        help=
        "If enabled, it will create a [num_grasp_samples x num_grasp_samples] dense grid of latent space values and generate grasps from these."
    )

    opts.parser.add_argument(
        '--sample_batch_size',
        type=int,
        default=10,
        help=
        "Set the batch size of the number of grasps we want to process and can fit into the GPU memory at each forward pass. The batch_size can be increased for a GPU with more memory."
    )
    return opts