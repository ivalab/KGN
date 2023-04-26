import cv2
from pyrr import Quaternion
import numpy as np
from scipy.spatial.transform import Rotation as R
from grasp_kpts import GraspKpts3d


class BasePnPSolver(object):
    """
    The base class for solving the PnP problem 
    """

    # Class variables
    cv2version = cv2.__version__.split('.')
    cv2majorversion = int(cv2version[0])

    def __init__(self,
                 kpt_type = "hedron",
                 open_width = 1,
                 use_center = False,
                 camera_intrinsic_matrix=None,
                 dist_coeffs=np.zeros((4, 1)),
                 min_required_points=4
                ):
                
        self.min_required_points = max(4, min_required_points)

        self._camera_intrinsic_matrix = camera_intrinsic_matrix
        self._dist_coeffs = dist_coeffs

        self.kpt_type = kpt_type
        self.use_center = use_center

        # the 3d kpt generator
        self.open_width = open_width
        self.kpts_3d_generator = GraspKpts3d(
            open_width=self.open_width,
            kpt_type=self.kpt_type
        )

    def set_camera_intrinsic_matrix(self, new_intrinsic_matrix):
        '''Sets the camera intrinsic matrix'''
        self._camera_intrinsic_matrix = new_intrinsic_matrix

    def set_dist_coeffs(self, dist_coeffs):
        '''Sets the camera intrinsic matrix'''
        self._dist_coeffs = dist_coeffs
    
    def set_kpt_type(self, kpt_type):
        self.kpt_type = kpt_type
        self.kpts_3d_generator.set_kpt_type(kpt_type)
    
    def set_open_width(self, open_width):
        self.open_width = open_width
        self.kpts_3d_generator.set_open_width(open_width)
    
    def assert_cam_intrinsic(self):
        """Make sure the camera intrinsic is initialized."""
        return self._camera_intrinsic_matrix is not None
    
    def get_grasp_kpts_3d(self):
        """
        Get the grasp kpts 3d coordinates in the local grasp frame.
        Returns:
            kpts_3d ((N_kpts, 3))
        """
        return self.kpts_3d_generator.get_local_vertices()

    def __check_pnp_result(self,
                           points,
                           projected_points,
                           fail_if_projected_diff_exceeds,
                           fail_if_projected_value_exceeds):
        """
        Check whether the output of PNP seems reasonable
        Inputs:
        - cuboid2d_points:  list of XY tuples
        - projected points:  np.ndarray of np.ndarrays
        """
        p1 = points
        p2 = projected_points.tolist()
        assert len(p1) == len(p2)

        # Compute max Euclidean 2D distance b/w points and projected points
        max_euclidean_dist = 0
        for i in range(len(p1)):
            if p1[i] is not None:
                dist = np.linalg.norm(np.array(p1[i]) - np.array(p2[i]))
                if dist > max_euclidean_dist:
                    max_euclidean_dist = dist

        # Compute max projected absolute value
        max_abs_value = 0
        for i in range(len(p2)):
            assert len(p2[i]) == 2
            for val in p2[i]:
                if val > max_abs_value:
                    max_abs_value = val

        # Return success (true) or failure (false)
        return max_euclidean_dist <= fail_if_projected_diff_exceeds \
               and max_abs_value <= fail_if_projected_value_exceeds

    def solve_pnp(self,
                grasp_kpts_2d, 
                OPENCV_RETURN = False,
                fail_if_projected_diff_exceeds=250,
                fail_if_projected_value_exceeds=1e5,
                verbose = False
                ):
        raise NotImplementedError
       

    # utility functions
    def convert_rvec_to_quaternion(self, rvec):
        '''Convert rvec (which is log quaternion) to quaternion in the order of (x, y, z, w)'''
        theta = np.sqrt(rvec[0] * rvec[0] + rvec[1] * rvec[1] + rvec[2] * rvec[2])  # in radians
        raxis = [rvec[0] / theta, rvec[1] / theta, rvec[2] / theta]

        # pyrr's Quaternion (order is XYZW), https://pyrr.readthedocs.io/en/latest/oo_api_quaternion.html
        return Quaternion.from_axis_rotation(raxis, theta)

        # Alternatively: pyquaternion
        # return Quaternion(axis=raxis, radians=theta)  # uses OpenCV's Quaternion (order is WXYZ)

    def eu_to_homog(self, eu_coords):
        """
        The euclidean coordinate to the homography coordinate
        Input:
            eu_coords (N, D): The euclidean coordinates
        Returns:
            homog_coords (N, D+1): The homography
        """
        eu_coords = np.array(eu_coords)
        N, D = eu_coords.shape
        homog_coords = np.concatenate(
            [eu_coords, np.ones((N, 1), dtype=float)],
            axis=1
        )
        return homog_coords
    
    def store_gt_info(self, scales_gt, coords_cam_gt):
        """Store GT info for debug"""
        self.scales_gt = scales_gt
        self.coords_cam_gt = coords_cam_gt 
    
