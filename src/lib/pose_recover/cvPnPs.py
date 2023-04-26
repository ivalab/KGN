# Copyright (c) 2021 NVIDIA Corporation. All rights reserved.
# This work is licensed under the NVIDIA Source Code License - Non-commercial.
# Full text can be found in LICENSE.md

from mimetypes import init
import cv2
import numpy as np
from scipy.spatial.transform import Rotation as R

from pose_recover.Base import BasePnPSolver


class CVEPnPSolver(BasePnPSolver):
    """
    This class is used to find the 6-DoF pose of a grasp given its projected keypoints.
    Runs perspective-n-point (PNP) algorithm.
    """

    def __init__(self, 
                kpt_type="hedron", 
                open_width = 1,
                use_center=False,
                camera_intrinsic_matrix=None, 
                dist_coeffs=np.zeros((4, 1)),
                min_required_points=4
            ):
        super().__init__(kpt_type, open_width, use_center, camera_intrinsic_matrix, dist_coeffs, min_required_points)

        # use the EPnP algorithm - both are tried, but not performing well in the oracle test
        #self.pnp_algorithm = cv2.SOLVEPNP_ITERATIVE
        self.pnp_algorithm = cv2.SOLVEPNP_EPNP



    def solve_pnp(self,
                grasp_kpts_2d, 
                center_2d = None,
                fail_if_projected_diff_exceeds=250,
                fail_if_projected_value_exceeds=1e5,
                verbose = False
                ):
        """
        Detects the rotation and traslation 
        of a grasp from its vertexes' 
        2D location in the image
        Inputs:
        - grasp_kpts_2d:  (N_kpts, 2) shape array
        - centers_2d: (1, 2) shape array
          ...
        Outputs:
        - location in 3D
        - pose in 3D (as quaternion)
        - projected points:  np.ndarray of np.ndarrays
        """
        location = None
        quaternion = None

        location_new = None
        quaternion_new = None
        reprojectionError = None

        obj_2d_points = np.array(grasp_kpts_2d, dtype=float)
        obj_3d_points = np.array(self.get_grasp_kpts_3d(), dtype=float)

        # use center
        if self.use_center:
            assert center_2d is not None
            obj_2d_points = np.concatenate([obj_2d_points, center_2d], axis=0)
            center_3d = (obj_3d_points[0:1, :] + obj_3d_points[1:2, :]) / 2.
            obj_3d_points = np.concatenate([obj_3d_points, center_3d], axis=0)


        valid_point_count = len(obj_2d_points)

        # Can only do PNP if we have more than 3 valid points
        is_points_valid = valid_point_count >= self.min_required_points

        if is_points_valid:
            # Usually, we use this one
            # NOTE: here we must pass the reprojectionError argument, following:
            # https://github.com/opencv/opencv/issues/16049
            r = np.array([], dtype=np.float32)
            ret, rvec, tvec, reprojectionError = cv2.solvePnPGeneric(
                obj_3d_points,
                obj_2d_points,
                self._camera_intrinsic_matrix,
                self._dist_coeffs,
                flags=self.pnp_algorithm,
                reprojectionError=r
            )

            if ret:

                rvec = np.array(rvec[0])
                tvec = np.array(tvec[0])
                reprojectionError = reprojectionError.flatten()[0]

                # Convert OpenCV coordinate system to OpenGL coordinate system
                transformation = np.identity(4)
                r = R.from_rotvec(rvec.reshape(1, 3))
                transformation[:3, :3] = r.as_matrix()
                transformation[:3, 3] = tvec.reshape(1, 3)
                M = np.zeros((4, 4))
                M[0, 1] = 1
                M[1, 0] = 1
                M[3, 3] = 1
                M[2, 2] = -1
                transformation = np.matmul(M, transformation)

                rvec_new = R.from_matrix(transformation[:3, :3]).as_rotvec()
                tvec_new = transformation[:3, 3]

                # OpenGL result, to be compared against GT
                location_new = list(x for x in tvec_new)
                quaternion_new = self.convert_rvec_to_quaternion(rvec_new)

                # OpenCV result
                location = list(x[0] for x in tvec)
                quaternion = self.convert_rvec_to_quaternion(rvec)

                # Still use OpenCV way to project 3D points
                projected_points, _ = cv2.projectPoints(obj_3d_points, rvec, tvec, self._camera_intrinsic_matrix,
                                                        self._dist_coeffs)
                projected_points = np.squeeze(projected_points)

                # Todo: currently, we assume pnp fails if z<0 or any result is NaN
                x, y, z = location
                if z < 0 or (np.any(np.isnan(quaternion))):
                    # # Get the opposite location
                    # location = [-x, -y, -z]
                    #
                    # # Change the rotation by 180 degree
                    # rotate_angle = np.pi
                    # rotate_quaternion = Quaternion.from_axis_rotation(location, rotate_angle)
                    # quaternion = rotate_quaternion.cross(quaternion)
                    location = None
                    quaternion = None

                    if verbose:
                        print("PNP solution is behind the camera (Z < 0) => Fail")
                else:
                    if verbose:
                        print("solvePNP found good results - location: {} - rotation: {} !!!".format(location, quaternion))

                    return location, quaternion, projected_points, reprojectionError
            else:
                if verbose:
                    print('Error:  solvePnP return false ****************************************')
        else:
            if verbose:
                print("Need at least 4 valid points in order to run PNP. Currently: {}".format(valid_point_count))



class CVP3PSolver(CVEPnPSolver):
    def __init__(self, kpt_type="hedron", open_width=1, use_center=False, camera_intrinsic_matrix=None, dist_coeffs=np.zeros((4, 1)), min_required_points=4):
        super().__init__(kpt_type,  open_width, use_center,camera_intrinsic_matrix, dist_coeffs, min_required_points)
        self.pnp_algorithm = cv2.SOLVEPNP_P3P
    

class CVIPPESolver(CVEPnPSolver):
    def __init__(self, kpt_type="hedron",  open_width=1, use_center=False,camera_intrinsic_matrix=None, dist_coeffs=np.zeros((4, 1)), min_required_points=4):
        if kpt_type == "hedron":
            raise ValueError("The IPPE solver only support planar keypoint types. The current {} is not one of them.".format(kpt_type))
        super().__init__(kpt_type, open_width, use_center, camera_intrinsic_matrix, dist_coeffs, min_required_points)
        self.pnp_algorithm = cv2.SOLVEPNP_IPPE