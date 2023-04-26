import cv2
from cv2 import dnn_KeypointsModel
from cv2 import reprojectImageTo3D
from matplotlib import scale
from matplotlib.pyplot import axis
import numpy as np
from pyrr import Quaternion
from scipy.spatial.transform import Rotation
from scipy.linalg import null_space
from torch import dtype

from pose_recover.Base import BasePnPSolver
from utils.keypoints import kpts_3d_to_2d
from utils.transform import apply_homog_transform, create_homog_matrix

class PlanerPnPSolver(BasePnPSolver):
    """
    Implementation of the PnP with the coplanar constraints.
    Currently only support the four points input
    """
    def __init__(self, 
                kpt_type="box", 
                open_width = 1,
                camera_intrinsic_matrix=None, 
                dist_coeffs=np.zeros((4, 1)),
                min_required_points=4
            ):
        assert kpt_type in ["box", "tail"],"Planar PnP solver only support planar keypoints type."
        super().__init__(kpt_type, open_width, camera_intrinsic_matrix, dist_coeffs, min_required_points)


        # store information for the debug purpose
        self.scales_pred = None
        self.coords_cam_pred = None

        self.scales_gt = None
        self.coords_cam_gt = None

    def solve_pnp(self,
                grasp_kpts_2d, 
                fail_if_projected_diff_exceeds=250,
                fail_if_projected_value_exceeds=1e5,
                verbose = False
                ):
        """
        Get the  
          ...
        Outputs:
        - location in 3D
        - pose in 3D (as quaternion)
        - projected points:  np.ndarray of np.ndarrays
        - reprojection error:   defined as the root mean square error (RMSE) between the input 2d kpts and the reprojected 2d kpts
        """
        # the 3d keypoints - and get the 2d case
        # NOTE: in my gripper frame design the points lie in the x-z plane
        kpts_3d = self.get_grasp_kpts_3d()
        kpts_3d_plane_2d = kpts_3d[:, [0, 2]]

        # homography of the 2d points
        kpts_2d_h = self.eu_to_homog(grasp_kpts_2d)

        # select root point - Both planar kpt type has four points
        idx_all = np.array([0, 1, 2, 3])
        for root_pt_idx in idx_all:
            # The root and other 2d points
            root_2d_h = kpts_2d_h[root_pt_idx, :].reshape((1, -1))
            others_2d_h = kpts_2d_h[idx_all != root_pt_idx, :]

            # the root and other 3d points in the same order
            root_3d_plane_2d = kpts_3d_plane_2d[root_pt_idx, :].reshape((1, -1))
            others_3d_plane_2d = kpts_3d_plane_2d[idx_all != root_pt_idx, :]

            # stack them together
            coords_2d_h = np.concatenate([root_2d_h, others_2d_h], axis = 0)
            coords_3d_world = np.concatenate(
                [kpts_3d[root_pt_idx, :].reshape((1, -1)), kpts_3d[idx_all!=root_pt_idx, :]],
                axis = 0
            )

            # get the linearly dependent vecors, and then the coefficents
            linear_Depend_Vecs = others_3d_plane_2d - root_3d_plane_2d
            alphas = self.cal_linear_depend_alphas(linear_Depend_Vecs)


            # construct the B matrix - (3, 4)
            B = self.construct_B(alphas, root_2d_h, others_2d_h)

            # get the Null space direction (4, n_null)
            # NOTE: there is one problem here: The solved scale direction might not be all positive.
            # If this case, just skip?
            ns = null_space(B)
            if ns.shape[1] != 1:
                # 3 or more colinear 2d projections, then direction should be easily determined
                return None, None, None, None
                raise NotImplementedError
            ns = ns[:, 0]
            if np.all(ns < 0): ns = -ns
            assert np.all(ns > 0), "{}".format(ns)

            # solve the scale beta
            beta = self.cal_ns_scale(ns, coords_2d_h, coords_3d_world)

            # Construct the camera frame 3d point coordinates, and solve the result
            scales = beta * ns
            root_3d_cam = root_2d_h * scales[0]
            others_3d_cam = others_2d_h * (scales[1:].reshape((-1, 1)))
            coords_3d_cam = np.concatenate([root_3d_cam, others_3d_cam], axis = 0)
            coords_3d_cam = coords_3d_cam @ np.linalg.inv(self._camera_intrinsic_matrix).T
            R, T = self.ICP(source=coords_3d_world, target=coords_3d_cam)

            # calculate the coordinates align error
            coords_3d_world_transformed = apply_homog_transform(
                points=coords_3d_world,
                trf_mat=create_homog_matrix(R_mat=R, T_vec=T)
            )
            self.coord_align_error = np.mean(
                np.linalg.norm(
                    coords_3d_world_transformed - coords_3d_cam,
                    axis=1
                )
            )
            
            # store the process results
            self.scales_pred = scales
            self.coords_cam_pred = coords_3d_cam

            # convert to the return system
            location = T
            r = Rotation.from_matrix(R)
            quaternion = r.as_quat()

            # calculate the projected ponits and the reprojection error
            rvec = cv2.Rodrigues(R)[0]
            projected_points, _ = cv2.projectPoints(kpts_3d, rvec, T, self._camera_intrinsic_matrix,
                                                    self._dist_coeffs)
            projected_points = np.squeeze(projected_points)
            reprojectionError = np.mean(
                np.linalg.norm(grasp_kpts_2d - projected_points, 2, axis=1)**2
            )**0.5
            self.reprojectionError = reprojectionError

            # NOTE: For now only use the first keypoint as the root
            # Return OpenCV result for demo
            return location, quaternion, projected_points, reprojectionError
    
    def cal_linear_depend_alphas(self, lDep_vec):
        """Calculate the coefficients of the zero linear sum. The coefficients will be normalize to 1 norm
        
        SO far only consider the case when there is a unique solution
        i.e. No two collinear vectors; minimal linearly dependent set
        Args:
            lDep_vect (N, D)
        """
        N, D = lDep_vec.shape
        if N != D+1:
            raise NotImplementedError("Only support minimal linearly depenent set") 
        
        W = lDep_vec.T
        ns = null_space(W)
        if ns.shape[1] != 1:
            raise NotImplementedError("Does not support collinear vector input")
        alphas = ns[:, 0]

        return alphas

    def construct_B(self, alphas, root_2d_h, others_2d_h):
        """The weight matrix for solving the scales

        Args:
            alphas (N_other): The linear dependency weights
            root_2d_h ((1, D)): 
            others_2d_h ((N_other, D)): 

        Returns:
            B(D, N): D=N-1, N=N_other+1
        """
        root_2d_h = root_2d_h.T
        others_2d_h = others_2d_h.T

        B_cols = []
        B_cols.append(np.sum(alphas) * root_2d_h)

        N_other, D = others_2d_h.shape
        for i in range(N_other):
            B_cols.append(-alphas[i] * others_2d_h[:, i:i+1])

        B = np.concatenate(B_cols, axis=1)

        return B

    def ICP(self, source, target, weights=None):
        """ Get the transformations that map the source coordinates to the target coordinates
        Implement the algorithm explained here: https://igl.ethz.ch/projects/ARAP/svd_rot.pdf
        Args:
            source (N, D)
            target (N, D)
            weights (N)
        Returns:
            R, T: The rotation matrix and the translation vector that transforms the source to the target
        """
        q = source
        p = target
        N_q, D_q = q.shape
        N_p, D_p = p.shape
        assert (N_q == N_p) and (D_q == D_p)
        if weights is None:
            weights = np.ones((N_q))
        q_avg = np.mean(q, axis=0)
        p_avg = np.mean(p, axis=0)
        q_tilt = (q - q_avg.reshape((1, -1))).T     #(D, N)
        p_tilt = (p - p_avg.reshape((1, -1))).T     #(D, N)
        W = np.diag(weights)

        # singular value decompostion
        u, s, vh = np.linalg.svd(q_tilt @ W @ p_tilt.T)

        # derive the transformation
        sign_mat = np.diag(np.ones(D_p, dtype=float))
        sign_mat[-1, -1] = np.linalg.det( vh.T @ u )
        R = vh.T @ sign_mat @ u.T
        T = p_avg.reshape((-1, 1)) - R @ q_avg.reshape((-1, 1))
        
        return R, T.squeeze()

    def cal_ns_scale(self, ns, coords_2d_h, coords_3d_world):
        """Solve the exact scale given the scale direction
        Inputs:
            coords_2d_h (N, 3)
            coords_3d_world (N, 3)
        """
        N = coords_2d_h.shape[0]

        K_inv = np.linalg.inv(self._camera_intrinsic_matrix)

        numerator = 0
        denominator = 0
        for i in range(N):
            for j in range(N):
                if j == i:
                    continue
                scaled_2d_h_i = ns[i] * coords_2d_h[i, :] 
                scaled_2d_h_j = ns[j] * coords_2d_h[j, :]
                a_ij = np.linalg.norm(
                    K_inv @ (scaled_2d_h_i - scaled_2d_h_j),
                    2
                )

                b_ij = np.linalg.norm(
                    (coords_3d_world[i, :] - coords_3d_world[j, :]),
                    2
                )

                numerator += a_ij * b_ij
                denominator += a_ij**2
        beta = numerator / denominator
        return beta

    def get_debug_info(self, verbose=False):
        """For figuring out which part introduces the most error.
        It will calculate the average error between the predicted camera frame coordinates and the GT camera frame coordinates (i.e. the scale prediction error)
            and that between the predicted transformed world coordinates and predicted camera frame coordinates (i.e. The transformation estimation error) 
        """
        # the scale prediciton error
        assert self.coords_cam_gt is not None and self.coords_cam_pred is not None
        coords_recover_error = np.mean(
            np.linalg.norm(
                self.coords_cam_gt - self.coords_cam_pred,
                axis=1
            )
        )

        # the coords align error
        coords_align_error = self.coord_align_error 

        # verbose
        if verbose:
            print("The coordinate recover error (due to the scales prediction): {}".format(coords_recover_error))
            print("The coordinate align error (due to the transformation prediction): {}".format(coords_align_error))

        return coords_recover_error, coords_align_error

    
    def output_debug_info(self):

        print("\n")

        print("The predicted scales: {}".format(self.scales_pred))
        print("The GT scales: {}".format(self.scales_gt))

        print("\n")

        print("The predicted coords in the camera frame: {}".format(self.coords_cam_pred))
        print("The GT coords in the camera frame: {}".format(self.coords_cam_gt))

        print("\n")


if __name__ == "__main__":
    solver = PlanerPnPSolver(kpt_type="box")

    trf = create_homog_matrix(R_mat=np.eye(3), T_vec=[1, 1, 1])

    pts_list = []
    pts_list.append([1, 0, 0])
    pts_list.append([0, 1, 0])
    pts_list.append([0, 0, 1])
    pts_list.append([0.2, 0.2, 0.2])
    points = np.array(pts_list)

    points_trf = apply_homog_transform(points, trf)

    print("The source points: {}".format(points))
    print("The transformed points: {}".format(points_trf))
    print("The transformation matrix: {}".format(trf))

    R, T = solver.ICP(points, points_trf)
    print(R)
    print(T)