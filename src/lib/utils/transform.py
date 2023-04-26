from scipy.spatial.transform import Rotation as R
import numpy as np

def create_homog_matrix(R_mat=np.eye(3), T_vec=np.zeros(3)):
    """Assemble the rotation and translation matrix to create a homogeneous transformation matrix

    Args:
        R_mat (array, (d,d), optional): The d-by-d rotation matrix. Default to no rotation matrix
            A convenient library to create the 3d rotation matrix is the scipy.spatial.transform:
            https://docs.scipy.org/doc/scipy/reference/spatial.transform.html
        T_vec (array (d, ), optional): The d-by-1 translation vector. Default to no translation matrix

    Returns:
        transform_mat (array, (d+1, d+1)). The homogeneous transformation matrix
    """

    R_mat = np.array(R_mat).squeeze()
    T_vec = np.array(T_vec).squeeze()

    # fetch the shape and assert the requirements
    m, n = R_mat.shape
    assert m == n, "The rotation matrix is not a square matrix. The shape is: {}".format([m, n])
    d = T_vec.size
    assert d == m, "The translation and the rotation are not of the same dimension. The dim(translation): {}, whereas the dim(rot): {}".format(d, m)

    # concatenate
    transform_mat = np.zeros((d+1, d+1), dtype=float)
    transform_mat[:d, :d] = R_mat
    transform_mat[:d, d] = T_vec
    transform_mat[d, d] = 1

    return transform_mat

def homog_matrix_to_RnT(homog_mat, R_format="mat"):
    """
    Break a homogeneous transformation matrix down into rotation and translation
    Args:
        homog_mat (4, 4):       The homog mat to be broken down
        R_format (str):         The format for rotation representation. "mat" or "quat"
                                mat - (3, 3) rotation matrix
                                quat - (4, ) quaternion representation in xyzw
    Return:
        T (3, ):                Translation matrix
        Rot (3, 3) or (4,)
    """
    T = homog_mat[:3, 3]
    Rot = homog_mat[:3, :3]
    if R_format == "mat":
        pass
    elif R_format == "quat":
        Rot = R.from_matrix(Rot).as_quat()
    else:
        raise NotImplementedError
    
    return T, Rot

def apply_homog_transform(points, trf_mat):
    """Apply the homogenesou transform matrix

    Args:
        points (array, (N,d)): The d-dimensional point coordinate as row vectors.
        trf_mat (array, (d+1,d+1)): The (d+1, d+1) homogeneous transformation matrix

    Returns:
        points_trf (array, (N, d)): The transformed point coordinates
    """

    trf_mat = np.array(trf_mat)
    points = np.array(points)

    # check the dimensions
    N, d = points.shape
    m, n = trf_mat.shape
    assert (m == d+1) and (n == d+1), "Expect the homogenous transform matrix and the cartesian point coordinates. \
        The received point coordinate dimension is: {}. The transform matrix dimension is: {}".format(d, [m, n])

    # apply the transform
    points_homog = np.concatenate([points, np.ones((N, 1))], axis=1)
    points_trf_homog = points_homog @ trf_mat.T
    points_trf = points_trf_homog[:, :d]

    return points_trf


def create_rot_mat_axisAlign(align_order):
    """Create the rotation matrix that align the new frame axes with the old axes, but not necessarily in a corresponding order.
    For example, create the rotation matrix to obtain a new frame whose:
    (1) x axis align with the original y
    (2) y axis align with the negative original z
    (3) z align with the original x

    Now only support 3-dimensional space rotation.

    Args:
        align_order (int array, (3,)).  A list with length of 3, containing only 1/2/3 or negative 1/2/3, that describes the alignment requirement.
            The first, second, and third element describes that for the new x, y, z axis respectively.
            The element 1/2/3 means to align with the original x/y/z axis respectively.
            E.g., For the above example, should input [2, -3, 1], and will generate the following matrix:
                [[0, 0, 1],
                [1, 0, 0],
                [0, -1, 0]]
    
    Returns:
        rot_mat (array, (3, 3)). The rotation matrix
    """
    # assert the input is valid
    align_order = np.array(align_order)  
    assert all(num in np.abs(align_order) for num in [1, 2, 3])
    # NOTE: there should be further constraint. Some requirement can not be met.
    # e.g. Align x with x, y with z, z with y, then the new coordinate won'd follow the right-hand rule


    # create the matrix
    rot_mat = np.zeros((3, 3))
    for idx, val in enumerate(align_order):
        rot_mat[abs(val)-1, idx] = val / abs(val)

    return rot_mat

def cam_pose_convert(camera_poses, mode="gl2cv"):
    trf = create_rot_mat_axisAlign([1, -2, -3])
    trf = create_homog_matrix(R_mat=trf)
    if mode == "cv2gl":
        return camera_poses @ trf
    elif mode == "gl2cv":
        return camera_poses @ np.linalg.inv(trf)

if __name__ == "__main__":
    print(create_rot_mat_axisAlign([2, -3,  1]))


    

    