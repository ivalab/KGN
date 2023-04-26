import math
import numpy as np
import os


def QuaternionToMatrix(quaternion, translation, first_w=True):
    # return 4,4
    qw, qx, qy, qz = quaternion

    if not first_w:
        qx, qy, qz, qw = quaternion

    n = 1.0 / math.sqrt(qx * qx + qy * qy + qz * qz + qw * qw)
    qx *= n
    qy *= n
    qz *= n
    qw *= n

    mat = np.zeros((4, 4), dtype=np.float32)
    mat[0, 0] = 1.0 - 2.0 * qy * qy - 2.0 * qz * qz
    mat[0, 1] = 2.0 * qx * qy - 2.0 * qz * qw
    mat[0, 2] = 2.0 * qx * qz + 2.0 * qy * qw

    mat[1, 0] = 2.0 * qx * qy + 2.0 * qz * qw
    mat[1, 1] = 1.0 - 2.0 * qx * qx - 2.0 * qz * qz
    mat[1, 2] = 2.0 * qy * qz - 2.0 * qx * qw

    mat[2, 0] = 2.0 * qx * qz - 2.0 * qy * qw
    mat[2, 1] = 2.0 * qy * qz + 2.0 * qx * qw
    mat[2, 2] = 1.0 - 2.0 * qx * qx - 2.0 * qy * qy

    mat[:3, 3] = translation
    mat[3, 3] = 1

    return mat


def project_camera_to_image(pc_camera, K):
    ''' 
    pc_camera: n, 3
    K: (3, 3)
    '''

    pc_2d = K @ pc_camera.T
    # n, 3
    pc_2d = pc_2d.T
    pc_2d[:, 0] /= pc_2d[:, 2]
    pc_2d[:, 1] /= pc_2d[:, 2]
    return pc_2d


def project_camera_to_image_v2(pc_camera, K):
    ''' 
    pc_camera: n, 3
    K: (3, 3)

    '''

    pc_2d = K @ pc_camera.T
    # n, 3
    pc_2d = pc_2d.T
    pc_2d[:, 0] /= pc_2d[:, 2]
    pc_2d[:, 1] /= pc_2d[:, 2]

    u = pc_2d[:, 0]
    v = pc_2d[:, 1]

    u = np.around(u).astype(np.int64)
    v = np.around(v).astype(np.int64)

    keep = (u >= 0) & (u < 512) & (v >= 0) & (v < 512)

    pc_2d = pc_2d[keep]
    u = u[keep]
    v = v[keep]

    im = np.zeros((512, 512))
    im[v, u] = pc_2d[:, 2]

    return im


def PointCloud2Detph(pc, K, orientation, position, gripper_length, init_pose, shape=(512, 512)):
    '''
    pc: n, 3
    K: (3, 3)
    orientation: quaternion (x, y, z, w)
    init_pose: (4, 4)

    '''
    assert pc.shape[1] == 3

    rot_mat = QuaternionToMatrix(orientation, position, first_w=False)
    rot_inv = np.linalg.inv(rot_mat)
    rot_mat = rot_inv
    # 4, n
    pc_one = np.concatenate([pc, np.ones((pc.shape[0], 1), dtype=np.float32)], 1)

    # pc_camera = rot_mat @ pc_one.T  # 4,n
    # pc_camera = pc_camera[:3, :].T  # 3,n
    # pc_camera = (init_pose[:3, :3] @ pc_camera.T).T

    pc_camera = (init_pose @ rot_mat @ pc_one.T).T
    pc_camera = pc_camera[:, :3]

    # pc_camera = pc_camera * np.array([1, -1, -1]) # n, 3

    # np.savetxt('tmp_camera_2.txt', pc_camera, delimiter=';', fmt='%.3f')

    # np.savetxt('tmp_camera_3.txt', pc_camera, delimiter=';', fmt='%.3f')

    # if (pc_camera[:, 2] <= 0.35).sum() > 0:
    #     warnings.warn( 'ddd: %d' %((pc_camera[:, 2] <= 0.35).sum() > 0))
    #     import ipdb
    #     ipdb.set_trace()

    offset = 0.35
    pc_camera[:, 2] = pc_camera[:, 2] + offset

    keep = pc_camera[:, 2] > 0
    pc_camera = pc_camera[keep, :]

    box = np.array([[-gripper_length / 2, -gripper_length / 2, offset],
                    [gripper_length / 2, gripper_length / 2, offset]])

    pc_2d = project_camera_to_image(pc_camera, K)
    box_2d = project_camera_to_image(box, K)

    # abandon outside points

    u, v = pc_2d[:, 0], pc_2d[:, 1]
    u = np.around(u).astype(np.int64)
    v = np.around(v).astype(np.int64)
    depth = pc_2d[:, 2]

    keep = (u >= 0) & (u < shape[1]) & (v >= 0) & (v < shape[0])

    u = u[keep]
    v = v[keep]
    depth = depth[keep]

    # small value cover large value
    order = np.argsort(-depth)

    v = v[order]
    u = u[order]
    depth = depth[order]

    im = np.zeros((shape[0], shape[1]))
    im[v, u] = depth

    valid_mask = np.zeros((shape[0], shape[1]))
    valid_mask[v, u] = 1

    box = np.around(box_2d[:, :2]).astype(np.int64).reshape(4)

    return im, box, valid_mask


def Depth2PointCloud(dmap, K, orientation=None, position=None, mask=None, org_size=None):
    '''
    K: (3, 3)
    orientation: (4,) quaternion (w, x, y, z)
    position: (3,)
    return (3, n)
    '''

    cx, cy = K[0, 2], K[1, 2]
    fx, fy = K[0, 0], K[1, 1]

    heigh, width = int(dmap.shape[0]), int(dmap.shape[1])
    [x, y] = np.meshgrid(np.arange(0, width), np.arange(0, heigh))
    # print('x shape:', x.ravel().shape)
    # print('y shape:', y.shape)
    if org_size is not None:
        org_h = org_size[0]
        org_w = org_size[1]
    else:
        org_h = heigh
        org_w = width

    cx = cx * width / org_w
    cy = cy * heigh / org_h

    x3 = (x - cx) * dmap * 1.0 / fx * org_w / width
    y3 = (y - cy) * dmap * 1.0 / fy * org_h / heigh
    z3 = dmap

    if mask is not None:
        y_idx, x_idx = mask.nonzero()
        x3 = x3[y_idx, x_idx]
        y3 = y3[y_idx, x_idx]
        z3 = z3[y_idx, x_idx]
    # print('x3:', x3.shape)
    # 3, n
    pc = np.stack([x3.ravel(), -y3.ravel(), -z3.ravel()], axis=0)

    if orientation is not None and position is not None:
        ex = QuaternionToMatrix(orientation, position)
        # 4, n
        pc_one = np.concatenate([pc, np.ones((1, pc.shape[1]))], 0)
        # 3, 4 x 4, n
        # ex_inv = np.linalg.inv(ex)
        # ex_inv = ex
        pc_one = ex @ pc_one
        pc = pc_one[:3, :]
    return pc


def grid_sample(x, s=2):
    assert x.ndim == 2
    return x[::s, :][:, ::s]


def Depth2PointCloudV2(dmap, K, orientation=None, position=None):
    '''
    K: (3, 3)
    orientation: (4,) quaternion (w, x, y, z)
    position: (3,)
    return (3, n)
    '''

    cx, cy = K[0, 2], K[1, 2]
    fx, fy = K[0, 0], K[1, 1]

    heigh, width = int(dmap.shape[0]), int(dmap.shape[1])
    [x, y] = np.meshgrid(np.arange(0, width), np.arange(0, heigh))

    x = grid_sample(x)
    y = grid_sample(y)
    dmap = grid_sample(dmap)

    x3 = (x - cx) * dmap * 1 / fx
    y3 = (y - cy) * dmap * 1 / fy
    z3 = dmap
    # 3, n
    pc = np.stack([x3.ravel(), -y3.ravel(), -z3.ravel()], axis=0)

    if orientation is not None and position is not None:
        ex = QuaternionToMatrix(orientation, position)
        # 4, n
        pc_one = np.concatenate([pc, np.ones((1, pc.shape[1]), dtype=np.float32)], 0)
        # 3, 4 x 4, n
        # ex_inv = np.linalg.inv(ex)
        # ex_inv = ex
        pc_one = ex @ pc_one
        pc = pc_one[:3, :]

    return pc


def encode_depth_to_image(dmap):
    min_v = np.min(dmap)
    max_v = np.max(dmap)

    v_range = max(1e-5, (max_v - min_v))
    dmap_norm = (dmap - min_v) / v_range

    dmap_norm = (dmap_norm * 2 ** 8).astype(np.uint8)
    dmap_norm[dmap == 0] = 255

    return dmap_norm


def getDictFromCameraInfoNpArray(camera_info_file):
    camera_info_array = np.load(camera_info_file)
    cameraInfoDict = {}
    for item in camera_info_array:
        cameraInfoDict[item['id'].decode()] = (item['position'], item['orientation'], item['calibration_matrix'])
    return cameraInfoDict


def grasp_view(dmap, intrinsic, camera_ori, camera_pos, grasp_ori, grasp_center, grasp_length, pc=None):
    # rectify the grasp pose to camera pose. we need to ensure the camera z-axis orient to x-axis (the robot gripper direction)

    init_rot = np.array([[0, -1, 0], [0, 0, -1], [1, 0, 0]])
    init_pose = np.zeros((4, 4))
    init_pose[:3, :3] = init_rot
    init_pose[3, 3] = 1

    pc = Depth2PointCloud(dmap, intrinsic, camera_ori, camera_pos)

    depth_cam, box, mask = PointCloud2Detph(pc.T, intrinsic, grasp_ori, grasp_center, grasp_length, init_pose)

    crop_depth = depth_cam[box[1]:box[3] + 1, box[0]:box[2] + 1]
    crop_mask = mask[box[1]:box[3] + 1, box[0]:box[2] + 1]
    # return depth_cam, box, mask
    return crop_depth, crop_mask


def grasp_view_pc(dmap, intrinsic, camera_ori, camera_pos, grasp_ori, grasp_center, grasp_length, pc, vis=False):
    # rectify the grasp pose to camera pose. we need to ensure the camera z-axis orient to x-axis (the robot gripper direction)

    init_rot = np.array([[0, -1, 0], [0, 0, -1], [1, 0, 0]])
    init_pose = np.zeros((4, 4), dtype=np.float32)
    init_pose[:3, :3] = init_rot
    init_pose[3, 3] = 1

    # pc = Depth2PointCloud(dmap, intrinsic, camera_ori, camera_pos)

    depth_cam, box, mask = PointCloud2Detph(pc.T, intrinsic, grasp_ori, grasp_center, grasp_length, init_pose)

    crop_depth = depth_cam[box[1]:box[3] + 1, box[0]:box[2] + 1]
    crop_mask = mask[box[1]:box[3] + 1, box[0]:box[2] + 1]
    # return depth_cam, box, mask
    if not vis:
        return crop_depth, crop_mask
    else:
        return depth_cam, box, crop_depth, crop_mask


if __name__ == '__main__':

    from skimage import io
    import cv2

    import matplotlib.pyplot as plt

    # folder = '3441002a52b1cb2946b2a76c074a3f45'
    folder = '217dd75324b38d07fec515d9b77eb8c1'
    cam = getDictFromCameraInfoNpArray('CameraInfo.npy')

    for k, v in cam.items():
        print(k)

        if '02_agl' not in k:
            continue
        dmap = io.imread(os.path.join(folder, k + '_depth0001.tiff'))
        img = io.imread(os.path.join(folder, k + '.png'))
        mask = io.imread(os.path.join(folder, k + '_mask0001.png'))  # 0,255

        position, orientation, intrinsic = v[0], v[1], v[2]

        intrinsic = intrinsic.reshape(3, 3)

        anns = np.load(folder + '.npy')
        # print(anns)
        idx = anns['success'].nonzero()[0]
        ann = anns[idx[0]]
        success = ann['success']

        grasp_center = ann['gripperCenter']
        grasp_length = ann['gripperLength']
        grasp_ori = ann['gripperOrientaion']

        # pc = np.load('meta/' + folder + '.npy')[:, :3].T
        depth_cam, box = grasp_view(dmap, intrinsic, orientation, position, grasp_ori, grasp_center, grasp_length)

        dmap_image = encode_depth_to_image(depth_cam)

        cv2.rectangle(dmap_image, (box[0], box[1]),
                      (box[2], box[3]), (255, 255, 255), 1)

        plt.imshow(dmap_image)
        # plt.show()

        crop_depth = depth_cam[box[1]:box[3] + 1, box[0]:box[2] + 1]

        fig = plt.figure(2)

        plt.imshow(crop_depth)
        plt.show()

        import ipdb

        ipdb.set_trace()