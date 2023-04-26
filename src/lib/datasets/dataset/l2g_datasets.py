"""
The YCB-8 and YCB-76 datasets adopted from L2G paper.
Adopted from:  https://github.com/antoalli/L2G/blob/main/dataset/sample_grasp_dataset.py
"""

import os
import sys
import os.path as osp
import cv2
import numpy as np
import torch
import torch.utils.data as data
from datasets.dataset.depth2points import Depth2PointCloud
import Imath
import OpenEXR


def read_camera_info(camera_info_file):
    camera_info_array = np.load(camera_info_file)
    cameraInfoDict = {}
    for item in camera_info_array:
        cameraInfoDict[item['id'].decode()] = (item['position'],
                                               item['orientation'],
                                               item['calibration_matrix'])
    return cameraInfoDict


def random_camera_view(cameraInfoDict=None):
    view_num = len(cameraInfoDict)
    view = np.random.choice(view_num, 1)[0]
    return cameraInfoDict['view%d' % view], view


def exr2tiff(exrpath):
    File = OpenEXR.InputFile(exrpath)
    PixType = Imath.PixelType(Imath.PixelType.FLOAT)
    DW = File.header()['dataWindow']
    Size = (DW.max.x - DW.min.x + 1, DW.max.y - DW.min.y + 1)
    rgb = [np.frombuffer(File.channel(c, PixType), dtype=np.float32) for c in 'RGB']
    r = np.reshape(rgb[0], (Size[1], Size[0]))
    mytiff = r
    return mytiff


def read_image(img_root, view):
    img_path = img_root
    # img = io.imread(img_path+'/render%d.png'%(view))
    depth = exr2tiff(img_path + '/render%dDepth0001.exr' % view)
    return depth


# quaternion : torch tensor of size [bs, 4]
# return matrix: torch tensor of size [bs, 9]
def quaternion2matrix(quaternion):
    sw = quaternion[:, 0] * quaternion[:, 0]
    sx = quaternion[:, 1] * quaternion[:, 1]
    sy = quaternion[:, 2] * quaternion[:, 2]
    sz = quaternion[:, 3] * quaternion[:, 3]

    m00 = (sx - sy - sz + sw)
    m11 = (-sx + sy - sz + sw)
    m22 = (-sx - sy + sz + sw)

    tmp1 = quaternion[:, 1] * quaternion[:, 2]
    tmp2 = quaternion[:, 3] * quaternion[:, 0]
    m10 = 2.0 * (tmp1 + tmp2)
    m01 = 2.0 * (tmp1 - tmp2)

    tmp1 = quaternion[:, 1] * quaternion[:, 3]
    tmp2 = quaternion[:, 2] * quaternion[:, 0]
    m20 = 2.0 * (tmp1 - tmp2)
    m02 = 2.0 * (tmp1 + tmp2)

    tmp1 = quaternion[:, 2] * quaternion[:, 3]
    tmp2 = quaternion[:, 1] * quaternion[:, 0]
    m21 = 2.0 * (tmp1 + tmp2)
    m12 = 2.0 * (tmp1 - tmp2)
    return torch.stack([m00, m01, m02, m10, m11, m12, m20, m21, m22], dim=1)


def zMove(quaternion, gripper_position, z_move_length=-0.015):
    """
    The authors of the original dataset forgot to remove a 1.5cm offset on the z-axis
    before releasing it. This function fixes this issue.
    """
    quaternion_ = torch.tensor(quaternion)
    rotation_matrix = quaternion2matrix(quaternion_).numpy().reshape(-1, 3, 3)
    rotation_z = rotation_matrix[:, :, 2]
    move_z = rotation_z * z_move_length
    gripper_position = gripper_position + move_z  # move away from object: avoid collision
    return gripper_position


def prune_and_normalize(tensor):
    """
    Scale the GPNet data to a unit dimension.
    Moreover, it prunes some points out of the bounds
    @param tensor: the tensor to be normalized
    @return:
    """
    tensor_x = tensor[:, 0]
    tensor_y = tensor[:, 1]
    tensor_z = tensor[:, 2]
    del_idx = (tensor_x < -0.22 / 2) | (tensor_x > 0.22 / 2) | (tensor_y < -0.22 / 2) | (tensor_y > 0.22 / 2) | (
            tensor_z > 0.22)
    tensor = tensor[del_idx == False]
    tensor = tensor / np.array([0.22 / 2, 0.22 / 2, 0.22])
    return tensor


def get_point_cloud_from_cam_info(img_path, camera_pose, view, shelf_th=0.002):
    ca_loc = camera_pose[0]
    ca_ori = camera_pose[1]
    intrinsic = camera_pose[2].reshape(3, 3)
    depth = read_image(img_path, view)
    org_size = depth.shape
    depth = cv2.resize(depth, (224, 224), interpolation=cv2.INTER_NEAREST)

    # convert it to point cloud and remove inf/nan points
    pc = Depth2PointCloud(depth, intrinsic, ca_ori, ca_loc, org_size=org_size).transpose()

    # remove the undefined and outlier points
    inf_idx = (pc != pc) | (np.abs(pc) > 100)
    pc[inf_idx] = 0.0

    # need to prune the shelf --> remove all points below a certain threshold on the z axis
    above_shelf_indexes = np.nonzero(pc[:, 2] > shelf_th)[0]
    pc = pc[above_shelf_indexes]

    return pc


def travel_image_dir(img_root):
    shape_dir = {}
    shapes = os.listdir(img_root)
    for s in shapes:
        shape_path = osp.join(img_root, s)
        if osp.isdir(shape_path):
            shape_dir[s] = shape_path
    return shape_dir


def match_contacts_with_pc(contacts, scores, pc, contact_th=0.0035, policy='soft'):
    assert policy in ['soft', 'hard'], "Must choose between hard or soft matching"
    if policy == 'soft':
        return match_contacts_with_pc_soft(contacts, scores, pc, contact_th=contact_th)
    else:
        return match_contacts_with_pc_hard(contacts, scores, pc, contact_th=contact_th)


def match_contacts_with_pc_hard(contacts, scores, pc, contact_th=0.0035):
    """
    Maps each contact pair to a point in the point cloud.
    @param contacts: the contact point pairs retrieved from the dataset [M x 3 x 2]
    @param scores: the grasp label (M x 1)
    @param pc: the point cloud [N x 3]
    @param contact_th: distance threshold for a contact point to be considered valid
    @return: the pc indexes corresponding to the closest pc point for each contact pair
    """

    pc_rescaled = pc * np.array([0.22 / 2, 0.22 / 2, 0.22])
    first_contact_pc_indexes = []
    contact_indexes = []

    for j in range(len(contacts)):
        score = scores[j]
        cont = contacts[j]
        c1, c2 = cont[0], cont[1]

        c1_rescaled = c1 * np.array([0.22 / 2, 0.22 / 2, 0.22])
        c2_rescaled = c2 * np.array([0.22 / 2, 0.22 / 2, 0.22])

        dist1 = np.sqrt(np.sum((pc_rescaled - c1_rescaled) ** 2, axis=1))
        dist2 = np.sqrt(np.sum((pc_rescaled - c2_rescaled) ** 2, axis=1))

        idx1 = np.argmin(dist1)
        d1_min = dist1[idx1]
        idx2 = np.argmin(dist2)
        d2_min = dist2[idx2]

        # add the contact pair index to the list if one of the two points is close enough to the pc
        if d1_min <= contact_th:
            contact_indexes.append((j, 0, 1))
            if idx1 not in first_contact_pc_indexes and score == 1:
                first_contact_pc_indexes.append(idx1)

        if d2_min <= contact_th:
            contact_indexes.append((j, 1, 0))
            if idx2 not in first_contact_pc_indexes and score == 1:
                first_contact_pc_indexes.append(idx2)

    first_contact_pc_indexes = np.array(first_contact_pc_indexes)
    contact_indexes = np.array(contact_indexes)

    return first_contact_pc_indexes, contact_indexes


def match_contacts_with_pc_soft(contacts, scores, pc, contact_th=0.0035):
    """
    Maps each contact pair to some points in the point cloud close enough to it.
    The contact pairs are filtered and only the ones close
    @param contacts: the contact point pairs retrieved from the dataset [M x 3 x 2]
    @param pc: the point cloud [N x 3]
    @param contact_th: distance threshold for a contact point to be considered valid
    @return:    - the pc indexes of the points close enough to a positively annotated grasp
                - for each of the selected contact pairs (index, 0/1, 0/1)
                    - index in the annotated contact pair list
                    - the pair index of the close ("first") contact point
                    - the pair index of the "second" contact point

    """

    pc_rescaled = pc * np.array([0.22 / 2, 0.22 / 2, 0.22])
    first_contact_pc_indexes = []
    contact_indexes = []
    for j in range(len(contacts)):
        cont = contacts[j]
        score = scores[j]
        c1, c2 = cont[0], cont[1]

        c1_rescaled = c1 * np.array([0.22 / 2, 0.22 / 2, 0.22])
        c2_rescaled = c2 * np.array([0.22 / 2, 0.22 / 2, 0.22])

        dist1 = np.sqrt(np.sum((pc_rescaled - c1_rescaled) ** 2, axis=1))
        dist2 = np.sqrt(np.sum((pc_rescaled - c2_rescaled) ** 2, axis=1))

        idxs1 = np.nonzero(dist1 <= contact_th)[0]
        idxs2 = np.nonzero(dist2 <= contact_th)[0]

        # add the contact pair index to the list if one of the two points is close enough to the pc
        if len(idxs1) > 0:
            contact_indexes.append((j, 0, 1))
        if len(idxs2) > 0:
            contact_indexes.append((j, 1, 0))

        for idx in idxs1:
            if idx not in first_contact_pc_indexes and score == 1:
                first_contact_pc_indexes.append(idx)

        for idx in idxs2:
            if idx not in first_contact_pc_indexes and score == 1:
                first_contact_pc_indexes.append(idx)

    first_contact_pc_indexes = np.array(first_contact_pc_indexes)
    contact_indexes = np.array(contact_indexes)

    return first_contact_pc_indexes, contact_indexes


def balance_grasp_indexes(scores, sample_num, posi_ratio):
    sel_posi_num = int(sample_num * posi_ratio)
    sel_nega_num = sample_num - sel_posi_num
    posi_idx = np.nonzero(scores.reshape(-1))[0].reshape(-1)
    nega_idx = np.nonzero((scores == 0).reshape(-1))[0].reshape(-1)

    if posi_idx.shape[0] >= sel_posi_num:
        idx = np.random.choice(posi_idx.shape[0], sel_posi_num, replace=False)
        posi_idx = posi_idx[idx]
    else:
        if posi_idx.shape[0] == 0:
            idx = np.array([], dtype=int)
        else:
            idx = np.random.choice(posi_idx.shape[0], sel_posi_num - posi_idx.shape[0])
            idx = np.concatenate([np.arange(posi_idx.shape[0]), idx], 0)
        posi_idx = posi_idx[idx]

    if nega_idx.shape[0] >= sel_nega_num:
        idx = np.random.choice(nega_idx.shape[0], sel_nega_num, replace=False)
        nega_idx = nega_idx[idx]
    else:
        if nega_idx.shape[0] == 0:
            idx = np.array([], dtype=int)
        else:
            idx = np.random.choice(nega_idx.shape[0], sel_nega_num - nega_idx.shape[0])
            idx = np.concatenate([np.arange(nega_idx.shape[0]), idx], 0)
        nega_idx = nega_idx[idx]

    selected_idxs = np.concatenate([posi_idx, nega_idx])
    np.random.shuffle(selected_idxs)
    return selected_idxs


class SampleGraspData(data.Dataset):
    def __init__(self,
                 data_root,
                 split='train',
                 view=0,
                 img_size=(224, 224), sample_ratio=1.0, sample_num=1000, tot_num=10000,
                 positive_ratio=0.5, contact_th=0.0035, matching_policy='soft'):
        """
        Class for ShapeNetSem-8 and YCB-8
        Extracts info from the grasp dataset.
        @param data_root: the root containing the grasp annotations
        @param img_size: the size of the depth image
        @param split: train or test
        @param sample_ratio: the ratio of grasp from sample_num to consider for each shape (train only)
        @param sample_num: the num of sample grasp to consider for the train of the classifier (train only)
        @param tot_num: the num of grasp to sample from the total truth (reduce computational time)
        @param positive_ratio: the positive annotated grasp ratio (train only)
        @param view: the view to consider (this parameter has no effect if split is train)
        """
        assert split in ['train', 'train_half', 'train_quarter', 'test', 'ycb8_test'], f"Unknown split {split}"
        self.split = split
        self.data_root = data_root
        self.img_size = img_size
        self.positive_ratio = positive_ratio
        self.sample_num = sample_num
        self.tot_num = tot_num
        self.matching_policy = matching_policy
        self.contact_th = contact_th
        self.view = view
        print(f"SampleGraspData - data_root: {self.data_root}, split: {self.split}")

        file_path = osp.join(self.data_root, "%s_set.csv" % self.split)
        # fix ycb_8 csv path
        if self.split == "ycb8_test":
            file_path = osp.join(self.data_root, "test_set.csv")

        assert osp.exists(file_path), f"Cannot find split file at {file_path}"
        self.shapes = []
        with open(file_path, 'r') as f:
            lines = f.readlines()
            for line in lines:
                shape = line.strip().split('.')[0]
                self.shapes.append(shape)
        assert len(self.shapes) > 0, f"Split file {file_path} is empty"
        self.pc_root = osp.join(data_root, 'point_clouds')  # in case of YCB test data
        self.img_root = osp.join(data_root, 'images')
        self.mesh_root = osp.join(data_root, "meshes")
        self.shape_dir = travel_image_dir(self.pc_root) if self.split == 'ycb8_test' else \
            travel_image_dir(self.img_root)
        self.anno_root = osp.join(data_root, 'annotations')
        self.sample_ratio = sample_ratio

    def __len__(self):
        return len(self.shapes)

    def __getitem__(self, index):
        shape = self.shapes[index]
        cam_info, contacts, angles, scores = self.read_grasps(shape)

        if self.split.startswith('train'):
            # is TRAINING
            # random view: view argument is ignored for training split
            camera_pose, view = random_camera_view(cam_info)
            pc = get_point_cloud_from_cam_info(self.shape_dir[shape], camera_pose, view)
            pc = prune_and_normalize(pc)
            contacts[:, 0, :] = prune_and_normalize(contacts[:, 0, :])
            contacts[:, 1, :] = prune_and_normalize(contacts[:, 1, :])
            first_contact_pc_indexes, contact_indexes = match_contacts_with_pc(
                contacts, scores, pc, self.contact_th, self.matching_policy)
            grasp_indexes = balance_grasp_indexes(scores[contact_indexes[:, 0]], self.sample_num, self.positive_ratio)
            return pc, first_contact_pc_indexes, contacts, angles, scores, contact_indexes, grasp_indexes, shape
        else:
            # is TESTING
            # view is fixed to self.view for reproducibility
            camera_pose = cam_info['view%d' % self.view]
            if self.split == 'test':
                # ShapeNetSem-8 test data
                pc = get_point_cloud_from_cam_info(self.shape_dir[shape], camera_pose, self.view)
            elif self.split == 'ycb8_test':
                # YCB-8 test data
                pc = np.load(osp.join(self.shape_dir[shape], f'pc{self.view}.npy'))
            else:
                raise ValueError(f"Unknown split {self.split}")

            pc = prune_and_normalize(pc)
            return pc, shape

    def read_grasps(self, shape):
        """
        Read all grasps annotations related to the specific shape
        @param shape: the shape to extract grasps of
        @return: for each of the M annotated grasp:
                    - the depth camera info (needed to build the pc)
                    - the contact pairs (c1,c2) [M x 2 x 3]
                    - the angle [M x 1]
                    - the label (0,1) [M x 1]
        """
        contacts = np.load(osp.join(self.anno_root, 'candidate', shape + '_contact.npy'))
        angles = np.load(osp.join(self.anno_root, 'candidate', shape + '_cos.npy'))
        labels = np.load(osp.join(self.anno_root, 'simulateResult', shape + '.npy'))
        assert labels.shape[0] == contacts.shape[0]

        # select grasps such that the ratio between positive and negative is self.posi_ratio
        # over sample in this phase since some of them will be filtered later
        if self.sample_num is not None:
            posi_idx = np.nonzero(labels.reshape(-1))[0].reshape(-1)
            nega_idx = np.nonzero((labels == False).reshape(-1))[0].reshape(-1)
            posi_num = posi_idx.shape[0]
            nega_num = nega_idx.shape[0]
            tot_sample_num = self.tot_num
            posi_exp = tot_sample_num // 2
            nega_exp = tot_sample_num - posi_exp
            if posi_num > posi_exp:
                posi_idx = np.random.choice(posi_idx, posi_exp, replace=False)
            else:
                posi_idx = np.concatenate((posi_idx, np.random.choice(posi_idx, posi_exp - posi_num, replace=True)),
                                          axis=None)
            if nega_num > nega_exp:
                nega_idx = np.random.choice(nega_idx, nega_exp, replace=False)
            else:
                nega_idx = np.concatenate((nega_idx, np.random.choice(nega_idx, nega_exp - nega_num, replace=True)),
                                          axis=None)

            all_idx = np.concatenate([posi_idx, nega_idx], 0)
            np.random.shuffle(all_idx)
            labels = labels[all_idx]
            contacts = contacts[all_idx]
            angles = angles[all_idx]

        cam_info_path = osp.join(self.img_root, shape, 'CameraInfo.npy')
        cam_info = read_camera_info(cam_info_path)

        assert (contacts.shape[0] == angles.shape[0] and contacts.shape[0] == angles.shape[0])
        return cam_info, contacts, angles, labels

class YCB8_test(SampleGraspData):
    """The YCB8 test data"""
    def __init__(self, opt, view=0, img_size=(224, 224), tot_num=10000, matching_policy='soft'):
        self.opt=opt
        super().__init__(
            data_root=os.path.join(opt.data_dir, "YCB-8"), 
            split="ycb8_test", 
            view=view, 
            img_size=img_size, 
            tot_num=tot_num, 
            matching_policy=matching_policy
        )


class YCB76_Data(data.Dataset):
    def __init__(self, data_root, split="test", view=0):
        """
        Extracts info from the YCB-76 test dataset - used only for simulation test, no grasp annotations available!
        @param data_root: the root dataset folder (containing test_set.csv)
        @param view: the view to consider for the depth image (test only, random at training time)
        """
        assert split == "test", f"YCB_76_Data - only test split available."
        assert 0 <= view <= 9, f"YCB_76_Data - only views 0 to 9 available."
        self.data_root = data_root
        self.split = split
        self.view = view
        print(f"YCB76_Data - data_root: {self.data_root}, split: {self.split}")
        file_path = osp.join(self.data_root, "%s_set.csv" % self.split)
        with open(file_path, 'r') as file:
            lines = file.readlines()
        self.shapes = [line.strip() for line in lines]
        self.pc_dir = osp.join(data_root, 'point_clouds')

    def __len__(self):
        return len(self.shapes)

    def __getitem__(self, index):
        shape = self.shapes[index]
        pc_fn = osp.join(self.pc_dir, shape, f'pc{self.view}.npy')
        pc = np.load(pc_fn)
        # pc = prune_and_normalize(pc)  # might crop pc's of large objects too significantly
        pc = pc / np.array([0.22 / 2, 0.22 / 2, 0.22])  # scaling
        return pc, shape
