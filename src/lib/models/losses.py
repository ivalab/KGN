# ------------------------------------------------------------------------------
# Portions of this code are from
# CornerNet (https://github.com/princeton-vl/CornerNet)
# Copyright (c) 2018, University of Michigan
# Licensed under the BSD 3-Clause License
# ------------------------------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils import _transpose_and_gather_feat, _inds_to_coords, _gather_subfeat
from utils.keypoints import get_vanishing_points

def _slow_neg_loss(pred, gt):
  '''focal loss from CornerNet'''
  pos_inds = gt.eq(1)
  neg_inds = gt.lt(1)

  neg_weights = torch.pow(1 - gt[neg_inds], 4)

  loss = 0
  pos_pred = pred[pos_inds]
  neg_pred = pred[neg_inds]

  pos_loss = torch.log(pos_pred) * torch.pow(1 - pos_pred, 2)
  neg_loss = torch.log(1 - neg_pred) * torch.pow(neg_pred, 2) * neg_weights

  num_pos  = pos_inds.float().sum()
  pos_loss = pos_loss.sum()
  neg_loss = neg_loss.sum()

  if pos_pred.nelement() == 0:
    loss = loss - neg_loss
  else:
    loss = loss - (pos_loss + neg_loss) / num_pos
  return loss


def _neg_loss(pred, gt):
  ''' Modified focal loss. Exactly the same as CornerNet.
      Runs faster and costs a little bit more memory
    Arguments:
      pred (batch x c x h x w)
      gt_regr (batch x c x h x w)
  '''
  pos_inds = gt.eq(1).float()
  neg_inds = gt.lt(1).float()

  neg_weights = torch.pow(1 - gt, 4)

  loss = 0

  pos_loss = torch.log(pred) * torch.pow(1 - pred, 2) * pos_inds
  neg_loss = torch.log(1 - pred) * torch.pow(pred, 2) * neg_weights * neg_inds

  num_pos  = pos_inds.float().sum()
  pos_loss = pos_loss.sum()
  neg_loss = neg_loss.sum()

  if num_pos == 0:
    loss = loss - neg_loss
  else:
    loss = loss - (pos_loss + neg_loss) / num_pos
  return loss

def _not_faster_neg_loss(pred, gt):
    pos_inds = gt.eq(1).float()
    neg_inds = gt.lt(1).float()    
    num_pos  = pos_inds.float().sum()
    neg_weights = torch.pow(1 - gt, 4)

    loss = 0
    trans_pred = pred * neg_inds + (1 - pred) * pos_inds
    weight = neg_weights * neg_inds + pos_inds
    all_loss = torch.log(1 - trans_pred) * torch.pow(trans_pred, 2) * weight
    all_loss = all_loss.sum()

    if num_pos > 0:
        all_loss /= num_pos
    loss -=  all_loss
    return loss

def _slow_reg_loss(regr, gt_regr, mask):
    num  = mask.float().sum()
    mask = mask.unsqueeze(2).expand_as(gt_regr)

    regr    = regr[mask]
    gt_regr = gt_regr[mask]
    
    regr_loss = nn.functional.smooth_l1_loss(regr, gt_regr, size_average=False)
    regr_loss = regr_loss / (num + 1e-4)
    return regr_loss

def _reg_loss(regr, gt_regr, mask):
  ''' L1 regression loss
    Arguments:
      regr (batch x max_objects x dim)
      gt_regr (batch x max_objects x dim)
      mask (batch x max_objects)
  '''
  num = mask.float().sum()
  mask = mask.unsqueeze(2).expand_as(gt_regr).float()

  regr = regr * mask
  gt_regr = gt_regr * mask
    
  regr_loss = nn.functional.smooth_l1_loss(regr, gt_regr, size_average=False)
  regr_loss = regr_loss / (num + 1e-4)
  return regr_loss

class FocalLoss(nn.Module):
  '''nn.Module warpper for focal loss variant designed by the CornerNet.

  The API takes in the predicted and the GT heatmap, both with the shape (b, c, h, w)
  '''
  def __init__(self):
    """
    """
    super(FocalLoss, self).__init__()
    self.neg_loss = _neg_loss

  def forward(self, out, target):
    return self.neg_loss(out, target)

class RegLoss(nn.Module):
  '''Regression loss for an output tensor
    Arguments:
      output (batch x dim x h x w)
      mask (batch x max_objects)
      ind (batch x max_objects)
      target (batch x max_objects x dim)
  '''
  def __init__(self):
    super(RegLoss, self).__init__()
  
  def forward(self, output, mask, ind, target):
    pred = _transpose_and_gather_feat(output, ind)
    loss = _reg_loss(pred, target, mask)
    return loss

class RegL1Loss(nn.Module):
  """The L1 regression loss

  Args:
      output (b, d, h, w):
      mask (b, max_objects):  The weight mask on the location
      ind (b, max_objects):   The location index - row first index
      target (b, max_objects, d):   The GT at the indexed location
  Returns:
      loss:   The weighted average (w/sum(w_i))
  """
  def __init__(self):
    """
    """
    super(RegL1Loss, self).__init__()
  
  def forward(self, output, mask, ind, target):
    pred = _transpose_and_gather_feat(output, ind)
    mask = mask.unsqueeze(2).expand_as(pred).float()
    # loss = F.l1_loss(pred * mask, target * mask, reduction='elementwise_mean')
    loss = F.l1_loss(pred * mask, target * mask, size_average=False)
    loss = loss / (mask.sum() + 1e-4)
    return loss

class NormRegL1Loss(nn.Module):
  def __init__(self):
    super(NormRegL1Loss, self).__init__()
  
  def forward(self, output, mask, ind, target):
    pred = _transpose_and_gather_feat(output, ind)
    mask = mask.unsqueeze(2).expand_as(pred).float()
    # loss = F.l1_loss(pred * mask, target * mask, reduction='elementwise_mean')
    pred = pred / (target + 1e-4)
    target = target * 0 + 1
    loss = F.l1_loss(pred * mask, target * mask, size_average=False)
    loss = loss / (mask.sum() + 1e-4)
    return loss

class RegWeightedL1Loss(nn.Module):
  """The weighted L1 regression loss

  Args:
      output (b, d, h, w):
      mask (b, max_targets, d):  The weight mask on both the location and the dimension
      ind (b, max_targets):   The location index 
      target (b, max_targets, d):   The GT at the indexed location
  Returns:
      loss:   The weighted average (w/sum(w_i))
  """
  def __init__(self):
    super(RegWeightedL1Loss, self).__init__()
  
  def forward(self, output, mask, ind, target):
    pred = _transpose_and_gather_feat(output, ind)
    mask = mask.float()
    # loss = F.l1_loss(pred * mask, target * mask, reduction='elementwise_mean')
    loss = F.l1_loss(pred * mask, target * mask, size_average=False)
    loss = loss / (mask.sum() + 1e-4)
    return loss

class L1Loss(nn.Module):
  def __init__(self):
    super(L1Loss, self).__init__()
  
  def forward(self, output, mask, ind, target):
    pred = _transpose_and_gather_feat(output, ind)
    mask = mask.unsqueeze(2).expand_as(pred).float()
    loss = F.l1_loss(pred * mask, target * mask, reduction='elementwise_mean')
    return loss
  
def _cos_sim(coords):
  """Given the stacked coordinates of 4 points,
  calculate the cosine similarity between the pt0-pt1 and the pt2-pt3

  Args:
    coords (N, 8)

  Returns:
    cos_sim (N, )
  """
  pts0 = coords[:, :2]
  pts1 = coords[:, 2:4]
  pts2 = coords[:, 4:6]
  pts3 = coords[:, 6:]

  cos_sim = F.cosine_similarity(pts0 - pts1, pts2 - pts3)
  return cos_sim

class VPtsLoss_reg(nn.Module):
  """The vanishing point loss for the offset regression mode
  for the box type keypoints

  Args:
    output_offsets (b, 8, h, w):                    The 4 center-kpt offsets for each pixel
    ind (b, max_targets):                           The index indicating the target locations for the GT targets
    vpts (b, max_targets * num_vpts, 2)             The GT vanishing point coordinates 
    vpts_fin_mask (b, max_targets * num_vpts, 2)    The mask indicating where the GT vpts are finite
    vpts_inf_mask (b, max_targets * num_vpts, 2)    The mask indicating where the GT vpts are infinite 
  """

  def __init__(self) -> None:
    super(VPtsLoss_reg, self).__init__()
  
  #def forward(self, output_offsets, mask, ind, target_offsets):
  def forward(self, output_offsets, ind, vpts, vpts_fin_mask, vpts_inf_mask):
    # the shapes
    b, _, h, w = output_offsets.shape
    _, max_targets = ind.shape


    #### get the vanishing points
    # predicted center-kpts offset. (b, max_targets, 8)
    pred_offsets = _transpose_and_gather_feat(output_offsets, ind)

    # GT center coordinates (b, max_targets, 2)
    centers = _inds_to_coords(ind, w)

    # the predicted keypoints (b, max_targets, 8)
    pred_kpts = pred_offsets + centers.repeat(1, 1, 4)
    
    # the predicted vpts (b, max_targets, 4)
    pred_vpts, pred_mask, pred_corners_ordered = get_vanishing_points(pred_kpts.view(-1, 8), return_corners_ordered=True, mode="torch")
    pred_vpts, pred_mask = pred_vpts.view(b, max_targets, 4), pred_mask.view(b, max_targets, 4)

    # the GT vpts (b, max_targets, 4)
    gt_vpts = vpts.view(b, max_targets, 4)

    # The F1 loss for the finite vanishing point. 
    mask_fin = vpts_fin_mask.view(b, max_targets, 4).float()
    loss_finite = F.l1_loss(pred_vpts * mask_fin, gt_vpts * mask_fin, size_average=False)
    loss_finite = loss_finite / (mask_fin.sum() + 1e-4)

    # The cosine similarity loss for the infinite GT vpts case.
    # For now ignore the infinite pred vpts + finite gt vpts case, assuming the pred vpts won't be infinite in the next iteration
    corner_set1, corner_set2 = pred_corners_ordered[:, :8], pred_corners_ordered[:, 8:]
    loss_inf_1 = - _cos_sim(corner_set1)
    loss_inf_2 = - _cos_sim(corner_set2)
    vpts_inf_mask = vpts_inf_mask.view(b * max_targets, 4)
    mask_inf_1, mask_inf_2 = vpts_inf_mask[:, 0].float(), vpts_inf_mask[:, 2].float() # both (N, )
    loss_inf = torch.sum(loss_inf_1 * mask_inf_1 + loss_inf_2 * mask_inf_2) / (mask_inf_1.sum() + mask_inf_2.sum() + 1e-4)

    # the vpt loss together
    # loss = loss_inf + loss_finite
    loss = loss_finite 

    return loss
  


class VPtsLoss_clf(nn.Module):
  """The vanishing point loss for the offset regression mode
  for the box type keypoints


  Args:
    output_offsets (b, 8 * clf_num, h, w):                    The 4 center-kpt offsets for each pixel
    ori_clses (b, max_targets)                                The GT class labels for each target
    ind (b, max_targets):                                     The index indicating the target locations for the GT targets
    vpts (b, max_targets * num_vpts, 2)                       The GT vanishing point coordinates 
    vpts_fin_mask (b, max_targets * num_vpts, 2)              The mask indicating where the GT vpts are finite
    vpts_inf_mask (b, max_targets * num_vpts, 2)              The mask indicating where the GT vpts are infinite 
  """
  def __init__(self) -> None:
    super(VPtsLoss_clf, self).__init__()
  
  def forward(self, output_offsets, ori_clses, ind, vpts, vpts_fin_mask, vpts_inf_mask):
    # the shapes
    b, _, h, w = output_offsets.shape
    _, max_targets = ind.shape


    #### get the vanishing points
    # predicted center-kpts offset. (b, max_targets, 8 * clf_num)
    pred_offsets = _transpose_and_gather_feat(output_offsets, ind)

    # NOTE: here is the only difference. It needs to use the orientation class label to collect the correct offsets
    # The resulting offsets are again (b, max_targets, 8)
    start_ind = ori_clses * 8
    pred_offsets = _gather_subfeat(pred_offsets, start_ind, c=8)

    # GT center coordinates (b, max_targets, 2)
    centers = _inds_to_coords(ind, w)

    # the predicted keypoints (b, max_targets, 8)
    pred_kpts = pred_offsets + centers.repeat(1, 1, 4)
    
    # the predicted vpts (b, max_targets, 4)
    pred_vpts, pred_mask, pred_corners_ordered = get_vanishing_points(pred_kpts.view(-1, 8), return_corners_ordered=True, mode="torch")
    pred_vpts, pred_mask = pred_vpts.view(b, max_targets, 4), pred_mask.view(b, max_targets, 4)

    # the GT vpts (b, max_targets, 4)
    gt_vpts = vpts.view(b, max_targets, 4)

    # The F1 loss for the finite vanishing point. 
    mask_fin = vpts_fin_mask.view(b, max_targets, 4).float()
    loss_finite = F.l1_loss(pred_vpts * mask_fin, gt_vpts * mask_fin, size_average=False)
    loss_finite = loss_finite / (mask_fin.sum() + 1e-4)

    # The cosine similarity loss for the infinite GT vpts case.
    # For now ignore the infinite pred vpts + finite gt vpts case, assuming the pred vpts won't be infinite in the next iteration
    corner_set1, corner_set2 = pred_corners_ordered[:, :8], pred_corners_ordered[:, 8:]
    loss_inf_1 = - _cos_sim(corner_set1)
    loss_inf_2 = - _cos_sim(corner_set2)
    vpts_inf_mask = vpts_inf_mask.view(b * max_targets, 4)
    mask_inf_1, mask_inf_2 = vpts_inf_mask[:, 0].float(), vpts_inf_mask[:, 2].float() # both (N, )
    loss_inf = torch.sum(loss_inf_1 * mask_inf_1 + loss_inf_2 * mask_inf_2) / (mask_inf_1.sum() + mask_inf_2.sum() + 1e-4)

    # the vpt loss together
    # loss = loss_inf + loss_finite
    loss = loss_finite

    return loss

class VPtsLoss_kpts_offset(nn.Module):
  """The vanishing point loss for the keypoint refinement branch offset

  Args: 
    output_offsets (b, 2, h, w)
    kpts_ind (b, max_grasps * 4):                   indicating the grouping of the kpts inds for each grasp
    vpts (b, max_targets * num_vpts, 2)             The GT vanishing point coordinates 
    vpts_fin_mask (b, max_targets * num_vpts, 2)    The mask indicating where the GT vpts are finite
    vpts_inf_mask (b, max_targets * num_vpts, 2)    The mask indicating where the GT vpts are infinite 
  """
  def __init__(self) -> None:
    super().__init__()

  #def forward(self, output_offsets, mask, kpts_ind, gt_offsets):
  def forward(self, output_offsets, kpts_ind, vpts, vpts_fin_mask, vpts_inf_mask):
    # shapes
    b, _, h, w = output_offsets.shape
    _, max_targets, _ = vpts.shape
    max_targets = int(max_targets / 2)

    #### The predicted kpts subpixel-offsets (b, max_targets, 8)
    pred_offsets = _transpose_and_gather_feat(output_offsets, kpts_ind).view(b, max_targets, 8)

    #### get the vanishing points
    # kpts center coordinates (b, max_targets, 8)
    centers = _inds_to_coords(kpts_ind, w).view(b, max_targets, 4*2)

    # the predicted and the GT keypoints (b, max_targets, 8)
    pred_kpts = pred_offsets + centers
    
    # the predicted and the GT vanishing points (b, max_targets, 4)
    pred_vpts, pred_mask, pred_corners_ordered = \
      get_vanishing_points(pred_kpts.view(-1, 8), return_corners_ordered=True, mode="torch")
    pred_vpts, pred_mask = pred_vpts.view(b, max_targets, 4), pred_mask.view(b, max_targets, 4)

    # the GT vpts (b, max_targets, 4)
    gt_vpts = vpts.view(b, max_targets, 4)

    # for debug
    # print("\n\n From the offset loss function, the vpts info of the first grasp:")
    # print("The GT vpts:")
    # print(gt_vpts[0, 0, :])
    # print("The pred kpts:")
    # print(pred_kpts[0, 0, :])
    # print("The pred vpts:")
    # print(pred_vpts[0, 0, :])
    # exit()

    # The F1 loss for the finite vanishing point. 
    mask_fin = vpts_fin_mask.view(b, max_targets, 4).float()
    loss_finite = F.l1_loss(pred_vpts * mask_fin, gt_vpts * mask_fin, size_average=False)
    loss_finite = loss_finite / (mask_fin.sum() + 1e-4)

    # The cosine similarity loss for the infinite GT vpts case. The shapes (b*max_targets, D)
    # For now ignore the infinite pred vpts + finite gt vpts case, assuming the pred vpts won't be infinite in the next iteration
    corner_set1, corner_set2 = pred_corners_ordered[:, :8], pred_corners_ordered[:, 8:]
    loss_inf_1 = - _cos_sim(corner_set1)
    loss_inf_2 = - _cos_sim(corner_set2)
    vpts_inf_mask = vpts_inf_mask.view(b * max_targets, 4)
    mask_inf_1, mask_inf_2 = vpts_inf_mask[:, 0].float(), vpts_inf_mask[:, 2].float() # both (N, )
    loss_inf = torch.sum(loss_inf_1 * mask_inf_1 + loss_inf_2 * mask_inf_2) / (mask_inf_1.sum() + mask_inf_2.sum() + 1e-4)

    # the vpt loss together
    # loss = loss_inf + loss_finite
    loss = loss_finite

    return loss

class BinRotLoss(nn.Module):
  def __init__(self):
    super(BinRotLoss, self).__init__()
  
  def forward(self, output, mask, ind, rotbin, rotres):
    pred = _transpose_and_gather_feat(output, ind)
    loss = compute_rot_loss(pred, rotbin, rotres, mask)
    return loss

def compute_res_loss(output, target):
    return F.smooth_l1_loss(output, target, reduction='elementwise_mean')

# TODO: weight
def compute_bin_loss(output, target, mask):
    mask = mask.expand_as(output)
    output = output * mask.float()
    return F.cross_entropy(output, target, reduction='elementwise_mean')

def compute_rot_loss(output, target_bin, target_res, mask):
    # output: (B, 128, 8) [bin1_cls[0], bin1_cls[1], bin1_sin, bin1_cos, 
    #                 bin2_cls[0], bin2_cls[1], bin2_sin, bin2_cos]
    # target_bin: (B, 128, 2) [bin1_cls, bin2_cls]
    # target_res: (B, 128, 2) [bin1_res, bin2_res]
    # mask: (B, 128, 1)
    # import pdb; pdb.set_trace()
    output = output.view(-1, 8)
    target_bin = target_bin.view(-1, 2)
    target_res = target_res.view(-1, 2)
    mask = mask.view(-1, 1)
    loss_bin1 = compute_bin_loss(output[:, 0:2], target_bin[:, 0], mask)
    loss_bin2 = compute_bin_loss(output[:, 4:6], target_bin[:, 1], mask)
    loss_res = torch.zeros_like(loss_bin1)
    if target_bin[:, 0].nonzero().shape[0] > 0:
        idx1 = target_bin[:, 0].nonzero()[:, 0]
        valid_output1 = torch.index_select(output, 0, idx1.long())
        valid_target_res1 = torch.index_select(target_res, 0, idx1.long())
        loss_sin1 = compute_res_loss(
          valid_output1[:, 2], torch.sin(valid_target_res1[:, 0]))
        loss_cos1 = compute_res_loss(
          valid_output1[:, 3], torch.cos(valid_target_res1[:, 0]))
        loss_res += loss_sin1 + loss_cos1
    if target_bin[:, 1].nonzero().shape[0] > 0:
        idx2 = target_bin[:, 1].nonzero()[:, 0]
        valid_output2 = torch.index_select(output, 0, idx2.long())
        valid_target_res2 = torch.index_select(target_res, 0, idx2.long())
        loss_sin2 = compute_res_loss(
          valid_output2[:, 6], torch.sin(valid_target_res2[:, 1]))
        loss_cos2 = compute_res_loss(
          valid_output2[:, 7], torch.cos(valid_target_res2[:, 1]))
        loss_res += loss_sin2 + loss_cos2
    return loss_bin1 + loss_bin2 + loss_res
