from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from re import S

import torch
import torch.nn as nn
from .utils import _gather_feat, _transpose_and_gather_feat


def _nms(heat, kernel=3):
    pad = (kernel - 1) // 2

    hmax = nn.functional.max_pool2d(
        heat, (kernel, kernel), stride=1, padding=pad)
    keep = (hmax == heat).float()
    return heat * keep


def _left_aggregate(heat):
    '''
        heat: batchsize x channels x h x w
    '''
    shape = heat.shape
    heat = heat.reshape(-1, heat.shape[3])
    heat = heat.transpose(1, 0).contiguous()
    ret = heat.clone()
    for i in range(1, heat.shape[0]):
        inds = (heat[i] >= heat[i - 1])
        ret[i] += ret[i - 1] * inds.float()
    return (ret - heat).transpose(1, 0).reshape(shape)


def _right_aggregate(heat):
    '''
        heat: batchsize x channels x h x w
    '''
    shape = heat.shape
    heat = heat.reshape(-1, heat.shape[3])
    heat = heat.transpose(1, 0).contiguous()
    ret = heat.clone()
    for i in range(heat.shape[0] - 2, -1, -1):
        inds = (heat[i] >= heat[i + 1])
        ret[i] += ret[i + 1] * inds.float()
    return (ret - heat).transpose(1, 0).reshape(shape)


def _top_aggregate(heat):
    '''
        heat: batchsize x channels x h x w
    '''
    heat = heat.transpose(3, 2)
    shape = heat.shape
    heat = heat.reshape(-1, heat.shape[3])
    heat = heat.transpose(1, 0).contiguous()
    ret = heat.clone()
    for i in range(1, heat.shape[0]):
        inds = (heat[i] >= heat[i - 1])
        ret[i] += ret[i - 1] * inds.float()
    return (ret - heat).transpose(1, 0).reshape(shape).transpose(3, 2)


def _bottom_aggregate(heat):
    '''
        heat: batchsize x channels x h x w
    '''
    heat = heat.transpose(3, 2)
    shape = heat.shape
    heat = heat.reshape(-1, heat.shape[3])
    heat = heat.transpose(1, 0).contiguous()
    ret = heat.clone()
    for i in range(heat.shape[0] - 2, -1, -1):
        inds = (heat[i] >= heat[i + 1])
        ret[i] += ret[i + 1] * inds.float()
    return (ret - heat).transpose(1, 0).reshape(shape).transpose(3, 2)


def _h_aggregate(heat, aggr_weight=0.1):
    return aggr_weight * _left_aggregate(heat) + \
        aggr_weight * _right_aggregate(heat) + heat


def _v_aggregate(heat, aggr_weight=0.1):
    return aggr_weight * _top_aggregate(heat) + \
        aggr_weight * _bottom_aggregate(heat) + heat


'''
# Slow for large number of categories
def _topk(scores, K=40):
    batch, cat, height, width = scores.size()
    topk_scores, topk_inds = torch.topk(scores.view(batch, -1), K)

    topk_clses = (topk_inds / (height * width)).int()

    topk_inds = topk_inds % (height * width)
    topk_ys   = (topk_inds / width).int().float()
    topk_xs   = (topk_inds % width).int().float()
    return topk_scores, topk_inds, topk_clses, topk_ys, topk_xs
'''


def _topk_channel(scores, K=40):
    """topK for each channel

    Args:
        scores (b, c, h, w):
        K (int, optional): . Defaults to 40.

    Returns:
        topk_scores, topk_inds, topk_ys, top_xs: (b, c, K)
    """
    batch, cat, height, width = scores.size()

    # (b, c, K) for both
    topk_scores, topk_inds = torch.topk(scores.view(batch, cat, -1), K)

    topk_inds = topk_inds % (height * width)
    topk_ys = (topk_inds / width).int().float()
    topk_xs = (topk_inds % width).int().float()

    return topk_scores, topk_inds, topk_ys, topk_xs


def _topk(scores, K=40):
    """Get the topk for each batch along all the channel, height, and width dimension. 

    Args:
        scores (b, c, h, w):
        K (int, optional): The top-k number K. Defaults to 40.

    Returns:
        topk_score: (b, K)   
        topk_inds: (b, K)       The vectorized top k index in the location (heigh, width) map. Following the pytorch rule the lowest dimension (width index) fill in first
        topk_clses: (b, K)      The class index (along 2nd dimension) for the top-K data.
        topk_ys: (b, K)         Also the location index. height index
        topk_xs: (b, K)         Also the location index. Width index
    """
    batch, cat, height, width = scores.size()

    # (b, c, K)
    topk_scores, topk_inds = torch.topk(scores.view(batch, cat, -1), K)

    topk_inds = topk_inds % (height * width)
    topk_ys = (topk_inds / width).int().float()
    topk_xs = (topk_inds % width).int().float()

    topk_score, topk_ind = torch.topk(topk_scores.view(batch, -1), K)
    topk_clses = (topk_ind / K).int()
    topk_inds = _gather_feat(
        topk_inds.view(batch, -1, 1), topk_ind).view(batch, K)
    topk_ys = _gather_feat(topk_ys.view(batch, -1, 1), topk_ind).view(batch, K)
    topk_xs = _gather_feat(topk_xs.view(batch, -1, 1), topk_ind).view(batch, K)

    return topk_score, topk_inds, topk_clses, topk_ys, topk_xs

def grasp_pose_decode(
        opt,
        heat, w, kps, reg=None, hm_kpts=None, kpts_offset=None, scales=None, K=100):
    """Decode and reorganize the network output.
    It will produce the following for the top candidates:
    (1) 2d keypoint projections indicating the grasp pose
    (2) The open width

    Args:
        heat (_type_): _description_
        w (_type_): _description_
        kps (_type_): The keypoint-center offsets
        reg (_type_, optional): _description_. Defaults to None.
        hm_kpts (_type_, optional): _description_. Defaults to None.
        kpts_offset (_type_, optional): _description_. Defaults to None.
        K (int, optional): _description_. Defaults to 100.

    Returns:
        detections (tensor, [B, K', D]): Stack of the 2d keypoint projections, width, and the relevant info including:\
            (1) scores - the center Point probability
            i.e. detections = [center_location, kpt_locations, open_width, scores, clses, (grasp_scale)]
            B is the batch size. K is the candidate number \
            D = 2 + 2*num_kpts + 1 + 1 + 1 + (1)

            NOTE: K' might be lower than the K, if some candidates are filtered out.
    """

    # parse the batch number, keypoint number and the orientation class number
    batch, cls_num, height, width = heat.size()
    num_kpts = kps.shape[1] // (2 * cls_num)

    centers, kps, w, scores, clses, scls = _center_branch_decode_ori(opt, heat, w, kps, reg, K=K, scales=scales)
    
    # =========== The keypoint refinement branch ========================
    # NOTE: Currently it is done by select the topk kpts per channel and filter with a  threshold,
    if hm_kpts is not None:
        # Try to add back the nms for the kpts
        if not opt.no_nms:
            hm_kpts = _nms(hm_kpts)

        # hm_kpts = _nms(hm_kpts)

        thresh = opt.kpts_hm_thresh
        kps = kps.view(batch, K, num_kpts, 2).permute(
            0, 2, 1, 3).contiguous()  # b x C x K x 2
        reg_kps = kps.unsqueeze(3).expand(batch, num_kpts, K, K, 2)
        hm_kpts_score, hm_inds, hm_ys, hm_xs = _topk_channel(
            hm_kpts, K=K)  # b x C x K
        if kpts_offset is not None:
            kpts_offset = _transpose_and_gather_feat(
                kpts_offset, hm_inds.view(batch, -1))
            kpts_offset = kpts_offset.view(batch, num_kpts, K, 2)
            hm_xs = hm_xs + kpts_offset[:, :, :, 0]
            hm_ys = hm_ys + kpts_offset[:, :, :, 1]
        else:
            hm_xs = hm_xs + 0.5
            hm_ys = hm_ys + 0.5

        # the mask of big-enough kpt score. SEt low score to -1, and ridiculously far locations
        # so that it won't be selected when assigning the keypoints
        mask = (hm_kpts_score > thresh).float()
        hm_kpts_score = (1 - mask) * -1 + mask * hm_kpts_score  # set the scores lower than the threshold to -1
        hm_ys = (1 - mask) * (-10000) + mask * hm_ys        # set the locations of the low scores to a ridiculously negative value
        hm_xs = (1 - mask) * (-10000) + mask * hm_xs        # set the locations of the low scores to a ridiculously negative value
        # the keypoint locations from its own branch
        hm_kps = torch.stack([hm_xs, hm_ys], dim=-1).unsqueeze(
            2).expand(batch, num_kpts, K, K, 2)
        dist = (((reg_kps - hm_kps) ** 2).sum(dim=4) ** 0.5)    # b x num_kpts x K x K, reg_kpts - hm_kpts
        min_dist, min_ind = dist.min(dim=3)  # b x num_kpts x K(reg_kpts)
        hm_kpts_score = hm_kpts_score.gather(2, min_ind).unsqueeze(-1)  # b x num_kpts x K x 1
        min_dist = min_dist.unsqueeze(-1)
        min_ind = min_ind.view(batch, num_kpts, K, 1, 1).expand(
            batch, num_kpts, K, 1, 2)
        hm_kps = hm_kps.gather(3, min_ind)
        hm_kps = hm_kps.view(batch, num_kpts, K, 2)
        
        # the condition for omitting the refinement: 
        # keypoint heatmap score lower than the threshold
        mask = hm_kpts_score < thresh
        mask = (mask > 0).float().expand(batch, num_kpts, K, 2)

        # if undesired (mask=1), use the kps from the center. Else, use the refined kpt location
        kps = (1 - mask) * hm_kps + mask * kps
        kps = kps.permute(0, 2, 1, 3).contiguous().view(
            batch, K, num_kpts * 2)
    
    # collect
    if opt.sep_scale_branch:
        detections = torch.cat([centers, kps, w, scores, clses, scls], dim=2)
    else:
        detections = torch.cat([centers, kps, w, scores, clses], dim=2)

    return detections

def _center_branch_decode_ori(opt, heat, w, kps, reg, scales=None, K=100):
    """Get the top-K candidate keypoint sets from the center branch

    Args:
        heat (b, cls_ori, h, w): Center heatmap. 
        w (b, cls_ori, h, w): The open width 
        kps (b, 2*num_kpts*cls_ori, h, w): The keypoint center offset 
        reg (b, 2, h, w): The center subpixel offset 
        scales (b, cls_ori, h, w):  The predicted scale map
        K (int): top-k number

    Returns:
        centers (b, K, 2)
        kps (b, K, num_kpts*2)
        w (b, K, 1)
        scores (b, K, 1)
        clses (b, K, 1). Should be all zero here
        scales (b, K, 1)
    """
    batch, cls_num, height, width = heat.size()
    num_kpts = kps.shape[1] // (2 * cls_num)

    # perform nms on heatmaps
    if not opt.no_nms:
        heat = _nms(heat)

    # the top-k center point detection locations and orientation classes, (b, K)
    scores, inds, clses, ys, xs = _topk(heat, K=K)

    # get scales at the center points of corresponding orientation class
    if scales is not None:
        # import pdb; pdb.set_trace()
        scales = _transpose_and_gather_feat(scales, inds)
        scales = _gather_feat(
            scales.view(batch * K, cls_num, 1),
            clses.view(batch * K, 1).long()
        )
        scales = scales.view(batch, K, 1)
    else:
        scales = None

    # the keypoint locations from the center point for the orientation class
    kps = _transpose_and_gather_feat(kps, inds) # (b, k, 2*num_kpts*ori_cls)
    kps = kps.view(batch * K, cls_num, num_kpts * 2)
    kps = _gather_feat(kps, clses.view(batch * K, 1).long())    # (b*k, 1, 2*num_kpts)
    kps = kps.view(batch, K, num_kpts * 2)


    kps[..., ::2] += xs.view(batch, K, 1).expand(batch, K, num_kpts)
    kps[..., 1::2] += ys.view(batch, K, 1).expand(batch, K, num_kpts)

    # get the open-width at the center point for the orientation class
    w = _transpose_and_gather_feat(w, inds) #(b, K, ori_cls)
    w = _gather_feat(
        w.view(batch * K, cls_num, 1),
        clses.view(batch * K, 1).long()
    )
    w = w.view(batch, K, 1)

    # get the fine-grained center point location
    if reg is not None:
        reg = _transpose_and_gather_feat(reg, inds)
        reg = reg.view(batch, K, 2)
        xs = xs.view(batch, K, 1) + reg[:, :, 0:1]
        ys = ys.view(batch, K, 1) + reg[:, :, 1:2]
    else:
        xs = xs.view(batch, K, 1) + 0.5
        ys = ys.view(batch, K, 1) + 0.5
    centers = torch.cat([xs, ys], dim=-1)


    clses = clses.view(batch, K, 1).float()
    scores = scores.view(batch, K, 1)

    return centers, kps, w, scores, clses, scales
