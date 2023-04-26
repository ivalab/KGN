from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from .image import transform_preds

from utils.keypoints import get_ori_cls, ori_cls_2_angle

def grasp_pose_post_process(opt, dets, c, s, h, w):
  """Post process the results to get the original-size image coords

  Args:
      dets (tensor, [B, K, D]): The network detection results. 
              detections (tensor, [B, K, D]): Stack of the 2d keypoint projections, width, and the relevant info including:\
            (1) scores - the center Point probability
            i.e. detections = [center_locations, kpt_locations, open_width, scores, ori_cls]
            B is the batch size. K is the candidate number \
            D = 2 + 2*num_kpts + 1 + 1 + ... 
      c (_type_): _description_
      s (_type_): _description_
      h (_type_): _description_
      w (_type_): _description_

  Returns:
      ret :  {1: preds}, where preds is: \
        preds (array, (K, D)), where K is the top-K results. D is the dimension number for describing the results, \
          including: center_coords, kpts_coords, open width, the scores.
          Hence: D = 2 + 2*kpts_num + 1 + 1
  """
  ret = []
  for i in range(dets.shape[0]):
    dets_filtered = filter_dets(opt, dets[i])
    centers = transform_preds(dets_filtered[:, :2].reshape(-1, 2), c[i], s[i], (w, h))
    kpts = transform_preds(dets_filtered[:, 2:10].reshape(-1, 2), c[i], s[i], (w, h))
    top_preds = np.concatenate(
      [centers.reshape(-1, 2), kpts.reshape(-1, 8),
        dets_filtered[:, 10:]], axis=1).astype(np.float32).tolist()
    ret.append({np.ones(1, dtype=np.int32)[0]: top_preds})
  return ret

def filter_dets(opt, dets):
  """_summary_

  Args:
      opt (_type_): _description_
      dets (K, D, np.array): _description_

  Returns:
      dets (K', D): The filtered results. K' <= K
  """
  K, D = dets.shape
  # raise NotImplementedError("Haven't tested the orientation-based filtering")
  ori_cls_pred = dets[:, 12]
  lr_kpts = dets[:, 2 : 10].reshape((K, 4, 2))
  ori_cls_kpts = get_ori_cls(lr_kpts, range_mode=0, total_cls_num=opt.ori_num)
  mask = (ori_cls_kpts == ori_cls_pred)
  dets = dets[mask, :].reshape(-1, D)

  return dets 
