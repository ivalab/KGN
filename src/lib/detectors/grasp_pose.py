from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import cv2
import numpy as np
from progress.bar import Bar
import time
import torch

try:
  from external.nms import soft_nms_39
except:
  print('NMS not imported! If you need it,'
        ' do \n cd $CenterNet_ROOT/src/lib/external \n make')
from models.decode import grasp_pose_decode 
from models.utils import flip_tensor, flip_lr_off, flip_lr
from utils.image import get_affine_transform
from utils.post_process import grasp_pose_post_process 
from utils.debugger import Debugger

from .base_detector import BaseDetector

class GraspPoseDetector(BaseDetector):
  def __init__(self, opt):
    super(GraspPoseDetector, self).__init__(opt)
    #self.flip_idx = opt.flip_idx

  def process(self, images, return_time=False):
    """ Run the model and decode to get the detection results

    Returns:
        output (dict):    Raw model output.
        dets (tensor, [B, K, D]): Stack of the 2d keypoint projections, width, and the relevant info including:\
            (1) scores - the center Point probability
            i.e. detections = [center_locations, kpt_locations, open_width, scores]
            B is the batch size. K is the candidate number \
            D = 2 + 2*num_kpts + 1 + 1 + ...
    """
    with torch.no_grad():
      torch.cuda.synchronize()
      output = self.model(images)[-1]
      output['hm'] = output['hm'].sigmoid_()
      #if self.opt.hm_hp and not self.opt.mse_loss:
      #  output['hm_hp'] = output['hm_hp'].sigmoid_()

      reg = output['reg'] if self.opt.reg_offset else None
      #hm_hp = output['hm_hp'] if self.opt.hm_hp else None
      #hp_offset = output['hp_offset'] if self.opt.reg_hp_offset else None
      if self.opt.kpts_refine:
        output["hm_kpts"] = output["hm_kpts"].sigmoid_()
        hm_kpts = output["hm_kpts"]
        kpts_offset = output["kpts_offset"]
      else:
        hm_kpts = None
        kpts_offset = None
      
      if self.opt.sep_scale_branch:
        scales = output["scales"]
      else:
        scales = None
      torch.cuda.synchronize()
      forward_time = time.time()
      
      if self.opt.flip_test:
        raise NotImplementedError
        output['hm'] = (output['hm'][0:1] + flip_tensor(output['hm'][1:2])) / 2
        output['wh'] = (output['wh'][0:1] + flip_tensor(output['wh'][1:2])) / 2
        output['hps'] = (output['hps'][0:1] + 
          flip_lr_off(output['hps'][1:2], self.flip_idx)) / 2
        hm_hp = (hm_hp[0:1] + flip_lr(hm_hp[1:2], self.flip_idx)) / 2 \
                if hm_hp is not None else None
        reg = reg[0:1] if reg is not None else None
        hp_offset = hp_offset[0:1] if hp_offset is not None else None
      
      dets = grasp_pose_decode(
        self.opt,
        output['hm'], output['w'], output['kpts_center_offset'], 
        scales = scales,
        reg=reg, hm_kpts=hm_kpts, kpts_offset=kpts_offset, K=self.opt.K
        )

    if return_time:
      return output, dets, forward_time
    else:
      return output, dets

  def post_process(self, dets, meta, scale=1):
    """Post process the result that recover the image coordinate and reorgize the format

    Args:
        dets (_type_): _description_
        meta (_type_): _description_
        scale (int, optional): _description_. Defaults to 1.

    Returns:
        dets: List of the dictionary of results for each class:
          [{cls_idx: cls_results}]
          where cls_results: (array (K, D)), K is the candidate number, \
            D is the necessary info dimension 12:
              2 (center coords) + 2*4 (coords for the 4 kpts) + 1 (width) + 1 (score) + 1(cls)
  
    """
    dets = dets.detach().cpu().numpy().reshape(1, -1, dets.shape[2])
    dets = grasp_pose_post_process(
      self.opt,
      dets.copy(), [meta['c']], [meta['s']],
      meta['out_height'], meta['out_width'])
    for j in range(1, self.num_classes + 1):
      if self.opt.sep_scale_branch:
        dets[0][j] = np.array(dets[0][j], dtype=np.float32).reshape(-1, 2 + 2*4+1+1+1+1)
      else:
        dets[0][j] = np.array(dets[0][j], dtype=np.float32).reshape(-1, 2 + 2*4+1+1+1)
      dets[0][j][:, :10] /= scale
    return dets[0]

  def merge_outputs(self, detections):
    """Merge the detection results for multi-scales

    Args:
        detections (List[detes]):  A list of detection result in the format of the pose_process function
          for different scales. They are already recovered to the scale 1

    Returns:
        results: same format as above
          List of the dictionary of results for each class:
            [{cls_idx: cls_results}]
            where cls_results: (array (K, D)), K is the candidate number, \
              D is the necessary info dimension 10: 2*4 (coords for the 4 kpts) + 1 (width) + 1 (score)
    """
    results = {}
    results[1] = np.concatenate(
        [detection[1] for detection in detections], axis=0).astype(np.float32)
    if self.opt.nms or len(self.opt.test_scales) > 1:
      soft_nms_39(results[1], Nt=0.5, method=2)
    results[1] = results[1].tolist()
    return results

  def debug(self, debugger, images, dets, output, scale=1):
    """
    The heatmaps for both the center and the keypoints

    Args:
      dets:     The detection results from the self.process function
      output:   The network output fed in and modified by the self.process function
    """
    dets = dets.detach().cpu().numpy().copy()
    dets[:, :, :10] *= self.opt.down_ratio
    img = images[0].detach().cpu().numpy().transpose(1, 2, 0)
    if self.opt.input_mod == "RGBD":
      img = np.clip(((
        img * self.std[:, :, :3] + self.mean[:, :, :3]) * 255.), 0, 255).astype(np.uint8)
    else:
      img = np.clip(((
        img * self.std + self.mean) * 255.), 0, 255).astype(np.uint8)

    # import matplotlib.pyplot as plt
    # plt.imshow(
    #   output['hm'][0].detach().cpu().numpy().max(axis=0, keepdims=True).transpose(1, 2, 0)
    # )
    # plt.show()
    pred = debugger.gen_colormap(output['hm'][0].detach().cpu().numpy())
    debugger.add_blend_img(img, pred, 'pred_hm')

    #if self.opt.hm_hp:
    if self.opt.kpts_refine:
      pred = debugger.gen_colormap_hp(
        output['hm_kpts'][0].detach().cpu().numpy())
      debugger.add_blend_img(img, pred, 'pred_hmkpts')
  
  def show_results(self, debugger, image, results, display=True):
    debugger.add_img(image.astype(np.uint8), img_id='grasp_pose')
    for bbox in results[1]:
      if bbox[11] > self.opt.vis_thresh:
        debugger.add_ps_grasp_kpts(bbox[2:10], img_id='grasp_pose')
    if display:
      debugger.show_all_imgs(pause=self.pause)

  def save_results(self, debugger, prefix=None):
    prefix_content = "{}_".format(prefix) if prefix is not None \
      else ''
    debugger.save_all_imgs(
        self.opt.debug_dir, prefix=prefix_content)