from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from multiprocessing import reduction
from re import S

import torch
import numpy as np

from models.losses import FocalLoss, RegL1Loss, RegLoss, RegWeightedL1Loss
from models.decode import grasp_pose_decode
from models.utils import _sigmoid, flip_tensor, flip_lr_off, flip_lr
from utils.debugger import Debugger
from utils.post_process import grasp_pose_post_process
from utils.oracle_utils import gen_oracle_map
from .base_trainer import BaseTrainer


class GraspPoseLoss_clf(torch.nn.Module):
    def __init__(self, opt):
        super(GraspPoseLoss_clf, self).__init__()
        self.opt = opt

        # grasp width
        self.crit_w = RegWeightedL1Loss() 

        # center loss
        self.crit_hm = FocalLoss()
        self.crit_reg = RegL1Loss() if opt.reg_loss == 'l1' else \
            RegLoss() if opt.reg_loss == 'sl1' else None

        self.crit_kpts_center = RegWeightedL1Loss() if not opt.dense_kpts else \
            torch.nn.L1Loss(reduction='sum')
        self.crit_hm_kpts = FocalLoss()
        
        # the scale loss
        if opt.sep_scale_branch:
            self.crit_scale = RegWeightedL1Loss()
            
        """CenterNet version
        self.crit = FocalLoss()
        self.crit_hm_hp = torch.nn.MSELoss() if opt.mse_loss else FocalLoss()
        self.crit_kp = RegWeightedL1Loss() if not opt.dense_kpts else \
            torch.nn.L1Loss(reduction='sum')
        self.crit_reg = RegL1Loss() if opt.reg_loss == 'l1' else \
            RegLoss() if opt.reg_loss == 'sl1' else None
        self.opt = opt
        """ 

    def forward(self, outputs, batch):

        opt = self.opt
        hm_loss, w_loss, off_loss = 0, 0, 0
        scale_loss = 0

        kpts_center_loss, hm_kpts_loss, kpts_offset_loss = 0, 0, 0

        for s in range(opt.num_stacks):
            output = outputs[s]
            output['hm'] = _sigmoid(output['hm'])
            if opt.kpts_refine and not opt.mse_loss:
                output['hm_kpts'] = _sigmoid(output['hm_kpts'])

            if opt.eval_oracle_hmkpts:
                output['hm_kpts'] = batch['hm_kpts']
            if opt.eval_oracle_hm:
                output['hm'] = batch['hm']
            if opt.eval_oracle_kps:
                if opt.dense_kpts:
                    output['kpts'] = batch['dense_kpts']
                else:
                    raise NotImplementedError("Haven't check what does gen_oracle_map mean")
                    output['kpts'] = torch.from_numpy(gen_oracle_map(
                        batch['kpts'].detach().cpu().numpy(),
                        batch['ind'].detach().cpu().numpy(),
                        opt.output_res, opt.output_res)).to(opt.device)
            if opt.eval_oracle_kpts_offset:
                output['kpts_offset'] = torch.from_numpy(gen_oracle_map(
                    batch['kpts_offset'].detach().cpu().numpy(),
                    batch['kpts_ind'].detach().cpu().numpy(),
                    opt.output_res, opt.output_res)).to(opt.device)
            
            # The center branch losses
            hm_loss += self.crit_hm(output['hm'], batch['hm']) / opt.num_stacks

                
            kpts_center_loss += self.crit_kpts_center(output['kpts_center_offset'], batch['kpts_center_mask'],
                                    batch['ind'], batch['kpts_center_offset']) / opt.num_stacks

            if opt.reg_offset and opt.off_weight > 0:
                off_loss += self.crit_reg(output['reg'], batch['reg_mask'],
                                          batch['ind'], batch['reg']) / opt.num_stacks
            
            # The open width prediction loss
            if opt.w_weight > 0:
                w_loss += self.crit_w(output['w'], batch['w_mask'],
                                         batch['ind'], batch['w']) / opt.num_stacks
            
            # The keypoints refinement branch loss
            if opt.kpts_refine and opt.off_weight > 0:
                kpts_offset_loss += self.crit_reg(
                    output['kpts_offset'], batch['kpts_mask'],
                    batch['kpts_ind'], batch['kpts_offset']) / opt.num_stacks
            if opt.kpts_refine and opt.hm_kpts_weight > 0:
                hm_kpts_loss += self.crit_hm_kpts(
                    output['hm_kpts'], batch['hm_kpts']) / opt.num_stacks
            
            # The scale loss
            if opt.sep_scale_branch and opt.scale_weight > 0:
                scale_loss += self.crit_scale(
                    output['scales'], batch['scales_mask'],
                    batch['ind'], batch['scales']
                ) / opt.num_stacks
                
        

        loss = opt.hm_weight * hm_loss + opt.w_weight * w_loss + \
            opt.off_weight * off_loss + opt.kpts_center_weight * kpts_center_loss + \
            opt.hm_kpts_weight * hm_kpts_loss + opt.off_weight * kpts_offset_loss + \
            opt.scale_weight * scale_loss
        
        
        loss_stats = {'loss': loss, 'hm_loss': hm_loss, 'w_loss': w_loss,
                    'kpts_center_loss': kpts_center_loss,'reg_loss(center_offset)': off_loss,
                    'hm_kpts_loss': hm_kpts_loss, 'kpts_offset_loss': kpts_offset_loss,
                    "scale_loss": scale_loss
                    }

                       
        return loss, loss_stats


class GraspPoseTrainer(BaseTrainer):
    def __init__(self, opt, model, optimizer=None):
        super(GraspPoseTrainer, self).__init__(opt, model, optimizer=optimizer)

    def _get_losses(self, opt):
        loss_states = ['loss', 'hm_loss', 'w_loss', 'kpts_center_loss',
                   'reg_loss(center_offset)']

        # The options to get the keypoint refinement losses
        if self.opt.kpts_refine:
            loss_states.append('hm_kpts_loss')
            loss_states.append('kpts_offset_loss')

        # The scale loss
        if self.opt.sep_scale_branch:
            loss_states.append("scale_loss")

        loss = GraspPoseLoss_clf(opt)
        return loss_states, loss

    def debug(self, batch, output, iter_id):
        opt = self.opt
        reg = output['reg'] if opt.reg_offset else None
        hm_kpts = output['hm_kpts'] if opt.kpts_refine else None
        kpts_offset = output['kpts_offset'] if opt.kpts_refine else None
        dets = grasp_pose_decode(
            self.opt,
            output['hm'], output['w'], output['kpts_center_offset'],
            reg=reg, hm_kpts=hm_kpts, kpts_offset=kpts_offset, 
            scales=output["scales"] if "scales" in output.keys() else None, 
            K=self.opt.K)
        dets = dets.detach().cpu().numpy().reshape(1, -1, dets.shape[2])

        dets[:, :, :10] *= opt.input_res / opt.output_res
        dets_gt = batch['meta']['gt_det'].numpy().reshape(1, -1, dets.shape[2])
        dets_gt[:, :, :10] *= opt.input_res / opt.output_res
        for i in range(1):
            debugger = Debugger(
                dataset=opt.dataset, ipynb=(opt.debug == 3), theme=opt.debugger_theme,
                kpt_type=self.opt.kpt_type
            )
            img = batch['input'][i].detach().cpu().numpy().transpose(1, 2, 0)
            img = np.clip(((
                img * opt.std + opt.mean) * 255.), 0, 255).astype(np.uint8)
            # img = img[:,:,::-1] # bgr
            img = img[:, :, :3][:, :, ::-1] # RGBD to bgr
            pred = debugger.gen_colormap(
                output['hm'][i].detach().cpu().numpy())
            gt = debugger.gen_colormap(batch['hm'][i].detach().cpu().numpy())
            debugger.add_blend_img(img, pred, 'pred_hm')
            debugger.add_blend_img(img, gt, 'gt_hm')

            debugger.add_img(img, img_id='out_pred')
            for k in range(len(dets[i])):
                if dets[i, k, 11] > opt.center_thresh:
                    debugger.add_ps_grasp_kpts(dets[i, k, 2:10], img_id='out_pred')

            debugger.add_img(img, img_id='out_gt')
            for k in range(len(dets_gt[i])):
                if dets_gt[i, k, 11] > opt.center_thresh:
                    debugger.add_ps_grasp_kpts(dets_gt[i, k, 2:10], img_id='out_gt')

            if opt.kpts_refine:
                pred = debugger.gen_colormap_hp(
                    output['hm_kpts'][i].detach().cpu().numpy())
                gt = debugger.gen_colormap_hp(
                    batch['hm_kpts'][i].detach().cpu().numpy())
                debugger.add_blend_img(img, pred, 'pred_hmkpts')
                debugger.add_blend_img(img, gt, 'gt_hmkpts')

            if opt.debug == 4:
                debugger.save_all_imgs(
                    opt.debug_dir, prefix='{}'.format(iter_id))
            else:
                debugger.show_all_imgs(pause=True)

    def save_result(self, output, batch, results):
        reg = output['reg'] if self.opt.reg_offset else None
        hm_kpts = output['hm_kpts'] if self.opt.kpts_refine else None
        kpts_offset = output['kpts_offset'] if self.opt.kpts_refine else None
        dets = grasp_pose_decode(
            output['hm'], output['w'], output['kpts_center_offset'],
            reg=reg, hm_kpts=hm_kpts, kpts_offset=kpts_offset, 
            scales=output["scales"] if "scales" in output.keys() else None, 
            K=self.opt.K)
        dets = dets.detach().cpu().numpy().reshape(1, -1, dets.shape[2])

        dets_out = grasp_pose_post_process(
            dets.copy(), batch['meta']['c'].cpu().numpy(),
            batch['meta']['s'].cpu().numpy(),
            output['hm'].shape[2], output['hm'].shape[3])


        results[batch['meta']['img_id'].cpu().numpy()[0]] = dets_out[0]
