# -*- coding: utf-8 -*-

# Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG) is
# holder of all proprietary rights on this computer program.
# You can only use this computer program if you have closed
# a license agreement with MPG or you get the right to use the computer
# program from someone who is authorized to grant you that right.
# Any use of the computer program without a valid license is prohibited and
# liable to prosecution.
#
# Copyright©2019 Max-Planck-Gesellschaft zur Förderung
# der Wissenschaften e.V. (MPG). acting on behalf of its Max Planck Institute
# for Intelligent Systems. All rights reserved.
#
# Contact: ps-license@tuebingen.mpg.de

import torch
import torch.nn as nn
import numpy as np
from lib.utils.geometry import batch_rodrigues
import os.path as osp
from lib.utils.eval_utils import compute_accel, compute_error_accel, batch_compute_similarity_transform_torch, compute_error_verts, compute_errors, plot_accel


def perm_index_reverse(indices):
    indices_reverse = np.copy(indices)

    for i, j in enumerate(indices):
        indices_reverse[j] = i

    return indices_reverse


class TCMRLoss(nn.Module):
    def __init__(
            self,
            e_loss_weight=60.,
            e_3d_loss_weight=30.,
            e_pose_loss_weight=1.,
            e_shape_loss_weight=0.001,
            d_motion_loss_weight=1.,
            device='cuda',
    ):
        super(TCMRLoss, self).__init__()
        self.e_loss_weight = e_loss_weight
        self.e_3d_loss_weight = e_3d_loss_weight
        self.e_pose_loss_weight = e_pose_loss_weight
        self.e_shape_loss_weight = e_shape_loss_weight
        self.d_motion_loss_weight = d_motion_loss_weight

        self.device = device
        self.criterion_shape = nn.L1Loss().to(self.device)
        self.criterion_keypoints = nn.MSELoss(reduction='none').to(self.device)
        self.criterion_regr = nn.MSELoss().to(self.device)
        # self.criterion_accel = nn.MSELoss().to(self.device)
        # self.criterion_accel = nn.L1Loss().to(self.device)
        self.criterion_attention = nn.CrossEntropyLoss()

        self.enc_loss = batch_encoder_disc_l2_loss
        self.dec_loss = batch_adv_disc_l2_loss

    def forward(
            self,
            generator_outputs,
            data_2d,
            data_3d,
            scores,
            data_body_mosh=None,
            data_motion_mosh=None,
            body_discriminator=None,
            motion_discriminator=None,
    ):
        # to reduce time dimension
        reduce = lambda x: x.contiguous().view((x.shape[0] * x.shape[1],) + x.shape[2:])
        # flatten for weight vectors
        flatten = lambda x: x.reshape(-1)
        # accumulate all predicted thetas from IEF
        accumulate_thetas = lambda x: torch.cat([output['theta'] for output in x],0)

        # if data_2d:
        #     sample_2d_count = data_2d['kp_2d'].shape[0]
        #     real_2d = torch.cat((data_2d['kp_2d'], data_3d['kp_2d']), 0)
        # else:
        #     sample_2d_count = 0
        #     real_2d = data_3d['kp_2d']

        # real_2d = reduce(real_2d)
        # real_3d = reduce(data_3d['kp_3d'])
        # # data_3d_theta = reduce(data_3d['theta'])
        # #
        # # w_3d = data_3d['w_3d'].type(torch.bool)
        # # w_smpl = data_3d['w_smpl'].type(torch.bool)
        #
        # total_predict_thetas = accumulate_thetas(generator_outputs)
        #
        real_3d = reduce(data_3d['kp_3d'])
        preds = generator_outputs[-1]
        # pred_j3d = preds['kp_3d'][sample_2d_count:]
        # pred_theta = preds['theta'][sample_2d_count:]
        #
        # theta_size = pred_theta.shape[:2]
        #
        # pred_theta = reduce(pred_theta)
        # pred_j2d = reduce(preds['kp_2d'])
        # print('pred_j3d',preds['kp_3d'].shape)
        # pred_j3d torch.Size([32, 3, 49, 3])
        pred_j3d = reduce(preds['kp_3d'])
        #
        # w_3d = flatten(w_3d)
        # w_smpl = flatten(w_smpl)
        #
        # pred_theta = pred_theta[w_smpl]
        # pred_j3d = pred_j3d[w_3d]
        # data_3d_theta = data_3d_theta[w_smpl]
        # real_3d = real_3d[w_3d]

        # Generator Loss
        # loss_kp_2d = self.keypoint_loss(pred_j2d, real_2d, openpose_weight=1., gt_weight=1.) * self.e_loss_weight
        # print('pred_j3d',pred_j3d.shape)
        # print('real_3d',real_3d.shape)
        # pred_j3d torch.Size([96, 49, 3])
        loss_kp_3d = self.keypoint_3d_loss(pred_j3d, real_3d)
        loss_kp_3d = loss_kp_3d * self.e_3d_loss_weight
        # loss_accel_3d = self.accel_3d_loss(pred_accel, real_accel) * 100 #self.e_3d_loss_weight
        # loss_attention = self.attetion_loss(pred_scores)
        #
        # real_shape, pred_shape = data_3d_theta[:, 75:], pred_theta[:, 75:]
        # real_pose, pred_pose = data_3d_theta[:, 3:75], pred_theta[:, 3:75]

        loss_dict = {
            'loss_kp_3d': loss_kp_3d,
        }
        # if pred_theta.shape[0] > 0:
        #     loss_pose, loss_shape = self.smpl_losses(pred_pose, pred_shape, real_pose, real_shape)
        #     loss_shape = loss_shape * self.e_shape_loss_weight
        #     loss_pose = loss_pose * self.e_pose_loss_weight
        #     loss_dict['loss_shape'] = loss_shape
        #     loss_dict['loss_pose'] = loss_pose

        gen_loss = torch.stack(list(loss_dict.values())).sum()

        return gen_loss, loss_dict

        # exclude motion discriminator
        # Motion Discriminator Loss
        # end_idx = 75
        # start_idx = 6
        # pred_motion = total_predict_thetas
        # e_motion_disc_loss = self.enc_loss(motion_discriminator(pred_motion[:, :, start_idx:end_idx]))
        # e_motion_disc_loss = e_motion_disc_loss * self.d_motion_loss_weight
        #
        # fake_motion = pred_motion.detach()
        # real_motion = data_motion_mosh['theta']
        # fake_disc_value = motion_discriminator(fake_motion[:, :, start_idx:end_idx])
        # real_disc_value = motion_discriminator(real_motion[:, :, start_idx:end_idx])
        # d_motion_disc_real, d_motion_disc_fake, d_motion_disc_loss = self.dec_loss(real_disc_value, fake_disc_value)
        #
        # d_motion_disc_real = d_motion_disc_real * self.d_motion_loss_weight
        # d_motion_disc_fake = d_motion_disc_fake * self.d_motion_loss_weight
        # d_motion_disc_loss = d_motion_disc_loss * self.d_motion_loss_weight
        #
        # loss_dict['e_m_disc_loss'] = e_motion_disc_loss
        # loss_dict['d_m_disc_real'] = d_motion_disc_real
        # loss_dict['d_m_disc_fake'] = d_motion_disc_fake
        # loss_dict['d_m_disc_loss'] = d_motion_disc_loss
        #
        # gen_loss = gen_loss + e_motion_disc_loss
        # motion_dis_loss = d_motion_disc_loss
        # return gen_loss, motion_dis_loss, loss_dict

    def attention_loss(self, pred_scores):
        gt_labels = torch.ones(len(pred_scores)).to(self.device).long()

        return -0.5 * self.criterion_attention(pred_scores, gt_labels)

    def accel_3d_loss(self, pred_accel, gt_accel):
        pred_accel = pred_accel[:, 25:39]
        gt_accel = gt_accel[:, 25:39]

        if len(gt_accel) > 0:
            return self.criterion_accel(pred_accel, gt_accel).mean()
        else:
            return torch.FloatTensor(1).fill_(0.).to(self.device)

    def keypoint_loss(self, pred_keypoints_2d, gt_keypoints_2d, openpose_weight, gt_weight):
        """
        Compute 2D reprojection loss on the keypoints.
        The loss is weighted by the confidence.
        The available keypoints are different for each dataset.
        """
        conf = gt_keypoints_2d[:, :, -1].unsqueeze(-1).clone()
        conf[:, :25] *= openpose_weight
        conf[:, 25:] *= gt_weight
        loss = (conf * self.criterion_keypoints(pred_keypoints_2d, gt_keypoints_2d[:, :, :-1])).mean()
        return loss

    def keypoint_3d_loss(self, pred_keypoints_3d, gt_keypoints_3d):
        """
        Compute 3D keypoint loss for the examples that 3D keypoint annotations are available.
        The loss is weighted by the confidence.
        """
        # print('check 3d gt base', gt_keypoints_3d[:, 39, :])
        pred_keypoints_3d = pred_keypoints_3d[:, 25:39, :]
        gt_keypoints_3d = gt_keypoints_3d[:, 25:39, :]
        train_gt_np = gt_keypoints_3d[:, 25:39, :].detach().cpu().numpy()
        target_name = 'train.npy'
        np.save(target_name, train_gt_np)

        conf = gt_keypoints_3d[:, :, -1].unsqueeze(-1).clone()
        gt_keypoints_3d = gt_keypoints_3d[:, :, :-1].clone()
        gt_keypoints_3d = gt_keypoints_3d
        conf = conf
        S1_hat = batch_compute_similarity_transform_torch(pred_keypoints_3d, gt_keypoints_3d)
        mpjpe_pa = torch.sqrt(((S1_hat - gt_keypoints_3d) ** 2).sum(dim=-1))
        # mpjpe_pa = mpjpe_pa.mean(axis=-1)
        mpjpe_pa = mpjpe_pa.mean()
        return mpjpe_pa
        # pred_keypoints_3d = pred_keypoints_3d
        # if len(gt_keypoints_3d) > 0:
        #     gt_pelvis = (gt_keypoints_3d[:, 2,:] + gt_keypoints_3d[:, 3,:]) / 2
        #     gt_keypoints_3d = gt_keypoints_3d - gt_pelvis[:, None, :]
        #     pred_pelvis = (pred_keypoints_3d[:, 2,:] + pred_keypoints_3d[:, 3,:]) / 2
        #     pred_keypoints_3d = pred_keypoints_3d - pred_pelvis[:, None, :]
        #     # print(conf.shape, pred_keypoints_3d.shape, gt_keypoints_3d.shape)
        #     # return (conf * self.criterion_keypoints(pred_keypoints_3d, gt_keypoints_3d)).mean()
        #     return self.criterion_keypoints(pred_keypoints_3d, gt_keypoints_3d).mean()
        # else:
        #     return torch.FloatTensor(1).fill_(0.).to(self.device)

    def smpl_losses(self, pred_rotmat, pred_betas, gt_pose, gt_betas):
        pred_rotmat_valid = batch_rodrigues(pred_rotmat.reshape(-1,3)).reshape(-1, 24, 3, 3)
        gt_rotmat_valid = batch_rodrigues(gt_pose.reshape(-1,3)).reshape(-1, 24, 3, 3)
        pred_betas_valid = pred_betas
        gt_betas_valid = gt_betas
        if len(pred_rotmat_valid) > 0:
            loss_regr_pose = self.criterion_regr(pred_rotmat_valid, gt_rotmat_valid)
            loss_regr_betas = self.criterion_regr(pred_betas_valid, gt_betas_valid)
        else:
            loss_regr_pose = torch.FloatTensor(1).fill_(0.).to(self.device)
            loss_regr_betas = torch.FloatTensor(1).fill_(0.).to(self.device)
        return loss_regr_pose, loss_regr_betas


def batch_encoder_disc_l2_loss(disc_value):
    '''
        Inputs:
            disc_value: N x 25
    '''
    k = disc_value.shape[0]
    return torch.sum((disc_value - 1.0) ** 2) * 1.0 / k


def batch_adv_disc_l2_loss(real_disc_value, fake_disc_value):
    '''
        Inputs:
            disc_value: N x 25
    '''
    ka = real_disc_value.shape[0]
    kb = fake_disc_value.shape[0]
    lb, la = torch.sum(fake_disc_value ** 2) / kb, torch.sum((real_disc_value - 1) ** 2) / ka
    return la, lb, la + lb


def batch_encoder_disc_wasserstein_loss(disc_value):
    '''
        Inputs:
            disc_value: N x 25
    '''
    k = disc_value.shape[0]
    return -1 * disc_value.sum() / k


def batch_adv_disc_wasserstein_loss(real_disc_value, fake_disc_value):
    '''
        Inputs:
            disc_value: N x 25
    '''

    ka = real_disc_value.shape[0]
    kb = fake_disc_value.shape[0]

    la = -1 * real_disc_value.sum() / ka
    lb = fake_disc_value.sum() / kb
    return la, lb, la + lb


def batch_smooth_pose_loss(pred_theta):
    pose = pred_theta[:,:,3:75]
    pose_diff = pose[:,1:,:] - pose[:,:-1,:]
    return torch.mean(pose_diff).abs()


def batch_smooth_shape_loss(pred_theta):
    shape = pred_theta[:, :, 75:]
    shape_diff = shape[:, 1:, :] - shape[:, :-1, :]
    return torch.mean(shape_diff).abs()
