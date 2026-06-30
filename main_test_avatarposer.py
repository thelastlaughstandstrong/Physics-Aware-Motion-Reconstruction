'''
# --------------------------------------------
# main testing code
# --------------------------------------------
# AvatarPoser: Articulated Full-Body Pose Tracking from Sparse Motion Sensing (ECCV 2022)
# https://github.com/eth-siplab/AvatarPoser
# Jiaxi Jiang (jiaxi.jiang@inf.ethz.ch)
# Sensing, Interaction & Perception Lab,
# Department of Computer Science, ETH Zurich
'''

import os.path
import argparse
import csv
import numpy as np
import logging
import torch
from torch.utils.data import DataLoader
from utils import utils_logger
from utils import utils_option as option
from data.select_dataset import define_Dataset
from models.select_model import define_Model
from utils import utils_visualize as vis
from human_body_prior.tools.rotation_tools import aa2matrot
from human_body_prior.tools.angle_continuous_repres import geodesic_loss_R
from utils.physical_metrics import compute_ground_penetration

save_animation = False
resolution = (800,800)

def main(json_path='options/test_avatarposer.json'):

    '''
    # ----------------------------------------
    # Step--1 (prepare opt)
    # ----------------------------------------
    '''

    parser = argparse.ArgumentParser()
    parser.add_argument('-opt', type=str, default=json_path, help='Path to option JSON file.')

    opt = option.parse(parser.parse_args().opt, is_train=True)

    paths = (path for key, path in opt['path'].items() if 'pretrained' not in key)
    if isinstance(paths, str):
        if not os.path.exists(paths):
            os.makedirs(paths)
    else:
        for path in paths:
            if not os.path.exists(path):
                os.makedirs(path)

    # ----------------------------------------
    # update opt
    # ----------------------------------------
    # -->-->-->-->-->-->-->-->-->-->-->-->-->-
    init_iter, init_path_G = option.find_last_checkpoint(opt['path']['models'], net_type='G')
    opt['path']['pretrained_netG'] = init_path_G
    if init_path_G is not None:
        opt['path']['pretrained'] = init_path_G
    current_step = init_iter

    # --<--<--<--<--<--<--<--<--<--<--<--<--<-

    # ----------------------------------------
    # save opt to  a '../option.json' file
    # ----------------------------------------
    option.save(opt)

    # ----------------------------------------
    # return None for missing key
    # ----------------------------------------
    opt = option.dict_to_nonedict(opt)

    # ----------------------------------------
    # configure logger
    # ----------------------------------------
    logger_name = 'train'
    utils_logger.logger_info(logger_name, os.path.join(opt['path']['log'], logger_name+'.log'))
    logger = logging.getLogger(logger_name)


    '''
    # ----------------------------------------
    # Step--2 (creat dataloader)
    # ----------------------------------------
    '''

    # ----------------------------------------
    # 1) create_dataset
    # 2) creat_dataloader for train and test
    # ----------------------------------------
    dataset_type = opt['datasets']['test']['dataset_type']
    for phase, dataset_opt in opt['datasets'].items():

        if phase == 'test':
            test_set = define_Dataset(dataset_opt)
            test_loader = DataLoader(test_set, batch_size=dataset_opt['dataloader_batch_size'],
                                     shuffle=False, num_workers=1,
                                     drop_last=False, pin_memory=True)
        elif phase == 'train':
            continue
        else:
            raise NotImplementedError("Phase [%s] is not recognized." % phase)

    '''
    # ----------------------------------------
    # Step--3 (initialize model)
    # ----------------------------------------
    '''

    model = define_Model(opt)

    if opt['merge_bn'] and current_step > opt['merge_bn_startpoint']:
        logger.info('^_^ -----merging bnorm----- ^_^')
        model.merge_bnorm_test()

    model.init_test()

    _geodesic_loss = geodesic_loss_R(reduction='mean')

    mpjre_list = []
    mpjpe_list = []
    mpjve_list = []
    jitter_list = []
    foot_skate_list = []
    gpen_mean_list = []
    gpen_rate_list = []
    lower_mpjpe_list = []
    upper_mpjpe_list = []
    pelvis_err_list = []

    csv_path = os.path.join(opt['path']['log'], 'per_sequence_errors.csv')

    csv_file = open(csv_path, 'w', newline='')
    csv.writer(csv_file).writerow([
        'index', 'filepath',
        'MPJRE_deg', 'MPJPE_cm', 'MPJVE_cm_s',
        'Jitter_m_s3', 'FootSkate_cm_s',
        'GPen_mean_mm', 'GPen_rate',
        'LowerBody_MPJPE_cm', 'UpperBody_MPJPE_cm', 'Pelvis_err_cm',
    ])
    csv_writer = csv.writer(csv_file)

    for index, test_data in enumerate(test_loader):
        logger.info("testing the sample {}/{}".format(index, len(test_loader)))

        model.feed_data(test_data, test=True)
        model.test()

        body_parms_pred = model.current_prediction()
        body_parms_gt   = model.current_gt()

        predicted_angle    = torch.cat([body_parms_pred['root_orient'], body_parms_pred['pose_body']], dim=-1)
        predicted_position = body_parms_pred['position']
        predicted_body     = body_parms_pred['body']

        gt_angle    = torch.cat([body_parms_gt['root_orient'], body_parms_gt['pose_body']], dim=-1)
        gt_position = body_parms_gt['position']

        T = predicted_angle.shape[0]
        pred_rotmat = aa2matrot(predicted_angle.reshape(-1, 3)).reshape(T, -1, 3, 3)
        gt_rotmat   = aa2matrot(gt_angle.reshape(-1, 3)).reshape(T, -1, 3, 3)
        mpjre_ = _geodesic_loss(pred_rotmat.reshape(-1, 3, 3), gt_rotmat.reshape(-1, 3, 3))

        per_joint_err = torch.sqrt(torch.sum(torch.square(gt_position - predicted_position), dim=-1))
        mpjpe_ = per_joint_err.mean()

        gt_vel   = (gt_position[1:] - gt_position[:-1]) * 60
        pred_vel = (predicted_position[1:] - predicted_position[:-1]) * 60
        mpjve_ = torch.sqrt(torch.sum(torch.square(gt_vel - pred_vel), dim=-1)).mean()

        pred_jerk = (predicted_position[3:] - 3*predicted_position[2:-1]
                     + 3*predicted_position[1:-2] - predicted_position[:-3]) * (60**3)
        jitter_ = torch.sqrt(torch.sum(torch.square(pred_jerk), dim=-1)).mean()

        FOOT_JOINTS = [10, 11]
        H_THRESH = 0.033
        feet_pos        = predicted_position[:, FOOT_JOINTS, :]
        feet_horiz_disp = torch.sqrt(torch.sum(torch.square(feet_pos[1:, :, :2] - feet_pos[:-1, :, :2]), dim=-1))
        feet_h          = feet_pos[:-1, :, 2]
        skate_weight    = torch.clamp(2 - 2 ** (feet_h / H_THRESH), min=0.0, max=1.0)
        foot_skate_     = (feet_horiz_disp * skate_weight).mean() * 60

        pred_verts = predicted_body.v
        gpen_mean_, _, gpen_rate_ = compute_ground_penetration(pred_verts)

        lower_body_joints = [1, 2, 4, 5, 7, 8, 10, 11]
        upper_body_joints = [13, 14, 16, 17, 18, 19]
        lower_mpjpe_ = per_joint_err[..., lower_body_joints].mean()
        upper_mpjpe_ = per_joint_err[..., upper_body_joints].mean()
        pelvis_err_  = per_joint_err[..., 0].mean()

        if save_animation and mpjpe_.item() * 100 > 5.0:
            video_dir = os.path.join(opt['path']['images'], str(index))
            if not os.path.exists(video_dir):
                os.makedirs(video_dir)
            save_video_path_gt = os.path.join(video_dir, 'gt.avi')
            if not os.path.exists(save_video_path_gt):
                vis.save_animation(body_pose=body_parms_gt['body'], savepath=save_video_path_gt, bm=model.bm, fps=60, resolution=resolution)
            vis.save_animation(body_pose=predicted_body, savepath=os.path.join(video_dir, '{:d}.avi'.format(current_step)), bm=model.bm, fps=60, resolution=resolution)

        filepath = test_data['filepath'][0]
        csv_writer.writerow([
            index, filepath,
            round(mpjre_.item() * 57.2958, 4),
            round(mpjpe_.item() * 100, 4),
            round(mpjve_.item() * 100, 4),
            round(jitter_.item(), 4),
            round(foot_skate_.item() * 100, 4),
            round(gpen_mean_.item(), 4),
            round(gpen_rate_.item(), 4),
            round(lower_mpjpe_.item() * 100, 4),
            round(upper_mpjpe_.item() * 100, 4),
            round(pelvis_err_.item() * 100, 4),
        ])

        mpjre_list.append(mpjre_)
        mpjpe_list.append(mpjpe_)
        mpjve_list.append(mpjve_)
        jitter_list.append(jitter_)
        foot_skate_list.append(foot_skate_)
        gpen_mean_list.append(gpen_mean_)
        gpen_rate_list.append(gpen_rate_)
        lower_mpjpe_list.append(lower_mpjpe_)
        upper_mpjpe_list.append(upper_mpjpe_)
        pelvis_err_list.append(pelvis_err_)



    csv_file.close()
    logger.info('Per-sequence errors saved to {}'.format(csv_path))

    from collections import defaultdict
    category_errors = defaultdict(list)
    with open(csv_path, 'r') as f:
        for row in csv.DictReader(f):
            filename    = row['filepath'].split('/')[-1]
            motion_name = filename.replace('_poses.npz', '')
            parts       = motion_name.rsplit('_', 1)
            category    = parts[0] if len(parts) == 2 and parts[1].isdigit() else motion_name
            category_errors[category].append(float(row['MPJPE_cm']))

    summary_path = os.path.join(opt['path']['log'], 'per_category_errors.csv')
    with open(summary_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['category', 'num_sequences', 'mean_MPJPE_cm', 'max_MPJPE_cm'])
        for category, errors in sorted(category_errors.items(), key=lambda x: -sum(x[1]) / len(x[1])):
            writer.writerow([category, len(errors),
                             round(sum(errors) / len(errors), 4), round(max(errors), 4)])
    logger.info('Per-category summary saved to {}'.format(summary_path))

    mpjre        = sum(mpjre_list)        / len(mpjre_list)
    mpjpe        = sum(mpjpe_list)        / len(mpjpe_list)
    mpjve        = sum(mpjve_list)        / len(mpjve_list)
    jitter       = sum(jitter_list)       / len(jitter_list)
    foot_skate   = sum(foot_skate_list)   / len(foot_skate_list)
    gpen_mean    = sum(gpen_mean_list)    / len(gpen_mean_list)
    gpen_rate    = sum(gpen_rate_list)    / len(gpen_rate_list)
    lower_mpjpe  = sum(lower_mpjpe_list)  / len(lower_mpjpe_list)
    upper_mpjpe  = sum(upper_mpjpe_list)  / len(upper_mpjpe_list)
    pelvis_err   = sum(pelvis_err_list)   / len(pelvis_err_list)

    logger.info(
        'MPJRE [deg]: {:<.4f} | '
        'MPJPE [cm]: {:<.4f} | '
        'MPJVE [cm/s]: {:<.4f} | '
        'Jitter [m/s³]: {:<.4f} | '
        'Foot Skating [cm/s]: {:<.4f} | '
        'GPen mean [mm]: {:<.3f}  rate: {:<.4f} | '
        'Lower MPJPE [cm]: {:<.4f} | '
        'Upper MPJPE [cm]: {:<.4f} | '
        'Pelvis Err [cm]: {:<.4f}\n'.format(
            mpjre * 57.2958,
            mpjpe * 100,
            mpjve * 100,
            jitter,
            foot_skate * 100,
            gpen_mean.item(), gpen_rate.item(),
            lower_mpjpe * 100,
            upper_mpjpe * 100,
            pelvis_err * 100,
        )
    )


if __name__ == '__main__':
    main()
