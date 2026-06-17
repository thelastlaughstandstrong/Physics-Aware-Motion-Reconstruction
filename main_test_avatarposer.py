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

    rot_error = []
    pos_error = []
    vel_error = []
    jitter_error = []
    foot_skate_error = []
    gpen_mean_list = []
    gpen_max_list = []
    gpen_rate_list = []
    pos_error_hands = []

    csv_path = os.path.join(opt['path']['log'], 'per_sequence_errors.csv')

    # Resume: if CSV already exists, collect done indices and append; otherwise start fresh
    done_indices = set()
    if os.path.exists(csv_path):
        with open(csv_path, 'r') as _f:
            for row in csv.DictReader(_f):
                done_indices.add(int(row['index']))
        csv_file = open(csv_path, 'a', newline='')
        logger.info('Resuming — {} sequences already done, skipping them.'.format(len(done_indices)))
    else:
        csv_file = open(csv_path, 'w', newline='')
        # csv_writer_tmp = csv.writer(csv_file)
        # csv_writer_tmp.writerow(['index', 'filepath', 'pos_error_cm', 'vel_error_cm_s', 'hand_error_cm',
        #                          'lower_body_error_cm', 'upper_body_error_cm', 'pelvis_error_cm'])
        csv.writer(csv_file).writerow([
            'index', 'filepath',
            'rot_error_deg', 'pos_error_cm', 'vel_error_cm_s',
            'jitter_m_s3', 'foot_skate_cm_s',
            'gpen_mean_mm', 'gpen_max_mm', 'gpen_rate',
            'hand_error_cm', 'lower_body_error_cm', 'upper_body_error_cm', 'pelvis_error_cm',
        ])
    csv_writer = csv.writer(csv_file)

    for index, test_data in enumerate(test_loader):
        if index in done_indices:
            continue
        logger.info("testing the sample {}/{}".format(index, len(test_loader)))

        model.feed_data(test_data, test=True)

        if index == 0:
            print(test_data.keys())
            print(test_data['L'].shape)
            print(test_data['H'].shape)
            print("L dtype:", test_data['L'].dtype)
            print("H dtype:", test_data['H'].dtype)

        model.test()

        body_parms_pred = model.current_prediction()
        body_parms_gt = model.current_gt()
        
        if index == 0:
            print(body_parms_pred.keys())
            print("pose_body:", body_parms_pred['pose_body'].shape)
            print("root_orient:", body_parms_pred['root_orient'].shape)
            print("trans:", body_parms_pred['trans'].shape)
            print("position:", body_parms_pred['position'].shape)

        predicted_angle = body_parms_pred['pose_body']
        predicted_position = body_parms_pred['position']
        predicted_body = body_parms_pred['body']

        gt_angle = body_parms_gt['pose_body']
        gt_position = body_parms_gt['position']
        gt_body = body_parms_gt['body']



        predicted_position = predicted_position#.cpu().numpy()
        gt_position = gt_position#.cpu().numpy()

        predicted_angle = predicted_angle.reshape(body_parms_pred['pose_body'].shape[0],-1,3)
        gt_angle = gt_angle.reshape(body_parms_gt['pose_body'].shape[0],-1,3)

        T = predicted_angle.shape[0]
        pred_rotmat = aa2matrot(predicted_angle.reshape(-1, 3)).reshape(T, -1, 3, 3)
        gt_rotmat = aa2matrot(gt_angle.reshape(-1, 3)).reshape(T, -1, 3, 3)
        rot_error_ = _geodesic_loss(pred_rotmat.reshape(-1, 3, 3), gt_rotmat.reshape(-1, 3, 3))

        pos_error_ = torch.mean(torch.sqrt(torch.sum(torch.square(gt_position-predicted_position),axis=-1)))
        pos_error_hands_ = torch.mean(torch.sqrt(torch.sum(torch.square(gt_position-predicted_position),axis=-1))[...,[20,21]])

        gt_velocity = (gt_position[1:,...] - gt_position[:-1,...])*60
        predicted_velocity = (predicted_position[1:,...] - predicted_position[:-1,...])*60
        vel_error_ = torch.mean(torch.sqrt(torch.sum(torch.square(gt_velocity-predicted_velocity),axis=-1)))

        pred_jerk = (predicted_position[3:,...] - 3*predicted_position[2:-1,...] + 3*predicted_position[1:-2,...] - predicted_position[:-3,...]) * (60**3)
        jitter_error_ = torch.mean(torch.sqrt(torch.sum(torch.square(pred_jerk), axis=-1)))

        # Foot skate: horizontal (xy, AMASS world is z-up) sliding of the feet (joints 10, 11),
        # weighted down to ~0 once the foot lifts above H_THRESH so only ground-contact sliding counts.
        # Same weighting used by the T2M-GPT / EDGE / AGRoL-style "skating ratio" that HUMOS (cited by REWIND) builds on.
        FOOT_JOINTS = [10, 11]
        H_THRESH = 0.033  # meters
        feet_pos = predicted_position[:, FOOT_JOINTS, :]
        feet_horiz_disp = torch.sqrt(torch.sum(torch.square(feet_pos[1:, :, :2] - feet_pos[:-1, :, :2]), axis=-1))
        feet_h = feet_pos[:-1, :, 2]
        skate_weight = torch.clamp(2 - 2 ** (feet_h / H_THRESH), min=0.0, max=1.0)
        foot_skate_ = torch.mean(feet_horiz_disp * skate_weight) * 60

        # ---- Penetration metrics ----
        # Ground penetration: SMPL vertices vs flat floor at Z=0 (AMASS Z-up convention).
        pred_verts = predicted_body.v                                             # [T, 6890, 3]
        gpen_mean_, gpen_max_, gpen_rate_ = compute_ground_penetration(pred_verts)

        # Debug check on the first sequence: compare GT vs predicted to verify
        # floor calibration is correct (GT rate should be near 0).
        if index == 0:
            gt_verts = gt_body.v
            gt_gpen_mean, gt_gpen_max, gt_gpen_rate = compute_ground_penetration(gt_verts)
            logger.info(
                '[DEBUG seq 0] GT  ground pen: mean={:.2f}mm  max={:.2f}mm  rate={:.3f}'.format(
                    gt_gpen_mean.item(), gt_gpen_max.item(), gt_gpen_rate.item()))
            logger.info(
                '[DEBUG seq 0] PRED ground pen: mean={:.2f}mm  max={:.2f}mm  rate={:.3f}'.format(
                    gpen_mean_.item(), gpen_max_.item(), gpen_rate_.item()))
            logger.info(
                '[DEBUG seq 0] PRED foot Z min: {:.4f}m  vert Z min: {:.4f}m'.format(
                    predicted_position[:, [10, 11], 2].min().item(), pred_verts[:, :, 2].min().item()))
            if gt_gpen_rate.item() > 0.30:
                logger.info('[WARNING] GT ground penetration rate is unexpectedly high ({:.1%}). '
                            'Check that global translation places the floor at Z=0.'.format(
                                gt_gpen_rate.item()))

        per_joint_error = torch.sqrt(torch.sum(torch.square(gt_position - predicted_position), axis=-1))
        lower_body_joints = [1, 2, 4, 5, 7, 8, 10, 11]  # hips, knees, ankles, feet
        upper_body_joints = [13, 14, 16, 17, 18, 19]     # collars, shoulders, elbows
        lower_body_error_ = torch.mean(per_joint_error[..., lower_body_joints])
        upper_body_error_ = torch.mean(per_joint_error[..., upper_body_joints])
        pelvis_error_ = torch.mean(per_joint_error[..., 0])

        # if index in [0, 10, 20] and save_animation:

        if save_animation and pos_error_.item() * 100 > 5.0:
            video_dir = os.path.join(opt['path']['images'], str(index))
            if not os.path.exists(video_dir):
                os.makedirs(video_dir)

            save_video_path_gt = os.path.join(video_dir, 'gt.avi')
            if not os.path.exists(save_video_path_gt):
                vis.save_animation(body_pose=gt_body, savepath=save_video_path_gt, bm = model.bm, fps=60, resolution = resolution)

            save_video_path = os.path.join(video_dir, '{:d}.avi'.format(current_step))
            vis.save_animation(body_pose=predicted_body, savepath=save_video_path, bm = model.bm, fps=60, resolution = resolution)

        filepath = test_data['filepath'][0]
        csv_writer.writerow([
            index,
            filepath,
            round(rot_error_.item() * 57.2958, 4),
            round(pos_error_.item() * 100, 4),
            round(vel_error_.item() * 100, 4),
            round(jitter_error_.item(), 4),
            round(foot_skate_.item() * 100, 4),
            round(gpen_mean_.item(), 4),
            round(gpen_max_.item(), 4),
            round(gpen_rate_.item(), 4),
            round(pos_error_hands_.item() * 100, 4),
            round(lower_body_error_.item() * 100, 4),
            round(upper_body_error_.item() * 100, 4),
            round(pelvis_error_.item() * 100, 4),
        ])

        rot_error.append(rot_error_)
        pos_error.append(pos_error_)
        vel_error.append(vel_error_)
        jitter_error.append(jitter_error_)
        foot_skate_error.append(foot_skate_)
        gpen_mean_list.append(gpen_mean_)
        gpen_max_list.append(gpen_max_)
        gpen_rate_list.append(gpen_rate_)
        pos_error_hands.append(pos_error_hands_)



    csv_file.close()
    logger.info('Per-sequence errors saved to {}'.format(csv_path))

    # Step 5 — group errors by motion category
    from collections import defaultdict
    category_errors = defaultdict(list)
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            filename = row['filepath'].split('/')[-1]          # e.g. walk_01_poses.npz
            motion_name = filename.replace('_poses.npz', '')   # e.g. walk_01
            # strip trailing _<number> to get the category
            parts = motion_name.rsplit('_', 1)
            category = parts[0] if len(parts) == 2 and parts[1].isdigit() else motion_name
            category_errors[category].append(float(row['pos_error_cm']))

    summary_path = os.path.join(opt['path']['log'], 'per_category_errors.csv')
    with open(summary_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['category', 'num_sequences', 'mean_pos_error_cm', 'max_pos_error_cm'])
        for category, errors in sorted(category_errors.items(), key=lambda x: -sum(x[1])/len(x[1])):
            writer.writerow([
                category,
                len(errors),
                round(sum(errors) / len(errors), 4),
                round(max(errors), 4)
            ])
    logger.info('Per-category summary saved to {}'.format(summary_path))

    rot_error = sum(rot_error)/len(rot_error)
    pos_error = sum(pos_error)/len(pos_error)
    vel_error = sum(vel_error)/len(vel_error)
    jitter_error     = sum(jitter_error)     / len(jitter_error)
    foot_skate_error = sum(foot_skate_error) / len(foot_skate_error)
    gpen_mean        = sum(gpen_mean_list) / len(gpen_mean_list)
    gpen_max         = sum(gpen_max_list)  / len(gpen_max_list)
    gpen_rate        = sum(gpen_rate_list) / len(gpen_rate_list)
    pos_error_hands  = sum(pos_error_hands) / len(pos_error_hands)

    # testing log
    logger.info(
        'Average rotational error [deg]: {:<.5f}, '
        'Average positional error [cm]: {:<.5f}, '
        'Average velocity error [cm/s]: {:<.5f}, '
        'Average jitter [m/s3]: {:<.5f}, '
        'Average foot skate [cm/s]: {:<.5f}, '
        'Avg ground pen — mean: {:<.3f}mm  max: {:<.3f}mm  rate: {:<.4f}, '
        'Average positional error at hand [cm]: {:<.5f}\n'.format(
            rot_error * 57.2958,
            pos_error * 100,
            vel_error * 100,
            jitter_error,
            foot_skate_error * 100,
            gpen_mean.item(), gpen_max.item(), gpen_rate.item(),
            pos_error_hands * 100,
        )
    )


if __name__ == '__main__':
    main()
