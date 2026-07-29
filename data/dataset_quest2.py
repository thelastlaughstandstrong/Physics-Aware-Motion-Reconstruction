"""
Dataloader for SimXR's Quest2 synthetic motion data (vision-free segments), producing the same
training-sample schema data/dataset_amass.py emits (L/H/P/Head_trans_global/pos_pelvis_gt/
vel_pelvis_gt), so main_train_avatarposer.py / models/model_avatarposer.py need no changes.

Differs from AMASS in two ways, both driven by Quest2 using SimXR's own quest_humanoid skeleton
(25 joints) rather than SMPL (24) -- see models/quest_body_model.py for the verification that
these are genuinely different skeletons, not the same one plus an extra joint:
  - H (ground truth to predict) is 150-dim (25 joints x 6D local rotation) instead of AMASS's
    132-dim (22 SMPL joints x 6D).
  - The three HMD-tracked points (head, left wrist, right wrist) are quest_humanoid's own
    b_head/b_l_wrist/b_r_wrist joints (indices 6/12/18), not SMPL's joints 15/20/21.
The 54-dim HMD input feature layout itself (6D global rot + 6D rot-velocity + 3D pos + 3D
pos-velocity, per tracked joint) is unchanged -- see prepare_data.py for the original formula
this replicates.
"""

import glob
import os
import random
import sys

import numpy as np
import torch
from scipy.spatial.transform import Rotation as sRot
from torch.utils.data import Dataset

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PHC_ROOT = os.path.join(REPO_ROOT, "phc")
for _p in (REPO_ROOT, PHC_ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from poselib.poselib.skeleton.skeleton3d import SkeletonState, SkeletonTree

from models.quest_body_model import (
    QUEST_HEAD_IDX,
    QUEST_HUMANOID_MJCF,
    QUEST_L_WRIST_IDX,
    QUEST_NUM_JOINTS,
    QUEST_R_WRIST_IDX,
    QUEST_TRACKED_JOINTS,
)

_SKELETON_TREE = None  # lazy singleton -- avoid re-parsing the MJCF on every __getitem__ call


def _skeleton_tree():
    global _SKELETON_TREE
    if _SKELETON_TREE is None:
        _SKELETON_TREE = SkeletonTree.from_mjcf(QUEST_HUMANOID_MJCF)
    return _SKELETON_TREE


def _matrot2sixd(rotmat: np.ndarray) -> np.ndarray:
    """rotmat: [..., 3, 3] -> [..., 6], matching utils_transform.matrot2sixd's convention
    (first two columns of the rotation matrix)."""
    return np.concatenate([rotmat[..., :, 0], rotmat[..., :, 1]], axis=-1)


class Quest2_Dataset(Dataset):
    """Quest2 synthetic motion dataset (motion-only, vision-free segments)."""

    def __init__(self, opt):
        self.opt = opt
        self.window_size = opt['window_size']
        self.batch_size = opt['dataloader_batch_size']
        dataroot = opt['dataroot']
        subdir = 'train_seg_motion' if opt['phase'] == 'train' else 'test_seg_motion'
        self.filename_list = sorted(glob.glob(os.path.join(dataroot, subdir, '*.pkl')))
        print('-------------------------------number of {} data is {}'.format(
            opt['phase'], len(self.filename_list)))

    def __len__(self):
        return max(len(self.filename_list), self.batch_size)

    def _load_sequence(self, idx):
        import joblib
        filename = self.filename_list[idx]
        data = joblib.load(filename)
        (key, entry), = data.items()
        return key, entry

    def _build_features(self, entry):
        """Returns (hmd_position_global_full_gt_list [T-1,54], rotation_local_full_gt_list
        [T-1,150], head_global_trans_list [T-1,4,4], root_trans [T,3]) -- T-1 since frame 0 is
        dropped throughout (matching prepare_data.py: velocity terms need a previous frame)."""
        pose_quat = entry['pose_quat']
        pose_quat_global = entry['pose_quat_global']
        root_trans_offset = entry['root_trans_offset']
        pose_quat = pose_quat.numpy() if isinstance(pose_quat, torch.Tensor) else pose_quat
        pose_quat_global = (pose_quat_global.numpy() if isinstance(pose_quat_global, torch.Tensor)
                             else pose_quat_global)
        root_trans_offset = (root_trans_offset.numpy() if isinstance(root_trans_offset, torch.Tensor)
                              else root_trans_offset)

        T = pose_quat.shape[0]

        # World positions of all joints, via FK on the (non-upright) quest_humanoid skeleton --
        # self-consistent with the verified-correct pose_quat_global/root_trans_offset pairing,
        # rather than relying on head_gt_3d/head_fk_gt_3d's unverified derivation.
        sk_state = SkeletonState.from_rotation_and_root_translation(
            _skeleton_tree(),
            torch.from_numpy(pose_quat_global.astype(np.float32)),
            torch.from_numpy(root_trans_offset.astype(np.float32)),
            is_local=False,
        )
        positions = sk_state.global_translation.numpy()  # [T, 25, 3]

        tracked_global = pose_quat_global[:, QUEST_TRACKED_JOINTS, :]  # [T,3,4]
        tracked_rotmat = sRot.from_quat(tracked_global.reshape(-1, 4)).as_matrix().reshape(T, 3, 3, 3)
        rotation_global_6d = _matrot2sixd(tracked_rotmat)  # [T,3,6]
        input_rotation_global_6d = rotation_global_6d[1:]  # [T-1,3,6]

        rel_rotmat = np.matmul(
            np.transpose(tracked_rotmat[:-1], (0, 1, 3, 2)), tracked_rotmat[1:]
        )  # R(t-1)^-1 @ R(t), [T-1,3,3,3]
        input_rotation_velocity_global_6d = _matrot2sixd(rel_rotmat)  # [T-1,3,6]

        tracked_pos = positions[:, QUEST_TRACKED_JOINTS, :]  # [T,3,3]
        num_frames = T - 1

        hmd_position_global_full_gt_list = np.concatenate([
            input_rotation_global_6d.reshape(num_frames, -1),
            input_rotation_velocity_global_6d.reshape(num_frames, -1),
            tracked_pos[1:].reshape(num_frames, -1),
            (tracked_pos[1:] - tracked_pos[:-1]).reshape(num_frames, -1),
        ], axis=-1)  # [T-1, 54]

        # GT output: local 6D rotation for ALL 25 quest_humanoid joints (pose_quat is already
        # local, verified against the true MJCF hierarchy -- see models/quest_body_model.py).
        all_rotmat = sRot.from_quat(pose_quat.reshape(-1, 4)).as_matrix().reshape(T, QUEST_NUM_JOINTS, 3, 3)
        rotation_local_full_gt_list = _matrot2sixd(all_rotmat).reshape(T, -1)[1:]  # [T-1, 150]

        head_rot = tracked_rotmat[1:, 0]  # head is QUEST_TRACKED_JOINTS[0]
        head_pos = tracked_pos[1:, 0]
        head_global_trans = np.tile(np.eye(4, dtype=np.float32), (num_frames, 1, 1))
        head_global_trans[:, :3, :3] = head_rot
        head_global_trans[:, :3, 3] = head_pos

        return (
            torch.from_numpy(hmd_position_global_full_gt_list.astype(np.float32)),
            torch.from_numpy(rotation_local_full_gt_list.astype(np.float32)),
            torch.from_numpy(head_global_trans),
            torch.from_numpy(root_trans_offset.astype(np.float32)),
        )

    def __getitem__(self, idx):
        key, entry = self._load_sequence(idx)

        if self.opt['phase'] == 'train':
            while entry['pose_quat'].shape[0] < self.window_size + 2:
                idx = random.randint(0, idx if idx > 0 else len(self.filename_list) - 1)
                key, entry = self._load_sequence(idx)

        hmd, rot_gt, head_trans, root_trans = self._build_features(entry)

        if self.opt['phase'] == 'train':
            frame = np.random.randint(hmd.shape[0] - self.window_size + 1 - 1)
            input_hmd = hmd[frame:frame + self.window_size + 1, ...].reshape(self.window_size + 1, -1).float()
            output_gt = rot_gt[frame + self.window_size - 1: frame + self.window_size - 1 + 1, ...].float()

            return {
                'L': input_hmd,
                'H': output_gt,
                'P': 1,
                'Head_trans_global': head_trans[frame + self.window_size - 1:frame + self.window_size - 1 + 1, ...],
                'pos_pelvis_gt': root_trans[frame + self.window_size - 1:frame + self.window_size - 1 + 1, ...],
                'vel_pelvis_gt': (root_trans[frame + self.window_size - 1:frame + self.window_size - 1 + 1, ...]
                                   - root_trans[frame + self.window_size - 2:frame + self.window_size - 2 + 1, ...]),
            }
        else:
            # dataset_amass.py's test branch applies an extra [1:] on top of the pkl's
            # already-frame-0-dropped arrays (2 frames dropped in total, matching root_trans[2:]
            # below which is applied directly to the never-dropped full-length array).
            return {
                'L': hmd[1:].float(),
                'H': rot_gt[1:].float(),
                'P': {'pose_quat': entry['pose_quat'], 'pose_quat_global': entry['pose_quat_global'],
                      'root_trans_offset': entry['root_trans_offset']},
                'Head_trans_global': head_trans[1:],
                'pos_pelvis_gt': root_trans[2:],
                'vel_pelvis_gt': root_trans[2:] - root_trans[1:-1],
                'filepath': key,
            }
