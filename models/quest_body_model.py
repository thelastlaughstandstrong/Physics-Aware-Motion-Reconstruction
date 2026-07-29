"""
Drop-in body-model replacement for AvatarPoser's network, targeting SimXR's quest_humanoid
skeleton instead of SMPL-H.

models/network.py's AvatarPoser.fk_module calls `body_model(pose_body=..., root_orient=...)`
and reads `.Jtr` off the result (originally a human_body_prior.BodyModel forward pass). Quest2
synthetic data's pose_quat/pose_quat_global does NOT follow SMPL's kinematic tree -- verified
empirically via a parent-child rotation consistency check against SimXR's own quest_humanoid.xml
hierarchy (0.03 deg max residual, vs 101 deg under the SMPL-hierarchy hypothesis). This class
computes the same (pose_body, root_orient) -> joint-position mapping via poselib's differentiable
SkeletonState forward kinematics on that skeleton, so network.py needs no changes beyond sizing
its output layer to this skeleton's joint count.
"""
import os
import sys
from types import SimpleNamespace

import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PHC_ROOT = os.path.join(REPO_ROOT, "phc")
for _p in (REPO_ROOT, PHC_ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from poselib.poselib.core.rotation3d import quat_from_angle_axis
from poselib.poselib.skeleton.skeleton3d import SkeletonState, SkeletonTree

QUEST_HUMANOID_MJCF = os.path.join(PHC_ROOT, "phc/data/assets/mjcf/quest_humanoid.xml")
QUEST_NUM_JOINTS = 25  # includes root (b_root)

# 0-indexed positions in quest_humanoid's joint order (see SkeletonTree.node_names).
QUEST_HEAD_IDX = 6
QUEST_L_WRIST_IDX = 12
QUEST_R_WRIST_IDX = 18
QUEST_TRACKED_JOINTS = [QUEST_HEAD_IDX, QUEST_L_WRIST_IDX, QUEST_R_WRIST_IDX]


class QuestHumanoidBodyModel:
    """Mimics human_body_prior.BodyModel's forward(pose_body=, root_orient=, trans=) -> .Jtr
    interface, but via poselib SkeletonState FK on quest_humanoid.xml (25 joints) instead of an
    SMPL-H mesh. pose_body: [B, 24*3] axis-angle (24 non-root joints); root_orient: [B, 3]."""

    def __init__(self, device):
        self.skeleton_tree = SkeletonTree.from_mjcf(QUEST_HUMANOID_MJCF)
        self.device = device

    def to(self, device):
        self.device = device
        return self

    def eval(self):
        return self

    def __call__(self, pose_body, root_orient, trans=None, **_ignored):
        # SkeletonTree's own buffers (local_translation, parsed from the MJCF) live on CPU;
        # do the FK there and move the result back, rather than mutating the tree's internals.
        # .to() is autograd-differentiable, so this doesn't break the gradient path.
        out_device = pose_body.device
        pose_body = pose_body.cpu()
        root_orient = root_orient.cpu()

        B = pose_body.shape[0]
        aa = torch.cat([root_orient, pose_body], dim=-1).reshape(B, QUEST_NUM_JOINTS, 3)
        angle = aa.norm(dim=-1)
        quat = quat_from_angle_axis(angle, aa)  # [B, 25, 4] xyzw, differentiable
        if trans is None:
            trans = torch.zeros(B, 3, dtype=aa.dtype)
        else:
            trans = trans.cpu()
        sk_state = SkeletonState.from_rotation_and_root_translation(
            self.skeleton_tree, quat, trans, is_local=True
        )
        return SimpleNamespace(Jtr=sk_state.global_translation.to(out_device), v=None)
