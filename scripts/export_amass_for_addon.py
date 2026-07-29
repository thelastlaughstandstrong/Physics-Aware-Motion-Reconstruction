"""Repackage export_blender_mesh_motion.py's npz (pred/gt positions + already
Mujoco->SMPL-reordered, upright-undone, LOCAL quaternions -- see that script's
own module docstring for how those were derived and validated) into the
standard AMASS .npz format the official SMPL-X Blender addon's
object.smplx_add_animation operator expects: trans, gender, mocap_frame_rate,
betas, poses (T, num_joints*3) axis-angle, in the addon's own SMPLX_JOINT_NAMES
order.

Two things need a coordinate-frame fix that the *local* joint rotations don't:
  - trans (root/pelvis translation): the addon's own source comments say
    "SMPL-X local joint coordinates are Y-Up" -- our pred_pos/gt_pos are in
    PHC's Z-up world (already validated). Z-up (x,y,z) -> Y-up (x,z,-y).
  - the ROOT rotation specifically (global_orient, poses[:, 0:3]): unlike
    every other joint's rotation (which is parent-relative and therefore
    frame-agnostic -- a local elbow bend doesn't care which axis is "up"),
    the root has no parent, so its rotation IS tied to the world frame.
    It's conjugated by the same Z-up -> Y-up rotation (-90 deg about X) so
    that "identity root rotation" in the fed data means "matches this rig's
    own T-pose orientation" rather than PHC's world axes.

All 22 body joints beyond the root use the local rotations directly, no
remap. Joints with no data (jaw, eyes, all 30 finger joints) are zero
(rest pose) -- same set excluded in blender_render_mesh.py's custom rig,
for the same reason (no data / PHC runs robot.freeze_hand=True).

Usage:
    python scripts/export_amass_for_addon.py --npz <blender_mesh_KEY.npz> --out_dir <dir>
    # writes <out_dir>/amass_pred_KEY.npz and <out_dir>/amass_gt_KEY.npz
"""
import argparse
import os

import numpy as np
from scipy.spatial.transform import Rotation as sRot

SMPLX_JOINT_NAMES = (
    "pelvis", "left_hip", "right_hip", "spine1", "left_knee", "right_knee",
    "spine2", "left_ankle", "right_ankle", "spine3", "left_foot", "right_foot",
    "neck", "left_collar", "right_collar", "head", "left_shoulder",
    "right_shoulder", "left_elbow", "right_elbow", "left_wrist", "right_wrist",
    "jaw", "left_eye_smplhf", "right_eye_smplhf",
    "left_index1", "left_index2", "left_index3",
    "left_middle1", "left_middle2", "left_middle3",
    "left_pinky1", "left_pinky2", "left_pinky3",
    "left_ring1", "left_ring2", "left_ring3",
    "left_thumb1", "left_thumb2", "left_thumb3",
    "right_index1", "right_index2", "right_index3",
    "right_middle1", "right_middle2", "right_middle3",
    "right_pinky1", "right_pinky2", "right_pinky3",
    "right_ring1", "right_ring2", "right_ring3",
    "right_thumb1", "right_thumb2", "right_thumb3",
)

# our SMPL_BONE_ORDER_NAMES index -> SMPL-X joint name (same 22-joint mapping
# used in blender_render_mesh.py / export_blender_mesh_motion.py)
SMPL_TO_SMPLX = {
    0: "pelvis", 1: "left_hip", 2: "right_hip", 3: "spine1", 4: "left_knee",
    5: "right_knee", 6: "spine2", 7: "left_ankle", 8: "right_ankle", 9: "spine3",
    10: "left_foot", 11: "right_foot", 12: "neck", 13: "left_collar", 14: "right_collar",
    15: "head", 16: "left_shoulder", 17: "right_shoulder", 18: "left_elbow",
    19: "right_elbow", 20: "left_wrist", 21: "right_wrist",
}

# Rotation taking PHC's Z-up world to this rig's native Y-up convention:
# -90 deg about X, i.e. (x, y, z) -> (x, z, -y). As an xyzw quaternion:
Q_Z2Y = sRot.from_rotvec([-np.pi / 2, 0, 0])  # scipy Rotation, reused for both trans and root conjugation


def zup_to_yup_points(p):  # (..., 3)
    x, y, z = p[..., 0], p[..., 1], p[..., 2]
    return np.stack([x, z, -y], axis=-1)


def build_amass_npz(positions, rot_local_xyzw, fps):
    """positions: (T,24,3) SMPL_BONE_ORDER_NAMES order, PHC Z-up world.
    rot_local_xyzw: (T,24,4) xyzw, local/parent-relative (root = raw global)."""
    T = positions.shape[0]
    trans = zup_to_yup_points(positions[:, 0])  # (T,3)

    poses = np.zeros((T, len(SMPLX_JOINT_NAMES) * 3), dtype=np.float32)
    root_rot = sRot.from_quat(rot_local_xyzw[:, 0])  # PHC Z-up frame
    # NOT a conjugation/similarity transform -- the local/anatomical frame the
    # rotation is applied to is unchanged, only the world frame its RESULT is
    # expressed in changes. If Q_Z2Y.apply(p) remaps a Zup point to Yup, then
    # composing root_rot first and Q_Z2Y second gives the same point correctly
    # remapped: (Q_Z2Y * root_rot).apply(p) == Q_Z2Y.apply(root_rot.apply(p)).
    root_rot_rig = Q_Z2Y * root_rot
    poses[:, 0:3] = root_rot_rig.as_rotvec()

    for smpl_idx, name in SMPL_TO_SMPLX.items():
        if smpl_idx == 0:
            continue
        j = SMPLX_JOINT_NAMES.index(name)
        rotvec = sRot.from_quat(rot_local_xyzw[:, smpl_idx]).as_rotvec()
        poses[:, j * 3:(j + 1) * 3] = rotvec

    return {
        "trans": trans.astype(np.float32),
        "gender": "neutral",
        "mocap_frame_rate": float(fps),
        "betas": np.zeros(10, dtype=np.float32),
        "poses": poses,
        # Frame-1 target in PHC's native Z-up world (not this addon's own
        # convention) -- lets blender_render_addon.py empirically calibrate
        # out a constant offset the smplx_add_animation operator introduces
        # internally (confirmed via isolated bpy testing: feeding this same
        # trans/rotation directly to a bare pelvis bone reproduces the
        # intended position exactly, but the full operator call lands ~0.74m
        # off -- likely from its own shape-key/joint-location-update step --
        # so this is corrected empirically rather than chased further symbolically.
        "target_pelvis_zup_frame1": positions[0, 0].astype(np.float32),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True)
    ap.add_argument("--out_dir", default=None)
    args = ap.parse_args()

    d = np.load(args.npz, allow_pickle=True)
    fps = int(d["fps"])
    key = str(d["motion_key"])
    out_dir = args.out_dir or os.path.dirname(args.npz)

    pred = build_amass_npz(d["pred_pos"], d["pred_rot_local"], fps)
    gt = build_amass_npz(d["gt_pos"], d["gt_rot_local"], fps)

    pred_path = os.path.join(out_dir, f"amass_pred_{key}.npz")
    gt_path = os.path.join(out_dir, f"amass_gt_{key}.npz")
    np.savez(pred_path, **pred)
    np.savez(gt_path, **gt)
    print(f"[export] wrote {pred_path}")
    print(f"[export] wrote {gt_path}")


if __name__ == "__main__":
    main()
