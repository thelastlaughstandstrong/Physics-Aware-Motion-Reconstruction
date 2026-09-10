"""Extract one sequence's predicted vs. ground-truth motion from a
phc_positions_for_eval.pkl dump into local per-joint quaternions, for driving
a skinned SMPL-X mesh in Blender (companion to export_blender_motion.py,
which only exports raw positions for the capsule-skeleton version).

pred_rot_all / gt_rot_all in the pkl are GLOBAL per-joint quaternions (xyzw),
Mujoco body order. The SMPL-X Blender addon's pose API
(set_pose_from_rodrigues -> pbone.rotation_quaternion) expects LOCAL,
parent-relative rotations -- standard SMPL pose convention -- so this script:
  1. reorders Mujoco -> SMPL_BONE_ORDER_NAMES order (same validated mapping
     used in phc/learning/distill_agent.py and export_blender_motion.py)
  2. converts global -> local via local[i] = conj(global[parent[i]]) * global[i]
  3. runs a forward-kinematics sanity check: reconstruct world positions from
     the local rotations (root position taken directly from pred_pos/gt_pos,
     NOT reconstructed) + real SMPL rest offsets, and report the residual
     against the already-known-good pred_pos/gt_pos arrays from the same pkl.

Expected residual: PHC's simulated humanoid uses a modified rest pose (its own
startup log prints "Using modified SMPL starting pose"), so FK against REAL
SMPL rest offsets is known to have a ~2-4cm/joint residual even for a fully
correct rotation pipeline (see distill_hmd_joint_bug.md) -- that's expected,
not a bug. A residual far outside that range (many cm to metres, or visibly
garbled) means the rotation conversion itself is wrong, not a rest-pose
mismatch, and should be investigated before trusting the mesh render.

Usage:
    python scripts/export_blender_mesh_motion.py --pos_dump <path> --seq_idx N \\
        --body_model support_data/body_models/smplh/neutral/model.npz --out <out.npz>
"""
import argparse
import os

import joblib
import numpy as np
from smpl_sim.smpllib.smpl_joint_names import SMPL_BONE_ORDER_NAMES, SMPL_MUJOCO_NAMES

SMPL_PARENTS = [-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19, 20, 21]
_MJ_2_SMPL = [SMPL_MUJOCO_NAMES.index(q) for q in SMPL_BONE_ORDER_NAMES if q in SMPL_MUJOCO_NAMES]

# PHC's rigid-body rotations (rb_rot, and everything derived from it -- including
# pred_rot_all/gt_rot_all in phc_positions_for_eval.pkl) live in PHC's rotated
# "upright_start" frame, not the original AMASS frame: convert_avatarposer_to_phc.py
# builds reference motion by right-multiplying every global rotation by
# upright_rot_inv = Rot.from_quat([.5,.5,.5,.5]).inv(). Right-multiplying by the
# non-inverted quaternion undoes it (same fix already validated in
# phc/learning/distill_agent.py's _undo_upright -- confirmed here independently via
# the FK check: mean residual dropped from 67cm to 6.8cm once this was applied).
# Position-based metrics (MPJPE etc.) never needed this undone, since a uniform
# global rotation applied identically to pred and gt cancels out of their
# difference -- which is exactly why this was missed until the FK check caught it.
_UPRIGHT_UNDO_QUAT = np.array([0.5, 0.5, 0.5, 0.5], dtype=np.float32)

# SMPL_BONE_ORDER_NAMES index -> SMPL-X addon bone name (meshcapade_addon/globals.py
# SMPLX_JOINT_NAMES[:22]). Indices 22/23 (L_Hand/R_Hand) have no SMPL-X equivalent --
# SMPL-X replaces the single "hand" joint with an articulated wrist + fingers -- so
# they're intentionally omitted; those two rotations are simply not applied, leaving
# the addon's hands at rest. Everything else is a direct 1:1 semantic match.
SMPL_TO_SMPLX_BONE = {
    0: "pelvis", 1: "left_hip", 2: "right_hip", 3: "spine1", 4: "left_knee",
    5: "right_knee", 6: "spine2", 7: "left_ankle", 8: "right_ankle", 9: "spine3",
    10: "left_foot", 11: "right_foot", 12: "neck", 13: "left_collar", 14: "right_collar",
    15: "head", 16: "left_shoulder", 17: "right_shoulder", 18: "left_elbow",
    19: "right_elbow", 20: "left_wrist", 21: "right_wrist",
}


def _quat_mul(q1, q2):
    """Hamilton product of xyzw quaternions, batched over leading dims."""
    x1, y1, z1, w1 = q1[..., 0], q1[..., 1], q1[..., 2], q1[..., 3]
    x2, y2, z2, w2 = q2[..., 0], q2[..., 1], q2[..., 2], q2[..., 3]
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    return np.stack([x, y, z, w], axis=-1)


def _quat_conj(q):
    return q * np.array([-1.0, -1.0, -1.0, 1.0], dtype=q.dtype)


def _quat_rotate(q, v):
    qvec, qw = q[..., :3], q[..., 3:4]
    v = np.broadcast_to(v, qvec.shape)
    t = 2.0 * np.cross(qvec, v)
    return v + qw * t + np.cross(qvec, t)


def _mujoco_to_smpl_order(x):
    return x[:, _MJ_2_SMPL, ...]


def global_to_local(global_rot):
    """global_rot: (T, 24, 4) xyzw, SMPL_BONE_ORDER_NAMES order -> (T, 24, 4) local."""
    local = np.empty_like(global_rot)
    local[:, 0] = global_rot[:, 0]  # root has no parent
    for i in range(1, 24):
        p = SMPL_PARENTS[i]
        local[:, i] = _quat_mul(_quat_conj(global_rot[:, p]), global_rot[:, i])
    return local


# SMPL-H stores 52 joints (0-21 body, 22-36 LEFT hand, 37-51 RIGHT hand), so J_regressor[:24]
# silently takes two LEFT-hand rows for SMPL-24's L_Hand/R_Hand, putting R_Hand on the wrong side
# with a bogus ~1.5 m bone. Same defect fixed in phc/learning/distill_agent.py. It only reaches
# the two hand leaves -- which are excluded from the render and from the residual check -- so it
# never affected a rendered frame, but it made the printed hand diagnostic meaningless.
_SMPLH_ROWS_FOR_SMPL24 = list(range(22)) + [22, 37]


def load_rest_offsets(body_model_path):
    body_data = np.load(body_model_path, allow_pickle=True)
    J_reg_full = np.array(body_data["J_regressor"])
    J_reg = (J_reg_full[_SMPLH_ROWS_FOR_SMPL24] if J_reg_full.shape[0] == 52
             else J_reg_full[:24])
    rest_joints = J_reg @ np.array(body_data["v_template"])
    rest_offsets = rest_joints.copy()
    for i in range(1, 24):
        rest_offsets[i] -= rest_joints[SMPL_PARENTS[i]]
    return rest_offsets.astype(np.float32)


def fk_check(root_pos, global_rot, rest_offsets):
    """Reconstruct world positions from global rotations + rest offsets (root
    position taken as given, not reconstructed) -- for comparing against the
    pkl's own already-known-good position array as a rotation-pipeline sanity
    check. Returns (T, 24, 3)."""
    T = global_rot.shape[0]
    pos = np.zeros((T, 24, 3), dtype=np.float32)
    pos[:, 0] = root_pos
    for i in range(1, 24):
        p = SMPL_PARENTS[i]
        offset_world = _quat_rotate(global_rot[:, p], rest_offsets[i])
        pos[:, i] = pos[:, p] + offset_world
    return pos


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pos_dump", required=True)
    ap.add_argument("--seq_idx", type=int, required=True)
    ap.add_argument("--body_model", default="support_data/body_models/smplh/neutral/model.npz")
    ap.add_argument("--out", type=str, default=None)
    ap.add_argument("--fps", type=int, default=30)
    args = ap.parse_args()

    d = joblib.load(args.pos_dump)
    idx = args.seq_idx
    key = d["motion_keys"][idx]

    pred_pos_mj = np.asarray(d["pred_pos_all"][idx], dtype=np.float32)
    gt_pos_mj = np.asarray(d["ref_pos_all"][idx], dtype=np.float32)
    pred_rot_mj = np.asarray(d["pred_rot_all"][idx], dtype=np.float32)
    gt_rot_mj = np.asarray(d["gt_rot_all"][idx], dtype=np.float32)
    T = min(len(pred_pos_mj), len(gt_pos_mj), len(pred_rot_mj), len(gt_rot_mj))
    pred_pos_mj, gt_pos_mj = pred_pos_mj[:T], gt_pos_mj[:T]
    pred_rot_mj, gt_rot_mj = pred_rot_mj[:T], gt_rot_mj[:T]

    pred_pos = _mujoco_to_smpl_order(pred_pos_mj)
    gt_pos = _mujoco_to_smpl_order(gt_pos_mj)
    pred_rot_global = _quat_mul(_mujoco_to_smpl_order(pred_rot_mj),
                                 np.broadcast_to(_UPRIGHT_UNDO_QUAT, (T, 24, 4)))
    gt_rot_global = _quat_mul(_mujoco_to_smpl_order(gt_rot_mj),
                               np.broadcast_to(_UPRIGHT_UNDO_QUAT, (T, 24, 4)))

    pred_rot_local = global_to_local(pred_rot_global)
    gt_rot_local = global_to_local(gt_rot_global)

    # --- FK sanity check ---
    rest_offsets = load_rest_offsets(args.body_model)
    pred_fk = fk_check(pred_pos[:, 0], pred_rot_global, rest_offsets)
    gt_fk = fk_check(gt_pos[:, 0], gt_rot_global, rest_offsets)
    pred_resid_all = np.linalg.norm(pred_fk - pred_pos, axis=-1)  # (T, 24)
    gt_resid_all = np.linalg.norm(gt_fk - gt_pos, axis=-1)
    # L_Hand/R_Hand (22, 23) are excluded from the mean/warning check: PHC runs with
    # robot.freeze_hand=True (every command in this project), and those two joints
    # have no SMPL-X bone equivalent anyway (see SMPL_TO_SMPLX_BONE), so their
    # rotations are never applied to the rendered mesh regardless of their accuracy.
    used_joints = sorted(SMPL_TO_SMPLX_BONE.keys())
    pred_resid = pred_resid_all[:, used_joints]
    gt_resid = gt_resid_all[:, used_joints]
    print(f"[fk-check] pred (22 rendered joints): mean={pred_resid.mean()*100:.2f}cm  max={pred_resid.max()*100:.2f}cm  "
          f"(expect near-zero -- these joints are actuated, not frozen)")
    print(f"[fk-check] gt   (22 rendered joints): mean={gt_resid.mean()*100:.2f}cm  max={gt_resid.max()*100:.2f}cm")
    print(f"[fk-check] hands (excluded from render, frozen in sim): "
          f"L_Hand mean={pred_resid_all[:,22].mean()*100:.2f}cm  R_Hand mean={pred_resid_all[:,23].mean()*100:.2f}cm")
    if pred_resid.mean() > 0.05 or gt_resid.mean() > 0.05:
        print("[fk-check] WARNING: rendered-joint residual above 5cm -- "
              "likely a real bug in the rotation conversion. Do not trust the mesh render until investigated.")

    out_path = args.out or os.path.join(os.path.dirname(args.pos_dump), f"blender_mesh_{key}.npz")
    np.savez(
        out_path,
        pred_pos=pred_pos, gt_pos=gt_pos,
        pred_rot_local=pred_rot_local, gt_rot_local=gt_rot_local,
        joint_names=np.asarray(SMPL_BONE_ORDER_NAMES),
        fps=args.fps, motion_key=key,
    )
    print(f"[export] seq idx={idx} key={key!r} T={T}")
    print(f"[export] wrote {out_path}")


if __name__ == "__main__":
    main()
