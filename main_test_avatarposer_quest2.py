"""
Evaluate an AvatarPoser model trained on Quest2 synthetic data (quest_humanoid skeleton),
computing the same metric set as results/baselines/avatarposer_w40_baseline.json (rot_error_deg,
pos_error_cm, vel_error_cm_s, jitter_m_s3, foot_skate_cm_s, gpen_mean_mm, gpen_max_mm, gpen_rate)
plus SimXR-comparable Eg-mpjpe/success-rate (SimXR Table 5 definitions).

Doesn't reuse models/model_avatarposer.py's test() (SMPL-specific global-root reconstruction +
hardcoded wrist indices [20,21], not valid for quest_humanoid -- see options/train_avatarposer_
quest2.json's comment). Instead drives the network directly, and reconstructs world position by
anchoring on the *head's* known tracked pose (Head_trans_global -- literally part of the model's
input signal, unlike root/pelvis translation which is never directly tracked and would have been
an oracle shortcut): FK the predicted rotations with translation=0 to get a root-at-origin
skeleton, then rigidly shift the whole predicted skeleton so its own head lands on the known
head position, exactly matching how a real deployed system would reconstruct world pose.
NOTE: because head is only ~6 kinematic hops from a well-anchored point, "success" (head within
0.5m) will be near-100% by construction for basically any reasonable pose predictor -- this is
NOT the same test as SimXR's success metric, which comes from a temporally-extended physics
rollout where the simulated character can drift/fall over time; a static per-frame kinematic
regression has no such compounding-failure mode to trigger on. Eg-mpjpe/pos_error_cm (averaged
over all 25 joints, most of which are many hops from the head anchor and do reflect real
reconstruction error) is the metric that's actually informative here.

gpen (ground penetration) here is a joint-position approximation (floor at Z=0, all 25 joint
centers), NOT the true SMPL-H mesh-vertex method the original avatarposer_w40_baseline.json
uses -- quest_humanoid has no mesh asset ported into this repo (that's Phase B work), so this
number is directly comparable to evaluate_phc_metrics.py's compute_gpen() approximation, not to
the baseline JSON's mesh-based gpen numbers.

Usage:
    python main_test_avatarposer_quest2.py -opt options/train_avatarposer_quest2.json -ckpt <path>
"""
import argparse
import datetime
import json
import os
import sys

import numpy as np
import torch
from scipy.spatial.transform import Rotation as sRot
from torch.utils.data import DataLoader

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
PHC_ROOT = os.path.join(REPO_ROOT, "phc")
for _p in (REPO_ROOT, PHC_ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from poselib.poselib.skeleton.skeleton3d import SkeletonState, SkeletonTree

from data.select_dataset import define_Dataset
from models.select_model import define_G
from models.quest_body_model import QUEST_HEAD_IDX, QUEST_HUMANOID_MJCF, QUEST_NUM_JOINTS
from utils import utils_option as option

FOOT_JOINTS = [21, 24]  # b_l_foot_twist, b_r_foot_twist
FPS = 30.0
H_THRESH = 0.033


def sixd_to_quat_np(sixd: np.ndarray) -> np.ndarray:
    """sixd: [...,6] -> [...,4] xyzw, matching utils_transform.sixd2matrot's convention
    (first two columns of the rotation matrix, third via cross product)."""
    shape = sixd.shape[:-1]
    v = sixd.reshape(-1, 6)
    c1 = v[:, 0:3]
    c2 = v[:, 3:6]
    c3 = np.cross(c1, c2)
    rotmat = np.stack([c1, c2, c3], axis=-1)  # columns
    quat = sRot.from_matrix(rotmat).as_quat()
    return quat.reshape(*shape, 4)


def compute_foot_skate(pos: np.ndarray, ground_z: float) -> float:
    """pos: [N,25,3] world positions. Same weighting as evaluate_phc_metrics.py's
    compute_foot_skate, adapted to quest_humanoid's foot joints. ground_z: this sequence's
    empirical ground reference (see main()'s per-sequence calibration -- quest_humanoid's
    foot_twist joint is an ankle-twist auxiliary joint, not a toe/sole marker, so it never
    reaches literal Z=0 even when the foot is genuinely planted)."""
    feet_pos = pos[:, FOOT_JOINTS, :]
    feet_horiz_disp = np.linalg.norm(feet_pos[1:, :, :2] - feet_pos[:-1, :, :2], axis=-1)
    feet_h = feet_pos[:-1, :, 2] - ground_z
    skate_weight = np.clip(2 - 2 ** (feet_h / H_THRESH), 0.0, 1.0)
    return float((feet_horiz_disp * skate_weight).mean() * FPS)


def compute_gpen(pos: np.ndarray, ground_z: float):
    """Joint-position approximation (all 25 joints), NOT the mesh-vertex method -- see module
    docstring. ground_z: this sequence's empirical ground reference (see compute_foot_skate).
    Returns (mean_mm, max_mm, rate)."""
    z_vals = pos[:, :, 2]
    pen = np.maximum(ground_z - z_vals, 0.0)
    frame_max = pen.max(axis=-1)
    rate = float((frame_max > 1e-4).mean())
    max_mm = float(frame_max.max() * 1000.0)
    mask = pen > 1e-4
    mean_mm = float(pen[mask].mean() * 1000.0) if mask.any() else 0.0
    return mean_mm, max_mm, rate


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-opt', type=str, default='options/train_avatarposer_quest2.json')
    parser.add_argument('-ckpt', type=str, default=None,
                         help="Explicit checkpoint path (bypasses find_last_checkpoint, which "
                              "chokes on non-numeric filenames like latest_G.pth).")
    parser.add_argument('-out', type=str, default=None,
                         help="Output JSON path. Defaults to "
                              "results/baselines/avatarposer_quest2_phase_a.json -- ALWAYS pass "
                              "this explicitly when evaluating anything other than the canonical "
                              "from-scratch Phase A checkpoint, or it will silently overwrite "
                              "that baseline (bit us once already, 2026-07-29).")
    args = parser.parse_args()

    opt = option.parse(args.opt, is_train=True)
    opt = option.dict_to_nonedict(opt)

    if args.ckpt:
        init_path_G = args.ckpt
    else:
        _, init_path_G = option.find_last_checkpoint(opt['path']['models'], net_type='G')
    print(f"Loading checkpoint: {init_path_G}")
    if init_path_G is None:
        print("No checkpoint found -- has training run/saved yet?")
        sys.exit(1)

    device = torch.device('cuda' if opt['gpu_ids'] else 'cpu')
    netG = define_G(opt).to(device)
    state = torch.load(init_path_G, map_location=device)
    netG.load_state_dict(state, strict=True)
    netG.eval()

    skeleton_tree = SkeletonTree.from_mjcf(QUEST_HUMANOID_MJCF)

    test_set = define_Dataset(opt['datasets']['test'])
    test_loader = DataLoader(test_set, batch_size=1, shuffle=False, num_workers=0)

    window_size = opt['datasets']['test']['window_size']

    pos_errors_m, rot_errors_deg, vel_errors_m_s = [], [], []
    jitters_m_s3, foot_skates_m_s, success_flags = [], [], []
    gpen_means_mm, gpen_maxes_mm, gpen_rates = [], [], []

    with torch.no_grad():
        for idx, data in enumerate(test_loader):
            L = data['L'].squeeze(0).to(device)       # [T, 54]
            H = data['H'].squeeze(0).numpy()           # [T, 150] GT local 6D rotation
            root_trans = data['P']['root_trans_offset'].squeeze(0).numpy()  # [T+2,3] (pre-drop full length)
            head_trans_global = data['Head_trans_global'].squeeze(0).numpy()  # [T,4,4], aligned with H/L

            T = L.shape[0]
            if T <= window_size + 3:  # +3 for jitter's 3rd-order finite difference
                continue

            # Build all valid sliding windows [N, window_size+1, 54] -> predict last-frame pose.
            N = T - window_size
            windows = torch.stack([L[s:s + window_size + 1] for s in range(N)], dim=0)  # [N, W+1, 54]
            global_orientation, joint_rotation = netG(windows, do_fk=False)  # [N,6], [N,144]

            pred_6d = torch.cat([global_orientation, joint_rotation], dim=-1).cpu().numpy()  # [N,150]
            pred_quat = sixd_to_quat_np(pred_6d.reshape(N, QUEST_NUM_JOINTS, 6))  # [N,25,4] local

            # GT quat for the same predicted frames: H is already frame-0-and-1-dropped (test
            # branch), aligned with L the same way training targets are -- predicted frame i
            # (from window ending at L[i+window_size]) corresponds to H[i+window_size-1].
            gt_quat = sixd_to_quat_np(H.reshape(-1, QUEST_NUM_JOINTS, 6))  # [T,25,4] local
            gt_quat_aligned = gt_quat[window_size - 1: window_size - 1 + N]  # [N,25,4]

            # root_trans_offset is full-length (never frame-dropped in dataset_quest2.py);
            # H/L effectively start 2 frames in (see dataset_quest2.py's test-branch comment).
            trans_aligned = root_trans[2:][window_size - 1: window_size - 1 + N]  # [N,3]
            # Head_trans_global IS aligned with H/L already (both test-branch outputs).
            known_head_pos = head_trans_global[window_size - 1: window_size - 1 + N, :3, 3]  # [N,3]

            # Predicted skeleton, root pinned at the origin via FK -- then rigidly shift it so
            # its own (predicted) head position lands on the known/tracked head position, exactly
            # as a real system would: it only ever knows head/wrist tracked poses, never root.
            sk_pred = SkeletonState.from_rotation_and_root_translation(
                skeleton_tree, torch.from_numpy(pred_quat.astype(np.float32)),
                torch.zeros(N, 3), is_local=True,
            )
            pred_pos_rel = sk_pred.global_translation.numpy()  # [N,25,3], root at origin
            pred_grot = sk_pred.global_rotation.numpy()         # [N,25,4] xyzw, global (post-FK)
            shift = known_head_pos - pred_pos_rel[:, QUEST_HEAD_IDX, :]
            pred_pos = pred_pos_rel + shift[:, None, :]

            # GT world positions/rotations use the true root_trans_offset directly -- this is the
            # label, legitimately known during evaluation (no oracle shortcut on prediction side).
            sk_gt = SkeletonState.from_rotation_and_root_translation(
                skeleton_tree, torch.from_numpy(gt_quat_aligned.astype(np.float32)),
                torch.from_numpy(trans_aligned.astype(np.float32)), is_local=True,
            )
            gt_pos = sk_gt.global_translation.numpy()  # [N,25,3]
            gt_grot = sk_gt.global_rotation.numpy()     # [N,25,4]

            # --- position error (Eg-mpjpe / pos_error) ---
            err = np.linalg.norm(pred_pos - gt_pos, axis=-1)  # [N,25]
            pos_errors_m.append(err.mean())

            # --- success rate (SimXR definition; see caveat in module docstring) ---
            success_flags.append((err[:, QUEST_HEAD_IDX] < 0.5).mean())

            # --- rotation error (MPJRE): geodesic distance between GLOBAL (post-FK) quats ---
            rel = (sRot.from_quat(pred_grot.reshape(-1, 4)).inv()
                   * sRot.from_quat(gt_grot.reshape(-1, 4))).magnitude()
            rot_errors_deg.append(float(np.degrees(rel.mean())))

            # --- velocity error ---
            pred_vel = (pred_pos[1:] - pred_pos[:-1]) * FPS
            gt_vel = (gt_pos[1:] - gt_pos[:-1]) * FPS
            vel_errors_m_s.append(float(np.linalg.norm(pred_vel - gt_vel, axis=-1).mean()))

            # --- jitter (predicted motion only, no GT reference) ---
            pred_jerk = (pred_pos[3:] - 3 * pred_pos[2:-1] + 3 * pred_pos[1:-2] - pred_pos[:-3]) * (FPS ** 3)
            jitters_m_s3.append(float(np.linalg.norm(pred_jerk, axis=-1).mean()))

            # Per-sequence empirical ground reference: quest_humanoid's foot_twist joint is an
            # ankle-twist auxiliary joint (not a toe/sole marker), so it never reaches literal
            # Z=0 even when genuinely planted (verified: min ~0.087m during active walking, see
            # session notes). Use this GT sequence's own minimum foot height as its "on the
            # ground" reference, rather than assuming Z=0 -- self-calibrating, no hardcoded
            # constant, and derived from GT (not predictions) so it isn't circularly affected by
            # the model's own errors.
            ground_z = float(gt_pos[:, FOOT_JOINTS, 2].min())

            # --- foot skate (predicted) ---
            foot_skates_m_s.append(compute_foot_skate(pred_pos, ground_z))

            # --- ground penetration (predicted, joint-approximation -- see docstring) ---
            gp_mean, gp_max, gp_rate = compute_gpen(pred_pos, ground_z)
            gpen_means_mm.append(gp_mean)
            gpen_maxes_mm.append(gp_max)
            gpen_rates.append(gp_rate)

            if (idx + 1) % 100 == 0:
                print(f"  {idx+1}/{len(test_loader)} sequences processed…")

    print(f"\nProcessed {len(pos_errors_m)} sequences")
    metrics = {
        "rot_error_deg": float(np.mean(rot_errors_deg)),
        "pos_error_cm": float(np.mean(pos_errors_m) * 100),
        "vel_error_cm_s": float(np.mean(vel_errors_m_s) * 100),
        "jitter_m_s3": float(np.mean(jitters_m_s3)),
        "foot_skate_cm_s": float(np.mean(foot_skates_m_s) * 100),
        "gpen_mean_mm": float(np.mean(gpen_means_mm)),
        "gpen_max_mm": float(np.mean(gpen_maxes_mm)),
        "gpen_rate": float(np.mean(gpen_rates)),
        "eg_mpjpe_mm": float(np.mean(pos_errors_m) * 1000),
        "success_rate": float(np.mean(success_flags)),
    }
    for k, v in metrics.items():
        print(f"  {k:16s}: {v:.5f}")

    result = {
        "model": "AvatarPoser (Phase A: fit on Quest2 synthetic data, no vision)",
        "date": datetime.date.today().isoformat(),
        "checkpoint": init_path_G,
        "opt_file": args.opt,
        "num_sequences": len(pos_errors_m),
        "metrics": metrics,
        "metric_units": {
            "rot_error_deg": "degrees (geodesic distance, global post-FK rotation, mean per joint, all 25 quest_humanoid joints)",
            "pos_error_cm": "cm (MPJPE-style, all 25 joints, head-anchored reconstruction)",
            "vel_error_cm_s": "cm/s (finite-difference velocity)",
            "jitter_m_s3": "m/s³ (mean jerk of predicted motion, no GT reference)",
            "foot_skate_cm_s": "cm/s (weighted horizontal foot sliding while in contact, ground reference calibrated per-sequence -- see notes)",
            "gpen_mean_mm": "mm (joint-position approximation, NOT mesh-vertex -- quest_humanoid has no mesh asset ported yet; ground reference calibrated per-sequence)",
            "gpen_max_mm": "mm (joint-position approximation, per-sequence ground reference)",
            "gpen_rate": "fraction of frames with any joint below the per-sequence ground reference (joint-position approximation)",
            "eg_mpjpe_mm": "mm (same as pos_error_cm, in SimXR's units, for direct Table-5 comparison)",
            "success_rate": "fraction of frames with head position <0.5m from GT (SimXR Table 5 definition)",
        },
        "notes": [
            "Position reconstruction anchors the predicted skeleton on the known head-tracked "
            "position (Head_trans_global), not oracle root/pelvis translation -- matches how a "
            "real deployed system would work (root/pelvis is never directly tracked).",
            "success_rate is structurally near-100% for any reasonable static kinematic "
            "regressor anchored this way (head is ~6 kinematic hops from the anchor) -- NOT "
            "comparable to SimXR's success rate, which comes from a temporally-extended physics "
            "rollout that can compound errors/fail catastrophically over time. eg_mpjpe_mm/"
            "pos_error_cm is the position metric that's actually informative here.",
            "gpen_* is a joint-position approximation (all 25 joint centers vs floor), not the "
            "true SMPL-H mesh-vertex method results/baselines/avatarposer_w40_baseline.json "
            "uses -- quest_humanoid has no mesh asset in this repo (Phase B work). Comparable to "
            "evaluate_phc_metrics.py's compute_gpen() approximation, not to the baseline JSON.",
            "foot_skate_cm_s and gpen_* use a per-sequence empirical ground reference (each "
            "sequence's own GT minimum foot-joint height), not literal Z=0. Reason: "
            "quest_humanoid's leg chain ends at b_l/r_foot_twist, an ankle-twist auxiliary "
            "joint, not a toe/sole ground-contact joint like SMPL's L_Toe/R_Toe -- verified "
            "empirically that this joint never approaches Z=0 even during active walking (40 "
            "random test sequences: min height 0.046m, median 0.073m). Without this "
            "recalibration, foot_skate's H_THRESH=0.033 (tuned for SMPL's toe-at-the-floor "
            "convention) never triggers real contact detection, and gpen's Z<0 check almost "
            "never fires -- both were silently near-zero for this reason, not because the "
            "predicted motion is unusually clean.",
            "SimXR's own Headset-only ablation (Table 5, extracted via AI web-fetch that also "
            "produced unrelated garbled filler text in the same response -- treat as a working "
            "reference, not verified ground truth): success 73.0%, Eg-mpjpe 118.9mm.",
        ],
    }
    out_path = args.out or os.path.join(REPO_ROOT, 'results', 'baselines', 'avatarposer_quest2_phase_a.json')
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved results to {out_path}")


if __name__ == '__main__':
    main()
