"""
Evaluate PHC-corrected motion using AvatarPoser's metrics.

After running PHC evaluation with im_eval=True on AvatarPoser-converted sequences,
this script computes rot_error, pos_error, vel_error, jitter, foot_skate, and ground
penetration — the same metrics as the AvatarPoser baseline.

Usage (run from AvatarPoser repo root):

    # Step 1 — convert all test sequences to PHC format:
    python scripts/convert_avatarposer_to_phc.py --seq_idx $(seq 0 535 | tr '\\n' ' ') \\
           --out phc/sample_data/avatarposer_all.pkl

    # Step 2 — run PHC eval (from phc/ directory):
    cd phc && python phc/run_hydra.py learning=im_mcp exp_name=phc_kp_mcp_iccv test=True \\
      env=env_im_getup_mcp robot=smpl_humanoid robot.freeze_hand=True robot.box_body=False \\
      env.z_activation=relu \\
      "env.motion_file=sample_data/avatarposer_all.pkl" \\
      "env.models=['output/HumanoidIm/phc_kp_pnn_iccv/Humanoid.pth']" \\
      env.num_envs=16 env.obs_v=7 headless=True epoch=-1 im_eval=True
    # → writes output/HumanoidIm/phc_kp_mcp_iccv/phc_positions_for_eval.pkl

    # Step 3 — compute AvatarPoser metrics:
    python scripts/evaluate_phc_metrics.py \\
      --pos_dump phc/output/HumanoidIm/phc_kp_mcp_iccv/phc_positions_for_eval.pkl \\
      --motion_file phc/sample_data/avatarposer_all.pkl

Notes
-----
- pred_pos_all[i]: [T_i, 24, 3] — PHC SIMULATED positions, Mujoco body order, PHC upright_start
  frame (Z-up, floor at Z=0).
- For metrics that need GT comparison (pos_error, vel_error), GT body params are loaded from
  the AvatarPoser test dataset and converted to PHC frame using FK — matching the reference
  motion coordinate system exactly.
- rot_error requires local joint rotations which are not directly available from rigid body
  positions alone; it is computed from PHC's reference motion (the AvatarPoser prediction it
  was tracking) vs the true GT, as an upper-bound indicator.
"""

import argparse
import os
import re
import sys

import joblib
import numpy as np
import torch
from scipy.spatial.transform import Rotation as sRot
from torch.utils.data import DataLoader

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PHC_ROOT  = os.path.join(REPO_ROOT, "phc")
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, PHC_ROOT)

from data.select_dataset import define_Dataset
from utils import utils_option as option
from poselib.poselib.skeleton.skeleton3d import SkeletonState, SkeletonTree
from smpl_sim.smpllib.smpl_joint_names import SMPL_BONE_ORDER_NAMES, SMPL_MUJOCO_NAMES
from smpl_sim.smpllib.smpl_local_robot import SMPL_Robot as LocalRobot
from scripts.convert_avatarposer_to_phc import (
    ROBOT_CFG, SMPL_2_MUJOCO, build_skeleton_tree
)

# ── joint index mappings ──────────────────────────────────────────────────────
# SMPL_2_MUJOCO[mujoco_idx] = smpl_idx  (mujoco body i -> smpl joint index)
# MUJOCO_2_SMPL_REORDER[smpl_idx] = mujoco_idx  (smpl joint i -> mujoco body index)
MUJOCO_2_SMPL_REORDER = [0] * 24
for _muj, _smpl in enumerate(SMPL_2_MUJOCO):
    MUJOCO_2_SMPL_REORDER[_smpl] = _muj

# Foot toe joints in Mujoco body order (used for foot skate and gpen fallback)
L_TOE_MUJ = MUJOCO_2_SMPL_REORDER[10]   # SMPL joint 10 = L_Toe
R_TOE_MUJ = MUJOCO_2_SMPL_REORDER[11]   # SMPL joint 11 = R_Toe
FOOT_JOINTS_MUJ = [L_TOE_MUJ, R_TOE_MUJ]

NUM_SMPL_JOINTS = 22  # AvatarPoser uses 22 joints (no L_Hand / R_Hand)

_UPRIGHT_ROT_INV = sRot.from_quat([0.5, 0.5, 0.5, 0.5]).inv()


def parse_seq_idx(motion_key: str) -> int:
    """Extract test-set index from a key like 'pred_0042_BioMotionLab...'."""
    m = re.match(r"(?:pred|gt)_(\d+)_", motion_key)
    if m:
        return int(m.group(1))
    raise ValueError(f"Cannot parse seq index from key: {motion_key!r}")


def gt_positions_in_phc_frame(body_parms_gt: dict,
                               skeleton_tree: SkeletonTree,
                               skeleton_tree_upright: SkeletonTree,
                               source_fps: int = 60) -> np.ndarray:
    """
    Convert AvatarPoser GT body params → joint positions in PHC upright_start frame.
    Returns [T, 24, 3] in Mujoco body order.

    Replicates exactly what PHC's motion_lib does for the reference motion:
      1. root_trans_offset  — computed with the non-upright skeleton_tree (same as the PKL converter)
      2. pose_quat_global   — upright_start-rotated global rotations (same as the PKL converter)
      3. FK positions       — using PHC's upright skeleton_tree with is_local=False
    The upright skeleton_tree has joint offsets in the upright (Z-up, +X facing) frame, so FK
    with upright global rotations gives correct world-space positions that match Isaac Gym.
    """
    root_orient = body_parms_gt["root_orient"]
    pose_body   = body_parms_gt["pose_body"]
    trans       = body_parms_gt["trans"]
    if isinstance(root_orient, torch.Tensor):
        root_orient = root_orient.cpu().numpy()
    if isinstance(pose_body, torch.Tensor):
        pose_body   = pose_body.cpu().numpy()
    if isinstance(trans, torch.Tensor):
        trans       = trans.cpu().numpy()

    T = root_orient.shape[0]
    pose_aa = np.concatenate([root_orient, pose_body, np.zeros((T, 6))], axis=1)
    skip    = max(1, source_fps // 30)
    pose_aa = pose_aa[::skip]
    trans   = trans[::skip]
    N       = pose_aa.shape[0]

    # SMPL → Mujoco order, axis-angle → local quaternion
    pose_aa_mj = pose_aa.reshape(N, 24, 3)[:, SMPL_2_MUJOCO]
    pose_quat  = sRot.from_rotvec(pose_aa_mj.reshape(-1, 3)).as_quat().reshape(N, 24, 4)

    # root_trans_offset with non-upright skeleton (same as converter / PKL)
    root_trans_offset = torch.from_numpy(trans) + skeleton_tree.local_translation[0]

    # FK with non-upright skeleton → global rotations in SMPL frame
    sk_state = SkeletonState.from_rotation_and_root_translation(
        skeleton_tree, torch.from_numpy(pose_quat), root_trans_offset, is_local=True
    )

    # Apply upright_start rotation (same as converter)
    pose_quat_global = (
        sRot.from_quat(sk_state.global_rotation.reshape(-1, 4).numpy()) * _UPRIGHT_ROT_INV
    ).as_quat().reshape(N, -1, 4)

    # FK with PHC's upright skeleton_tree → correct world-space positions
    sk_state_up = SkeletonState.from_rotation_and_root_translation(
        skeleton_tree_upright,
        torch.from_numpy(pose_quat_global),
        root_trans_offset,
        is_local=False,
    )
    return sk_state_up.global_translation.numpy(), pose_quat_global  # [N,24,3], [N,24,4]


def compute_metrics(pred: np.ndarray, gt: np.ndarray, fps: int = 30) -> dict:
    """
    Compute AvatarPoser-style metrics for one sequence.
    Both pred and gt: [T, 22, 3] in any consistent world frame (metres, Z-up).
    fps: frame rate of the data (30 for PHC output).
    """
    pred_t = torch.from_numpy(pred.astype(np.float32))   # [T, 22, 3]
    gt_t   = torch.from_numpy(gt.astype(np.float32))     # [T, 22, 3]

    pos_error = torch.mean(
        torch.sqrt(torch.sum(torch.square(pred_t - gt_t), dim=-1))
    ).item()

    pred_vel = (pred_t[1:] - pred_t[:-1]) * fps
    gt_vel   = (gt_t[1:] - gt_t[:-1]) * fps
    vel_error = torch.mean(
        torch.sqrt(torch.sum(torch.square(pred_vel - gt_vel), dim=-1))
    ).item()

    pred_jerk = (
        pred_t[3:] - 3 * pred_t[2:-1] + 3 * pred_t[1:-2] - pred_t[:-3]
    ) * (fps ** 3)
    jitter = torch.mean(
        torch.sqrt(torch.sum(torch.square(pred_jerk), dim=-1))
    ).item()

    return {
        "pos_error_m": pos_error,
        "vel_error_m_s": vel_error,
        "jitter_m_s3": jitter,
    }


def compute_foot_skate(pred_full: np.ndarray, fps: int = 30,
                        H_THRESH: float = 0.033) -> float:
    """
    Foot skate metric using PHC toe joint positions.
    pred_full: [T, 24, 3] Mujoco body order.
    """
    feet_pos = torch.from_numpy(pred_full[:, FOOT_JOINTS_MUJ, :].astype(np.float32))  # [T, 2, 3]
    feet_horiz_disp = torch.sqrt(
        torch.sum(torch.square(feet_pos[1:, :, :2] - feet_pos[:-1, :, :2]), dim=-1)
    )
    feet_h = feet_pos[:-1, :, 2]
    skate_weight = torch.clamp(2 - 2 ** (feet_h / H_THRESH), min=0.0, max=1.0)
    return (torch.mean(feet_horiz_disp * skate_weight) * fps).item()


def compute_gpen(pred_full: np.ndarray, ground_z: float = 0.0) -> tuple:
    """
    Approximate ground penetration using rigid body positions.
    PHC physics enforces contact, so penetration should be near zero.
    pred_full: [T, 24, 3] Mujoco body order.
    Returns (mean_mm, max_mm, rate).
    """
    z_vals = pred_full[:, :, 2]                           # [T, 24]
    pen = np.maximum(ground_z - z_vals, 0.0)              # [T, 24] depth below floor
    frame_max = pen.max(axis=-1)                           # [T]
    pen_rate = float((frame_max > 1e-4).mean())
    max_mm   = float(frame_max.max() * 1000.0)
    mask = pen > 1e-4
    mean_mm  = float(pen[mask].mean() * 1000.0) if mask.any() else 0.0
    return mean_mm, max_mm, pen_rate


def compute_mpjre(pred_rot: np.ndarray, gt_rot: np.ndarray) -> float:
    """
    Mean per-joint rotation error (geodesic distance) in degrees.
    pred_rot, gt_rot: [T, 24, 4] global quaternions (xyzw), Mujoco body order.
    """
    T, J, _ = pred_rot.shape
    pred_r = sRot.from_quat(pred_rot.reshape(-1, 4))
    gt_r   = sRot.from_quat(gt_rot.reshape(-1, 4))
    rel    = (pred_r.inv() * gt_r).magnitude()          # geodesic angle in radians [T*J]
    return float(np.degrees(rel.mean()))


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pos_dump", required=True,
                        help="Path to phc_positions_for_eval.pkl (written by PHC im_eval).")
    parser.add_argument("--motion_file", default=None,
                        help="PHC motion pkl used during eval (for AvatarPoser MPJRE baseline).")
    parser.add_argument("--gt_pkl", default=None,
                        help="phc_bio_test.pkl — use instead of AvatarPoser test dataset for GT. "
                             "Keys must match motion_keys in pos_dump.")
    parser.add_argument("--opt", default="options/test_avatarposer.json",
                        help="AvatarPoser options JSON (only needed when --gt_pkl is not set).")
    parser.add_argument("--fps", type=int, default=30,
                        help="Frame rate of PHC output (default 30).")
    args = parser.parse_args()

    os.chdir(REPO_ROOT)

    print("Loading PHC simulation positions …")
    phc_dump    = joblib.load(args.pos_dump)
    motion_data = joblib.load(args.motion_file) if args.motion_file else {}
    pred_pos_all   = phc_dump["pred_pos_all"]
    ref_pos_all    = phc_dump.get("ref_pos_all", None)
    pred_rot_all   = phc_dump.get("pred_rot_all", None)
    gt_rot_all     = phc_dump.get("gt_rot_all", None)
    motion_keys    = phc_dump["motion_keys"]
    success_rate   = phc_dump.get("success_rate", float("nan"))
    terminate_hist = phc_dump.get("terminate_hist", [False] * len(motion_keys))

    print(f"  {len(pred_pos_all)} sequences  |  success_rate: {success_rate:.4f}")

    print("Building SMPL skeleton trees …")
    smpl_robot    = LocalRobot(ROBOT_CFG, data_dir=os.path.join(PHC_ROOT, "data/smpl"))
    skeleton_tree = build_skeleton_tree(smpl_robot)

    upright_mjcf  = os.path.join(PHC_ROOT, "phc/data/assets/mjcf/smpl_0_humanoid.xml")
    skeleton_tree_upright = SkeletonTree.from_mjcf(upright_mjcf)
    print(f"  upright skeleton loaded from {upright_mjcf}")

    # ── GT source: phc_test.pkl (preferred) OR AvatarPoser test dataset ─────
    # If --gt_pkl not given but --motion_file is, use motion_file as GT source
    # (they're the same file in the distill eval workflow).
    gt_pkl_path = args.gt_pkl or (args.motion_file if args.motion_file else None)

    if gt_pkl_path:
        print(f"Loading GT from {gt_pkl_path} …")
        gt_pkl = joblib.load(gt_pkl_path)
        # Build key → body params map directly from the PHC-format pkl
        # Each entry already has pose_aa (axis-angle) and root_trans_offset
        key_to_gt = {}
        for k, entry in gt_pkl.items():
            # pose_aa: [N, 72] at 30fps, already downsampled; trans: root_trans_offset
            key_to_gt[k] = entry
        print(f"  {len(key_to_gt)} GT sequences loaded")
        idx_to_data = None  # not used in gt_pkl mode
    else:
        print("Loading AvatarPoser test dataset …")
        opt = option.parse(args.opt, is_train=True)
        opt = option.dict_to_nonedict(opt)
        test_set    = define_Dataset(opt["datasets"]["test"])
        test_loader = DataLoader(test_set, batch_size=1, shuffle=False, num_workers=0)
        print("Indexing test dataset …")
        idx_to_data = {}
        for seq_idx, test_data in enumerate(test_loader):
            idx_to_data[seq_idx] = {
                "filepath": test_data["filepath"][0],
                "gt": {k: v.squeeze(0) for k, v in test_data["P"].items()},
            }
            if seq_idx % 50 == 0:
                print(f"  indexed {seq_idx}/{len(test_set)}", end="\r")
        print(f"  indexed {len(idx_to_data)} sequences           ")
        key_to_gt = None

    # ── Compute metrics per sequence ──────────────────────────────────────────
    pos_errors, vel_errors, rot_errors, ap_rot_errors, jitters, foot_skates = [], [], [], [], [], []
    gpen_means, gpen_maxes, gpen_rates = [], [], []

    # Tracking quality: mean |PHC_sim - AvatarPoser_ref| across all sequences.
    # Should match PHC's reported mpjpe_g (~42mm for seq 0).
    if ref_pos_all is not None:
        tracking_errors = []
        for p, r in zip(pred_pos_all, ref_pos_all):
            T = min(p.shape[0], r.shape[0])
            tracking_errors.append(np.mean(np.linalg.norm(p[:T] - r[:T], axis=-1)) * 1000)
        print(f"  PHC tracking quality (mean |sim-ref|): {np.mean(tracking_errors):.2f} mm  (PHC mpjpe_g)")

    skipped = 0
    for i, (key, pred_full) in enumerate(zip(motion_keys, pred_pos_all)):

        # ── Get GT positions in PHC frame ─────────────────────────────────────
        if key_to_gt is not None:
            # gt_pkl mode: GT already processed at 30fps with upright_start
            if key not in key_to_gt:
                print(f"  [skip] key not in gt_pkl: {key}")
                skipped += 1
                continue
            entry = key_to_gt[key]
            # Reconstruct global positions via FK from pose_quat_global
            pqg = entry["pose_quat_global"]  # [N, 24, 4] ndarray
            rto = entry["root_trans_offset"]  # [N, 3] Tensor or ndarray
            if isinstance(rto, torch.Tensor):
                rto = rto
            else:
                rto = torch.from_numpy(rto.astype(np.float32))
            pqg_t = torch.from_numpy(pqg.astype(np.float32))
            sk_state = SkeletonState.from_rotation_and_root_translation(
                skeleton_tree_upright, pqg_t, rto, is_local=False
            )
            gt_pos_mj  = sk_state.global_translation.numpy()   # [N, 24, 3]
            gt_rot_mj  = pqg                                    # [N, 24, 4]
        else:
            # AvatarPoser dataset mode
            try:
                seq_idx = parse_seq_idx(key)
            except ValueError as e:
                print(f"  [skip] {e}")
                skipped += 1
                continue
            if seq_idx not in idx_to_data:
                print(f"  [skip] seq {seq_idx} not in test set")
                skipped += 1
                continue
            gt_data              = idx_to_data[seq_idx]
            gt_pos_mj, gt_rot_mj = gt_positions_in_phc_frame(gt_data["gt"], skeleton_tree, skeleton_tree_upright, source_fps=60)

        T_pred = pred_full.shape[0]
        T_gt   = gt_pos_mj.shape[0]
        T      = min(T_pred, T_gt)

        if T < 4:
            skipped += 1
            continue

        # Reorder from Mujoco → SMPL, keep first 22 joints for error metrics
        pred_smpl = pred_full[:T, MUJOCO_2_SMPL_REORDER, :][:, :NUM_SMPL_JOINTS, :]  # [T, 22, 3]
        gt_smpl   = gt_pos_mj[:T, MUJOCO_2_SMPL_REORDER, :][:, :NUM_SMPL_JOINTS, :]  # [T, 22, 3]

        m = compute_metrics(pred_smpl, gt_smpl, fps=args.fps)
        pos_errors.append(m["pos_error_m"])
        vel_errors.append(m["vel_error_m_s"])
        jitters.append(m["jitter_m_s3"])

        if pred_rot_all is not None:
            T_r = min(pred_rot_all[i].shape[0], gt_rot_mj.shape[0])
            rot_errors.append(compute_mpjre(pred_rot_all[i][:T_r], gt_rot_mj[:T_r]))

        # AvatarPoser MPJRE: predicted global rotations (from motion pkl) vs GT
        if key in motion_data:
            ap_rot = motion_data[key]["pose_quat_global"]  # [N, 24, 4]
            T_ap   = min(ap_rot.shape[0], gt_rot_mj.shape[0])
            ap_rot_errors.append(compute_mpjre(ap_rot[:T_ap], gt_rot_mj[:T_ap]))

        foot_skates.append(compute_foot_skate(pred_full[:T], fps=args.fps))
        gp_mean, gp_max, gp_rate = compute_gpen(pred_full[:T])
        gpen_means.append(gp_mean)
        gpen_maxes.append(gp_max)
        gpen_rates.append(gp_rate)

        if (i + 1) % 20 == 0:
            print(f"  {i+1}/{len(motion_keys)} sequences processed …", end="\r")

    # Load pre-sim q_ref stats if available
    qref_stats = {}
    qref_log = os.path.join(os.path.dirname(args.pos_dump), "qref_gpen.log")
    if os.path.exists(qref_log):
        with open(qref_log) as f:
            for line in f:
                for key in ("qref_mpjpe (cm)", "gpen_mean  (mm)", "gpen_max   (mm)", "gpen_rate"):
                    if key in line:
                        qref_stats[key.strip()] = float(line.split(":")[-1].strip())

    lines = []
    lines.append(f"\nProcessed {len(pos_errors)} sequences  ({skipped} skipped)")
    lines.append("")
    lines.append("=" * 60)
    lines.append("PHC-corrected metrics (using AvatarPoser metric definitions)")
    lines.append("=" * 60)
    lines.append(f"  success_rate    : {success_rate:.4f}")
    if pos_errors:
        lines.append(f"  MPJPE (cm)      : {np.mean(pos_errors) * 100:.5f}")
        if "qref_mpjpe (cm)" in qref_stats:
            lines.append(f"  qref_MPJPE (cm) : {qref_stats['qref_mpjpe (cm)']:.5f}  [pre-sim kinematic]")
        lines.append(f"  MPJVE (cm/s)    : {np.mean(vel_errors) * 100:.5f}")
        if rot_errors:
            lines.append(f"  MPJRE (deg)     : {np.mean(rot_errors):.5f}")
        lines.append(f"  jitter (m/s³)   : {np.mean(jitters):.5f}")
        lines.append(f"  foot_skate(cm/s): {np.mean(foot_skates) * 100:.5f}")
        if qref_stats:
            lines.append(f"  gpen_mean (mm)  : {qref_stats.get('gpen_mean  (mm)', 0.0):.5f}  [pre-sim q_ref]")
            lines.append(f"  gpen_max  (mm)  : {qref_stats.get('gpen_max   (mm)', 0.0):.5f}  [pre-sim q_ref]")
            lines.append(f"  gpen_rate       : {qref_stats.get('gpen_rate', 0.0):.5f}  [pre-sim q_ref]")
        else:
            lines.append(f"  gpen_mean (mm)  : {np.mean(gpen_means):.5f}")
            lines.append(f"  gpen_max (mm)   : {np.mean(gpen_maxes):.5f}")
            lines.append(f"  gpen_rate       : {np.mean(gpen_rates):.5f}")
    lines.append("")
    lines.append("Baseline (AvatarPoser kinematic, for reference):")
    lines.append("  MPJPE (cm)      : 2.51398")
    lines.append("  MPJVE (cm/s)    : 24.11242")
    if ap_rot_errors:
        lines.append(f"  MPJRE (deg)     : {np.mean(ap_rot_errors):.5f}")
    lines.append("  jitter (m/s³)   : 1357.28")
    lines.append("  foot_skate(cm/s): 33.67")
    lines.append("  gpen_mean (mm)  : 23.79")
    lines.append("  gpen_rate       : 0.654")

    output = "\n".join(lines)
    print(output)

    log_path = os.path.join(os.path.dirname(args.pos_dump), "phc_eval.log")
    with open(log_path, "w") as f:
        f.write(output + "\n")
    print(f"\nResults saved to {log_path}")


if __name__ == "__main__":
    main()
