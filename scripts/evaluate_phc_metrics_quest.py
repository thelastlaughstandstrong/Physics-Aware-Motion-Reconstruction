"""
Computes the same 8-metric set as main_test_avatarposer_quest2.py's Phase A eval, but from a
real PHC-simulated rollout (Phase B) -- reads the phc_positions_for_eval.pkl dump that
im_amp_players.py writes generically for any humanoid_type (pred_pos_all/ref_pos_all are plain
[T, num_bodies, 3] arrays, no SMPL-specific plumbing), so no PHC-side changes were needed.

Formulas copied verbatim from main_test_avatarposer_quest2.py so Phase A vs Phase B numbers are
directly comparable: same foot joints, same H_THRESH, same per-sequence empirical ground
reference, same finite-difference jitter/velocity, same joint-position-approximation gpen.

Frame note: quest_humanoid.yaml has has_upright_start: False, so PHC never applies the SMPL-style
upright rotation to quest -- pred_pos_all/pred_rot_all are already in the same raw MJCF frame
Phase A's SkeletonTree FK uses. No upright-undo transform needed here (unlike the SMPL pipeline's
evaluate_phc_metrics.py).

Usage: python scripts/evaluate_phc_metrics_quest.py --pos_dump <path to phc_positions_for_eval.pkl>
"""
import argparse
import json

import joblib
import numpy as np
from scipy.spatial.transform import Rotation as sRot

FOOT_JOINTS = [21, 24]  # b_l_foot_twist, b_r_foot_twist
HEAD_IDX = 6            # b_head
FPS = 30.0
H_THRESH = 0.033
# The real Isaac Gym ground plane for quest_humanoid: normal=(0,0,1) (Z-up), distance=0 (see
# humanoid.py's _create_ground_plane, neither overridden for this robot) -- verified against
# actual rollout data too (foot-joint z clusters near 0 across the test set). Matches the SMPL
# pipeline's evaluate_phc_metrics.py, which also checks mesh vertices against the true Z=0 floor
# rather than an empirical per-sequence estimate.
GROUND_Z = 0.0


def compute_foot_skate(pos: np.ndarray, ground_z: float) -> float:
    feet_pos = pos[:, FOOT_JOINTS, :]
    feet_horiz_disp = np.linalg.norm(feet_pos[1:, :, :2] - feet_pos[:-1, :, :2], axis=-1)
    feet_h = feet_pos[:-1, :, 2] - ground_z
    skate_weight = np.clip(2 - 2 ** (feet_h / H_THRESH), 0.0, 1.0)
    return float((feet_horiz_disp * skate_weight).mean() * FPS)


def compute_gpen(pos: np.ndarray, ground_z: float):
    z_vals = pos[:, :, 2]
    pen = np.maximum(ground_z - z_vals, 0.0)
    frame_max = pen.max(axis=-1)
    rate = float((frame_max > 1e-4).mean())
    max_mm = float(frame_max.max() * 1000.0)
    mask = pen > 1e-4
    mean_mm = float(pen[mask].mean() * 1000.0) if mask.any() else 0.0
    return mean_mm, max_mm, rate


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pos_dump", required=True)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    d = joblib.load(args.pos_dump)
    pred_pos_all, ref_pos_all = d["pred_pos_all"], d["ref_pos_all"]
    pred_rot_all, gt_rot_all = d["pred_rot_all"], d["gt_rot_all"]
    motion_keys = d["motion_keys"]

    pos_errors_m, rot_errors_deg, vel_errors_m_s = [], [], []
    jitters_m_s3, foot_skates_m_s, success_flags = [], [], []
    gpen_means_mm, gpen_maxes_mm, gpen_rates = [], [], []

    for i, key in enumerate(motion_keys):
        pred_pos = pred_pos_all[i]  # [T,25,3]
        gt_pos = ref_pos_all[i]
        pred_grot = pred_rot_all[i] if pred_rot_all[i].ndim == 2 else pred_rot_all[i]
        gt_grot = gt_rot_all[i]
        T = pred_pos.shape[0]
        if T <= 3:
            continue

        err = np.linalg.norm(pred_pos - gt_pos, axis=-1)  # [T,25]
        pos_errors_m.append(err.mean())
        success_flags.append((err[:, HEAD_IDX] < 0.5).mean())

        rel = (sRot.from_quat(pred_grot.reshape(-1, 4)).inv()
               * sRot.from_quat(gt_grot.reshape(-1, 4))).magnitude()
        rot_errors_deg.append(float(np.degrees(rel.mean())))

        pred_vel = (pred_pos[1:] - pred_pos[:-1]) * FPS
        gt_vel = (gt_pos[1:] - gt_pos[:-1]) * FPS
        vel_errors_m_s.append(float(np.linalg.norm(pred_vel - gt_vel, axis=-1).mean()))

        pred_jerk = (pred_pos[3:] - 3 * pred_pos[2:-1] + 3 * pred_pos[1:-2] - pred_pos[:-3]) * (FPS ** 3)
        jitters_m_s3.append(float(np.linalg.norm(pred_jerk, axis=-1).mean()))

        # compute_foot_skate's height-gating (H_THRESH) is calibrated against each sequence's
        # own typical foot-joint contact height, not the true floor -- the foot joints (twist
        # joints, not sole vertices) sit a few cm above Z=0 even in solid ground contact, so
        # gating against the literal floor makes every frame read as "high up" and the skate
        # weight collapses to ~0 (confirmed empirically). gpen, in contrast, should measure
        # against the real physical floor -- see GROUND_Z above.
        per_seq_ground = float(gt_pos[:, FOOT_JOINTS, 2].min())
        foot_skates_m_s.append(compute_foot_skate(pred_pos, per_seq_ground))
        gp_mean, gp_max, gp_rate = compute_gpen(pred_pos, GROUND_Z)
        gpen_means_mm.append(gp_mean)
        gpen_maxes_mm.append(gp_max)
        gpen_rates.append(gp_rate)

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
        "success_rate_frame_head": float(np.mean(success_flags)),
        "success_rate_phc_survival": float(d["success_rate"]),
    }

    print(f"Processed {len(pos_errors_m)} sequence(s): {motion_keys}")
    for k, v in metrics.items():
        print(f"  {k:28s}: {v:.5f}")

    if args.out:
        with open(args.out, "w") as f:
            json.dump({"metrics": metrics, "motion_keys": motion_keys, "num_sequences": len(pos_errors_m)}, f, indent=2)
        print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
