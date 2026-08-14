"""
Re-score a Quest2 rollout against GROUND TRUTH instead of against the reference it tracked.

WHY THIS EXISTS. The naive baseline (frozen phc_prim_quest teacher tracking AvatarPoser's own
predicted motion, no distillation student) is run with env.motion_file pointing at the PREDICTED
pkls, so everything PHC dumps as "ground truth" in phc_positions_for_eval.pkl -- ref_pos_all,
gt_rot_all, and the im_eval success_rate -- is actually the prediction. Scoring that dump with
evaluate_phc_metrics_quest.py as-is answers "did physics track AvatarPoser's guess?", which is
NOT the question the distilled rows in results_comparison.md answer: those runs put the real
test_seg_motion pkls in env.motion_file, so their numbers are already sim-vs-truth. Comparing the
two directly would flatter the naive baseline by exactly the amount AvatarPoser's prediction is
wrong.

This script keeps the SIMULATED trajectory from the dump (pred_pos_all/pred_rot_all -- that part
is a genuine physics rollout either way) and swaps the reference side for the true GT pkls,
matched per sequence by motion key, then computes the same metric set as
evaluate_phc_metrics_quest.py so the result drops straight into the comparison table.

Usage:
    python scripts/score_naive_baseline_vs_gt.py \
      --pos_dump output/HumanoidIm/phc_prim_quest/phc_positions_for_eval.pkl \
      --gt_dir   data/quest2_synthetic/test_seg_motion \
      --out      results/baselines/quest2_naive_baseline_vs_gt.json
"""
import argparse
import json
import os
import sys

import joblib
import numpy as np
import torch
from scipy.spatial.transform import Rotation as sRot

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PHC_ROOT = os.path.join(REPO_ROOT, "phc")
for _p in (REPO_ROOT, PHC_ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from poselib.poselib.skeleton.skeleton3d import SkeletonState, SkeletonTree
from models.quest_body_model import QUEST_HUMANOID_MJCF, QUEST_HEAD_IDX

FPS = 30.0
FOOT_JOINTS = [21, 24]
H_THRESH = 0.033
SUCCESS_THRESH_M = 0.5


def gt_positions_and_rot(entry, tree):
    """FK the GT pkl onto world positions + global rotations, same construction
    data/dataset_quest2.py uses."""
    pqg = entry["pose_quat_global"]
    rt = entry["root_trans_offset"]
    pqg = pqg.numpy() if isinstance(pqg, torch.Tensor) else np.asarray(pqg)
    rt = rt.numpy() if isinstance(rt, torch.Tensor) else np.asarray(rt)
    st = SkeletonState.from_rotation_and_root_translation(
        tree, torch.from_numpy(pqg.astype(np.float32)),
        torch.from_numpy(rt.astype(np.float32)), is_local=False)
    return st.global_translation.numpy(), pqg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pos_dump", required=True)
    ap.add_argument("--gt_dir", default="data/quest2_synthetic/test_seg_motion")
    ap.add_argument("--out", required=True)
    ap.add_argument("--label", default="naive_baseline_quest2")
    ap.add_argument("--gt_offset", type=int, default=42,
                    help="GT frame index that the rollout's frame 0 corresponds to. DERIVED, not "
                         "tuned: (a) 41 from the prediction pipeline -- dataset_quest2's test "
                         "branch drops 2 raw frames (L[i] = raw frame i+2) and "
                         "convert_avatarposer_quest2_to_phc predicts the window's (W-1)th element, "
                         "so predicted frame n = raw frame n+W+1 = n+41 at window_size=40; plus "
                         "(b) 1 for a lag in PHC's own dump, established independently on the "
                         "Phase B artifacts, where the teacher tracked GROUND TRUTH directly (no "
                         "windowing at all) and still aligns best at offset 1 -- 23.6mm there vs "
                         "37.1mm at offset 0, matching Phase B's published 21.81mm. Scoring at "
                         "offset 0 instead reports ~568mm, which is temporal misalignment (a "
                         "walker covers ~1.5m in 42 frames), not tracking error.")
    args = ap.parse_args()

    d = joblib.load(args.pos_dump)
    tree = SkeletonTree.from_mjcf(QUEST_HUMANOID_MJCF)

    keys = d["motion_keys"]
    pred_pos_all, pred_rot_all = d["pred_pos_all"], d["pred_rot_all"]
    # PHC terminated each episode on distance-to-the-TRACKED-reference (the prediction here), so
    # a terminated clip's pred_pos array is truncated. Scoring only the frames it survived would
    # flatter it -- an early fall would be graded on the good frames before the fall. A clip
    # counts as success only if it neither terminated NOR drifted >0.5m from GT.
    terminate_hist = d.get("terminate_hist", [False] * len(keys))

    rot_errors, pos_errors_m, vel_errors_m_s = [], [], []
    jitters, foot_skates, success_flags, survival_flags = [], [], [], []
    per_activity = {}
    missing = 0

    for i, key in enumerate(keys):
        base = os.path.basename(str(key))
        gt_path = os.path.join(args.gt_dir, base if base.endswith(".pkl") else base + ".pkl")
        if not os.path.exists(gt_path):
            missing += 1
            continue
        (_, entry), = joblib.load(gt_path).items()
        gt_pos, gt_grot = gt_positions_and_rot(entry, tree)

        pred_pos = np.asarray(pred_pos_all[i])
        pred_grot = np.asarray(pred_rot_all[i])
        off = args.gt_offset
        gt_pos, gt_grot = gt_pos[off:], gt_grot[off:]
        T = min(len(pred_pos), len(gt_pos))
        if T < 3:
            continue
        pred_pos, gt_pos = pred_pos[:T], gt_pos[:T]
        pred_grot, gt_grot = pred_grot[:T], gt_grot[:T]

        err = np.linalg.norm(pred_pos - gt_pos, axis=-1)                     # [T,25]
        pos_errors_m.append(float(err.mean()))
        # frame_head: head stays within 0.5m; phc_survival: EVERY joint every frame within 0.5m
        # (PHC's own im_eval definition -- one bad frame fails the whole clip).
        success_flags.append(float((err[:, QUEST_HEAD_IDX] < SUCCESS_THRESH_M).mean()))
        terminated = bool(terminate_hist[i]) if i < len(terminate_hist) else False
        truncated = T < len(gt_pos) - 2
        survival_flags.append(float((not terminated) and (not truncated)
                                     and err.max() < SUCCESS_THRESH_M))

        rot_err = (sRot.from_quat(pred_grot.reshape(-1, 4))
                   * sRot.from_quat(gt_grot.reshape(-1, 4)).inv()).magnitude()
        rot_errors.append(float(np.degrees(rot_err).mean()))

        pred_vel = (pred_pos[1:] - pred_pos[:-1]) * FPS
        gt_vel = (gt_pos[1:] - gt_pos[:-1]) * FPS
        vel_errors_m_s.append(float(np.linalg.norm(pred_vel - gt_vel, axis=-1).mean()))

        acc = (pred_pos[2:] - 2 * pred_pos[1:-1] + pred_pos[:-2]) * (FPS ** 2)
        jerk = (acc[1:] - acc[:-1]) * FPS
        jitters.append(float(np.linalg.norm(jerk, axis=-1).mean()))

        ground = float(gt_pos[:, FOOT_JOINTS, 2].min())
        fp = pred_pos[:, FOOT_JOINTS, :]
        contact = (fp[:-1, :, 2] - ground) < H_THRESH
        horiz = np.linalg.norm(fp[1:, :, :2] - fp[:-1, :, :2], axis=-1) * FPS
        if contact.any():
            foot_skates.append(float(horiz[contact].mean()))

        act = base.split(".")[1] if "." in base else "unknown"
        per_activity.setdefault(act, []).append(survival_flags[-1])

    res = {
        "label": args.label,
        "scored_against": "ground truth (test_seg_motion), NOT the tracked reference",
        "num_sequences": len(pos_errors_m),
        "missing_gt": missing,
        # NOT identical to the distilled rows' phc_survival, and the difference is worth stating
        # when quoting it: those runs had GT in env.motion_file, so PHC terminated them against
        # GT live during the rollout. Here PHC terminated against the prediction and GT is
        # applied afterwards, so this is "survived its own reference AND stayed within 0.5m of
        # truth" rather than a like-for-like rerun. It is the conservative direction (a clip must
        # pass both tests), so it does not flatter the baseline.
        "success_rate_phc_survival": float(np.mean(survival_flags)),
        "success_rate_frame_head": float(np.mean(success_flags)),
        "mpjpe_mm": float(np.mean(pos_errors_m) * 1000),
        "mpjpe_cm": float(np.mean(pos_errors_m) * 100),
        "mpjre_deg": float(np.mean(rot_errors)),
        "mpjve_cm_s": float(np.mean(vel_errors_m_s) * 100),
        "jitter_m_s3": float(np.mean(jitters)),
        "foot_skate_cm_s": float(np.mean(foot_skates) * 100) if foot_skates else None,
        "per_activity_success": {k: {"rate": float(np.mean(v)), "n": len(v)}
                                  for k, v in sorted(per_activity.items())},
    }

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(res, f, indent=2)

    print(f"\nScored {res['num_sequences']} sequences against GROUND TRUTH "
          f"({missing} missing)\n")
    for k in ("success_rate_phc_survival", "success_rate_frame_head", "mpjpe_mm", "mpjre_deg",
              "mpjve_cm_s", "jitter_m_s3", "foot_skate_cm_s"):
        print(f"  {k:28s} {res[k]}")
    print("\n  per-activity success (phc_survival):")
    for k, v in res["per_activity_success"].items():
        print(f"    {k:16s} {v['rate']:.3f}  (n={v['n']})")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
