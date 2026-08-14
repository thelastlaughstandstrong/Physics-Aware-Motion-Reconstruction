"""
Per-clip metric table for the three rendered comparison clips (one per motion category), across
finetuned_backbone / option2_scratch / SimXR-no-vision.

Uses scripts/evaluate_phc_metrics_quest.py's OWN formulas and constants (imported directly, not
re-derived) so these per-clip numbers are directly comparable to the aggregate table in
results_comparison.md -- with the one difference that everything here is a single clip rather
than a mean over 1072.

Adds a root-drift / root-relative decomposition that the aggregate script doesn't compute:
positional error splits into "whole body is in the wrong place" (root drift) vs "limbs are in the
wrong configuration" (root-relative pose error), and for these models the split is the
interesting part -- see results_per_clip_renders.md.

Usage: python scripts/per_clip_metrics.py [-o results_per_clip_renders.md]
"""
import argparse
import os
import sys

import joblib
import numpy as np
from scipy.spatial.transform import Rotation as sRot

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "scripts"))
from evaluate_phc_metrics_quest import (  # noqa: E402  -- reuse the project's own definitions
    FOOT_JOINTS, HEAD_IDX, FPS, GROUND_Z, compute_foot_skate, compute_gpen,
)

CLIPS = [
    ("upper-body / planted",       "SUBJECT072.gestures.pkl_1281_1707"),
    ("full-body / transition",     "SUBJECT057.yoga.pkl_1643_2052"),
    ("lower-body / locomotion",    "SUBJECT084.walking.pkl_0_426"),
]
OURS = [
    ("finetuned_backbone@440k", "quest_distill_finetuned_backbone"),
    ("option2_scratch@400k",    "quest_distill_option2_scratch"),
    # Stage R (reward PPO) on top of finetuned_backbone. Only the BUG-FIXED rerun is reported:
    # the earlier trial (quest_distill_finetuned_backbone_stager) was trained with the
    # dropout/eval-mode bug that contaminated its PPO ratios (results_comparison.md fn.12), so
    # its numbers do not measure what Stage R does, despite its higher headline success rate.
    ("stageR_t2@442.1k",        "quest_distill_finetuned_backbone_stager_v2"),
]
SX_DUMPS = "/tmp/sxdumps"


def clip_metrics(pred_pos, gt_pos, pred_rot, gt_rot):
    T = pred_pos.shape[0]
    err = np.linalg.norm(pred_pos - gt_pos, axis=-1)
    rel = (sRot.from_quat(pred_rot.reshape(-1, 4)).inv()
           * sRot.from_quat(gt_rot.reshape(-1, 4))).magnitude()
    pred_vel = (pred_pos[1:] - pred_pos[:-1]) * FPS
    gt_vel = (gt_pos[1:] - gt_pos[:-1]) * FPS
    jerk = (pred_pos[3:] - 3 * pred_pos[2:-1] + 3 * pred_pos[1:-2] - pred_pos[:-3]) * (FPS ** 3)
    per_seq_ground = float(gt_pos[:, FOOT_JOINTS, 2].min())
    gp_mean, gp_max, gp_rate = compute_gpen(pred_pos, GROUND_Z)
    # decomposition: whole-body displacement vs limb configuration
    root_drift = np.linalg.norm(pred_pos[:, 0, :] - gt_pos[:, 0, :], axis=-1)
    local = np.linalg.norm((pred_pos - pred_pos[:, 0:1, :]) - (gt_pos - gt_pos[:, 0:1, :]), axis=-1)
    return {
        "frames": T,
        "eg_mpjpe_mm": float(err.mean() * 1000),
        "root_drift_mm": float(root_drift.mean() * 1000),
        "pose_err_mm": float(local.mean() * 1000),
        "rot_error_deg": float(np.degrees(rel.mean())),
        "vel_error_cm_s": float(np.linalg.norm(pred_vel - gt_vel, axis=-1).mean() * 100),
        "jitter_m_s3": float(np.linalg.norm(jerk, axis=-1).mean()),
        "foot_skate_cm_s": float(compute_foot_skate(pred_pos, per_seq_ground) * 100),
        "gpen_mean_mm": gp_mean,
        "gpen_rate": gp_rate,
        "succ_frame_head": float((err[:, HEAD_IDX] < 0.5).mean()),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-o", default=os.path.join(REPO, "results_per_clip_renders.md"))
    args = ap.parse_args()

    results = {}
    for _, clip in CLIPS:
        results[clip] = {}
        for label, exp in OURS:
            d = joblib.load(f"{REPO}/phc/output/HumanoidIm/{exp}/phc_positions_for_eval.pkl")
            succ = {os.path.basename(str(x)) for x in
                    joblib.load(f"{REPO}/phc/output/HumanoidIm/{exp}/long_succ.pkl")}
            idx = [i for i, k in enumerate(d["motion_keys"]) if clip in str(k)]
            if not idx:
                continue
            i = idx[0]
            m = clip_metrics(np.asarray(d["pred_pos_all"][i]), np.asarray(d["ref_pos_all"][i]),
                             np.asarray(d["pred_rot_all"][i]), np.asarray(d["gt_rot_all"][i]))
            m["survived"] = os.path.basename(str(d["motion_keys"][i])) in succ
            results[clip][label] = m
        p = f"{SX_DUMPS}/{clip}.pkl"
        if os.path.isfile(p):
            d = joblib.load(p)
            m = clip_metrics(np.asarray(d["pred_pos_all"][0]), np.asarray(d["ref_pos_all"][0]),
                             np.asarray(d["pred_rot_all"][0]), np.asarray(d["gt_rot_all"][0]))
            m["survived"] = len(joblib.load(f"{SX_DUMPS}/{clip}.succ.pkl")) > 0
            results[clip]["SimXR-no-vision@e1500"] = m
    joblib.dump(results, "/tmp/per_clip_results.pkl")

    cols = ["survived", "frames", "eg_mpjpe_mm", "root_drift_mm", "pose_err_mm", "rot_error_deg",
            "vel_error_cm_s", "jitter_m_s3", "foot_skate_cm_s", "gpen_mean_mm", "succ_frame_head"]
    for cat, clip in CLIPS:
        print(f"\n=== {cat}: {clip} ===")
        print(f"{'model':<24}" + "".join(f"{c:>16}" for c in cols))
        for label in ["finetuned_backbone@440k", "stageR_t2@442.1k",
                      "option2_scratch@400k", "SimXR-no-vision@e1500"]:
            m = results[clip].get(label)
            if not m:
                continue
            cells = []
            for c in cols:
                v = m[c]
                cells.append(f"{str(v):>16}" if isinstance(v, bool) else
                             (f"{v:>16d}" if isinstance(v, int) else f"{v:>16.2f}"))
            print(f"{label:<24}" + "".join(cells))


if __name__ == "__main__":
    main()
