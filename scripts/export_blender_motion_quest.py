"""Extract one sequence's predicted vs. ground-truth motion from a
phc_positions_for_eval.pkl dump (quest_humanoid variant) into a simple npz for
Blender import.

Quest-specific companion to export_blender_motion.py, which hardcodes SMPL's
24-joint SMPL_BONE_ORDER_NAMES/SMPL_MUJOCO_NAMES and can't be reused as-is --
quest_humanoid is a genuinely different 25-joint skeleton (verified: this
pkl's pred_pos_all/ref_pos_all are [T, 25, 3], not [T, 24, 3]).

Unlike the SMPL export script, no Mujoco -> "clean" joint-order remap is
needed here: quest_humanoid has no separate external convention to remap
into (SMPL_BONE_ORDER_NAMES exists because AMASS/SMPL papers use that
ordering; quest_humanoid has no equivalent), and this project has already
verified elsewhere (distill_agent_quest.py's module docstring) that PHC's
own quest_humanoid state already matches poselib's SkeletonTree node order
directly -- so pred_pos_all/ref_pos_all are used in their native order, paired
with that same SkeletonTree's own parent array (queried directly from
quest_humanoid.xml, not hand-copied, so it can't drift out of sync).

No rotations/FK needed here either -- pred_pos_all/ref_pos_all are already
absolute per-joint world positions, so a Blender skeleton can be built by
just connecting each joint to its parent's position every frame.

Usage:
    python scripts/export_blender_motion_quest.py --pos_dump <path> [--seq_idx N | --seq_key KEY] --out <out.npz>
    python scripts/export_blender_motion_quest.py --pos_dump <path> --list   # show per-sequence error to help pick one
"""
import argparse
import os
import sys

import joblib
import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PHC_ROOT = os.path.join(REPO_ROOT, "phc")
for _p in (REPO_ROOT, PHC_ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from poselib.poselib.skeleton.skeleton3d import SkeletonTree

QUEST_HUMANOID_MJCF = os.path.join(PHC_ROOT, "phc/data/assets/mjcf/quest_humanoid.xml")


def _quest_hierarchy():
    tree = SkeletonTree.from_mjcf(QUEST_HUMANOID_MJCF)
    return tree.parent_indices.numpy().astype(np.int32), np.asarray(list(tree.node_names))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pos_dump", required=True, help="Path to phc_positions_for_eval.pkl")
    ap.add_argument("--seq_idx", type=int, default=None)
    ap.add_argument("--seq_key", type=str, default=None)
    ap.add_argument("--out", type=str, default=None)
    ap.add_argument("--list", action="store_true", help="List sequences by mean position error and exit")
    ap.add_argument("--fps", type=int, default=30)
    args = ap.parse_args()

    d = joblib.load(args.pos_dump)
    pred_all, gt_all, keys = d["pred_pos_all"], d["ref_pos_all"], d["motion_keys"]

    if args.list:
        errs = [float(np.mean(np.linalg.norm(p - g, axis=-1))) for p, g in zip(pred_all, gt_all)]
        order = np.argsort(errs)
        print(f"{'idx':>5}  {'mean_err_m':>10}  {'T':>5}  key")
        for i in order:
            print(f"{i:>5}  {errs[i]:>10.4f}  {pred_all[i].shape[0]:>5}  {keys[i]}")
        return

    if args.seq_key is not None:
        idx = keys.index(args.seq_key)
    elif args.seq_idx is not None:
        idx = args.seq_idx
    else:
        raise SystemExit("Pass --seq_idx, --seq_key, or --list")

    pred = np.asarray(pred_all[idx], dtype=np.float32)   # [T, 25, 3]
    gt = np.asarray(gt_all[idx], dtype=np.float32)
    T = min(pred.shape[0], gt.shape[0])
    pred, gt = pred[:T], gt[:T]

    parents, joint_names = _quest_hierarchy()
    assert pred.shape[1] == len(parents), (
        f"pred_pos_all has {pred.shape[1]} joints but quest_humanoid.xml has {len(parents)} -- "
        "skeleton mismatch, don't trust this render"
    )

    mean_err = float(np.mean(np.linalg.norm(pred - gt, axis=-1)))
    key = keys[idx]

    out_path = args.out or os.path.join(
        os.path.dirname(args.pos_dump), f"blender_motion_quest_{os.path.basename(key)}.npz"
    )
    np.savez(
        out_path,
        pred_pos=pred,
        gt_pos=gt,
        parents=parents,
        joint_names=joint_names,
        fps=args.fps,
        motion_key=key,
    )
    print(f"[export] seq idx={idx} key={key!r} T={T} mean_pos_err={mean_err:.4f}m")
    print(f"[export] wrote {out_path}")


if __name__ == "__main__":
    main()
