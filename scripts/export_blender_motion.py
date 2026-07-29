"""Extract one sequence's predicted vs. ground-truth motion from a
phc_positions_for_eval.pkl dump into a simple npz for Blender import.

phc_positions_for_eval.pkl (written by im_amp_players.py) stores pred_pos_all /
ref_pos_all as lists of [T_i, 24, 3] arrays in PHC's upright-start world frame,
Mujoco body order. This script picks one sequence, reorders it into
SMPL_BONE_ORDER_NAMES order (so the standard 24-joint SMPL parent hierarchy
applies directly), and dumps positions + hierarchy for a bpy script to consume.

No rotations/FK needed here -- pred_pos_all/ref_pos_all are already absolute
per-joint world positions, so a Blender skeleton can be built by just connecting
each joint to its parent's position every frame.

Usage:
    python scripts/export_blender_motion.py --pos_dump <path> [--seq_idx N | --seq_key KEY] --out <out.npz>
    python scripts/export_blender_motion.py --pos_dump <path> --list   # show per-sequence error to help pick one
"""
import argparse
import os

import joblib
import numpy as np
from smpl_sim.smpllib.smpl_joint_names import SMPL_BONE_ORDER_NAMES, SMPL_MUJOCO_NAMES

# Standard 24-joint SMPL kinematic parents (SMPL_BONE_ORDER_NAMES order) --
# same constant already validated in phc/learning/distill_agent.py.
SMPL_PARENTS = [-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19, 20, 21]

_MJ_2_SMPL = [SMPL_MUJOCO_NAMES.index(q) for q in SMPL_BONE_ORDER_NAMES if q in SMPL_MUJOCO_NAMES]


def _mujoco_to_smpl_order(pos_mj: np.ndarray) -> np.ndarray:
    """[T, 24, 3] Mujoco body order -> [T, 24, 3] SMPL_BONE_ORDER_NAMES order."""
    return pos_mj[:, _MJ_2_SMPL, :]


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

    pred_mj = np.asarray(pred_all[idx], dtype=np.float32)   # [T, 24, 3] Mujoco order
    gt_mj = np.asarray(gt_all[idx], dtype=np.float32)
    T = min(pred_mj.shape[0], gt_mj.shape[0])
    pred_mj, gt_mj = pred_mj[:T], gt_mj[:T]

    pred_smpl = _mujoco_to_smpl_order(pred_mj)
    gt_smpl = _mujoco_to_smpl_order(gt_mj)

    mean_err = float(np.mean(np.linalg.norm(pred_smpl - gt_smpl, axis=-1)))
    key = keys[idx]

    out_path = args.out or os.path.join(
        os.path.dirname(args.pos_dump), f"blender_motion_{key}.npz"
    )
    np.savez(
        out_path,
        pred_pos=pred_smpl,
        gt_pos=gt_smpl,
        parents=np.asarray(SMPL_PARENTS, dtype=np.int32),
        joint_names=np.asarray(SMPL_BONE_ORDER_NAMES),
        fps=args.fps,
        motion_key=key,
    )
    print(f"[export] seq idx={idx} key={key!r} T={T} mean_pos_err={mean_err:.4f}m")
    print(f"[export] wrote {out_path}")


if __name__ == "__main__":
    main()
