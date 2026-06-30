"""
Build PHC-format motion PKLs from AvatarPoser's train/test split.

Reads ground-truth body params directly from all three AvatarPoser datasets:
  data_cameraready_fps60/{BioMotionLab_NTroje,CMU,MPI_HDM05}/{train,test}/*.pkl
and converts them to the format expected by PHC's MotionLibSMPL.

Output:
  phc/sample_data/phc_train.pkl   (~4811 sequences across all 3 datasets, 30 fps)
  phc/sample_data/phc_test.pkl    (~536 sequences across all 3 datasets, 30 fps)

Run from repo root (conda activate avatarposer):
    python scripts/build_phc_splits.py
    python scripts/build_phc_splits.py --split test    # test only
    python scripts/build_phc_splits.py --max 50        # quick smoke test
"""

import argparse
import os
import sys

import joblib
import numpy as np
import torch
from scipy.spatial.transform import Rotation as sRot
from tqdm import tqdm

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PHC_ROOT  = os.path.join(REPO_ROOT, "phc")
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, PHC_ROOT)

from poselib.poselib.skeleton.skeleton3d import SkeletonState, SkeletonTree
from smpl_sim.smpllib.smpl_joint_names import SMPL_BONE_ORDER_NAMES, SMPL_MUJOCO_NAMES
from smpl_sim.smpllib.smpl_local_robot import SMPL_Robot as LocalRobot

SMPL_2_MUJOCO = [
    SMPL_BONE_ORDER_NAMES.index(q)
    for q in SMPL_MUJOCO_NAMES
    if q in SMPL_BONE_ORDER_NAMES
]

ROBOT_CFG = {
    "mesh": False,
    "rel_joint_lm": True,
    "upright_start": False,
    "remove_toe": False,
    "real_weight": True,
    "real_weight_porpotion_capsules": True,
    "real_weight_porpotion_boxes": True,
    "replace_feet": True,
    "masterfoot": False,
    "big_ankle": True,
    "freeze_hand": True,
    "box_body": False,
    "master_range": 50,
    "body_params": {},
    "joint_params": {},
    "geom_params": {},
    "actuator_params": {},
    "model": "smpl",
}

GENDER_MAP = {"male": 1, "female": 2, "neutral": 0}


def build_skeleton_tree(smpl_robot: LocalRobot) -> SkeletonTree:
    xml_path = "/tmp/smpl_humanoid_build_splits.xml"
    beta = np.zeros(16)
    smpl_robot.load_from_skeleton(
        betas=torch.from_numpy(beta[None,]),
        gender=[0],
        objs_info=None,
    )
    smpl_robot.write_xml(xml_path)
    return SkeletonTree.from_mjcf(xml_path)


def convert_body_params(
    root_orient: np.ndarray,
    pose_body:   np.ndarray,
    trans:       np.ndarray,
    skeleton_tree: SkeletonTree,
    source_fps: int = 60,
) -> dict:
    T = root_orient.shape[0]

    pose_aa = np.concatenate(
        [root_orient, pose_body, np.zeros((T, 6))], axis=1
    )  # (T, 72) — 24 joints

    skip    = max(1, source_fps // 30)
    pose_aa = pose_aa[::skip]
    trans   = trans[::skip]
    N       = pose_aa.shape[0]

    pose_aa_mj = pose_aa.reshape(N, 24, 3)[:, SMPL_2_MUJOCO]

    pose_quat = (
        sRot.from_rotvec(pose_aa_mj.reshape(-1, 3))
        .as_quat()
        .reshape(N, 24, 4)
    )

    root_trans_offset = (
        torch.from_numpy(trans.astype(np.float32)) + skeleton_tree.local_translation[0]
    )

    sk_state = SkeletonState.from_rotation_and_root_translation(
        skeleton_tree,
        torch.from_numpy(pose_quat.astype(np.float32)),
        root_trans_offset,
        is_local=True,
    )

    upright_rot_inv = sRot.from_quat([0.5, 0.5, 0.5, 0.5]).inv()
    pose_quat_global = (
        sRot.from_quat(sk_state.global_rotation.reshape(-1, 4).numpy()) * upright_rot_inv
    ).as_quat().reshape(N, -1, 4)

    sk_state = SkeletonState.from_rotation_and_root_translation(
        skeleton_tree,
        torch.from_numpy(pose_quat_global.astype(np.float32)),
        root_trans_offset,
        is_local=False,
    )

    return {
        "pose_quat_global":  sk_state.global_rotation.numpy(),
        "pose_quat":         sk_state.local_rotation.numpy(),
        "trans_orig":        trans,
        "root_trans_offset": root_trans_offset,
        "beta":              np.zeros(16),
        "gender":            "neutral",
        "pose_aa":           pose_aa,
        "fps":               30.0,
    }


def convert_split(split: str, data_root: str, skeleton_tree: SkeletonTree, max_seqs: int = None) -> dict:
    split_dir = os.path.join(data_root, split)
    files = sorted(f for f in os.listdir(split_dir) if f.endswith(".pkl"))
    if max_seqs:
        files = files[:max_seqs]

    motion_dict = {}
    skipped = 0

    for fname in tqdm(files, desc=f"Converting {split}"):
        fpath = os.path.join(split_dir, fname)
        try:
            d = joblib.load(fpath)
        except Exception as e:
            print(f"  SKIP {fname}: load error — {e}")
            skipped += 1
            continue

        bp = d.get("body_parms_list", {})
        if not isinstance(bp, dict) or "root_orient" not in bp:
            print(f"  SKIP {fname}: missing body_parms_list")
            skipped += 1
            continue

        root_orient = np.array(bp["root_orient"])
        pose_body   = np.array(bp["pose_body"])
        trans       = np.array(bp["trans"])
        fps         = int(d.get("framerate", 60))
        filepath    = str(d.get("filepath", fname))

        if root_orient.shape[0] < 10:
            skipped += 1
            continue

        try:
            entry = convert_body_params(root_orient, pose_body, trans, skeleton_tree, source_fps=fps)
        except Exception as e:
            print(f"  SKIP {fname}: conversion error — {e}")
            skipped += 1
            continue

        if entry["pose_aa"].shape[0] < 5:
            skipped += 1
            continue

        key = f"gt_{split}_{os.path.splitext(fname)[0]}_{filepath.replace('/', '_').replace('.npz', '')}"
        motion_dict[key] = entry

    print(f"  {split}: {len(motion_dict)} converted, {skipped} skipped")
    return motion_dict


DATASETS = ["BioMotionLab_NTroje", "CMU", "MPI_HDM05"]


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--split", choices=["train", "test", "both"], default="both")
    parser.add_argument("--dataroot", default="data_cameraready_fps60")
    parser.add_argument("--outdir", default="phc/sample_data")
    parser.add_argument("--max", type=int, default=None,
                        help="Max sequences per split (for quick smoke tests)")
    args = parser.parse_args()

    os.chdir(REPO_ROOT)
    out_dir = os.path.join(REPO_ROOT, args.outdir)
    os.makedirs(out_dir, exist_ok=True)

    print("Building SMPL skeleton tree …")
    smpl_robot    = LocalRobot(ROBOT_CFG, data_dir=os.path.join(PHC_ROOT, "data/smpl"))
    skeleton_tree = build_skeleton_tree(smpl_robot)

    splits = ["train", "test"] if args.split == "both" else [args.split]

    for split in splits:
        combined = {}
        for dataset in DATASETS:
            data_root = os.path.join(REPO_ROOT, args.dataroot, dataset)
            if not os.path.isdir(os.path.join(data_root, split)):
                print(f"  Skipping {dataset}/{split} (not found)")
                continue
            motion_dict = convert_split(split, data_root, skeleton_tree, max_seqs=args.max)
            combined.update(motion_dict)
        out_path = os.path.join(out_dir, f"phc_{split}.pkl")
        joblib.dump(combined, out_path)
        print(f"Saved {len(combined)} sequences → {out_path}")


if __name__ == "__main__":
    main()
