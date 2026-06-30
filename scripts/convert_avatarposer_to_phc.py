"""
Convert AvatarPoser predicted output for one or more test sequences into
the PHC-compatible motion pkl format.

Run from the AvatarPoser repo root (conda activate avatarposer first):

    # Convert AvatarPoser MODEL predictions for sequence 0:
    python scripts/convert_avatarposer_to_phc.py --seq_idx 0

    # Convert ground-truth poses instead (sanity check — PHC should track these perfectly):
    python scripts/convert_avatarposer_to_phc.py --seq_idx 0 --use_gt

    # Convert multiple sequences into one pkl (easier to browse in PHC viewer):
    python scripts/convert_avatarposer_to_phc.py --seq_idx 0 5 10 --out phc/sample_data/ap_multi.pkl

Output defaults to:  phc/sample_data/avatarposer_seq<idx>.pkl

Then visualise with PHC (run from phc/ directory):
    python phc/run_hydra.py learning=im_mcp exp_name=phc_kp_mcp_iccv test=True \\
      env=env_im_getup_mcp robot=smpl_humanoid robot.freeze_hand=True robot.box_body=False \\
      env.z_activation=relu \\
      "env.motion_file=sample_data/avatarposer_seq0.pkl" \\
      "env.models=['output/HumanoidIm/phc_kp_pnn_iccv/Humanoid.pth']" \\
      env.num_envs=1 env.obs_v=7 headless=False epoch=-1 no_virtual_display=True
"""

import argparse
import os
import sys

import joblib
import numpy as np
import torch
from scipy.spatial.transform import Rotation as sRot
from torch.utils.data import DataLoader

# ── paths ─────────────────────────────────────────────────────────────────────
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PHC_ROOT  = os.path.join(REPO_ROOT, "phc")
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, PHC_ROOT)

# ── AvatarPoser imports ────────────────────────────────────────────────────────
from data.select_dataset import define_Dataset
from models.select_model import define_Model
from utils import utils_option as option

# ── PHC / poselib imports ──────────────────────────────────────────────────────
from poselib.poselib.skeleton.skeleton3d import SkeletonState, SkeletonTree
from smpl_sim.smpllib.smpl_joint_names import SMPL_BONE_ORDER_NAMES, SMPL_MUJOCO_NAMES
from smpl_sim.smpllib.smpl_local_robot import SMPL_Robot as LocalRobot

# Joint permutation: SMPL canonical order → Mujoco order (used by PHC)
SMPL_2_MUJOCO = [
    SMPL_BONE_ORDER_NAMES.index(q)
    for q in SMPL_MUJOCO_NAMES
    if q in SMPL_BONE_ORDER_NAMES
]

# Neutral robot config (matches PHC's smpl_humanoid robot)
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


def build_skeleton_tree(smpl_robot: LocalRobot) -> SkeletonTree:
    """Write a neutral SMPL humanoid XML and parse its skeleton tree."""
    xml_path = "/tmp/smpl_humanoid_avatarposer.xml"
    beta = np.zeros(16)
    smpl_robot.load_from_skeleton(
        betas=torch.from_numpy(beta[None,]),
        gender=[0],          # 0 = neutral
        objs_info=None,
    )
    smpl_robot.write_xml(xml_path)
    return SkeletonTree.from_mjcf(xml_path)


def convert_body_params(
    root_orient: np.ndarray,   # [T, 3]  axis-angle
    pose_body:   np.ndarray,   # [T, 63] axis-angle  (joints 1-21)
    trans:       np.ndarray,   # [T, 3]
    skeleton_tree: SkeletonTree,
    source_fps: int = 60,
) -> dict:
    """
    Convert a single sequence of AvatarPoser body params into the dict format
    that PHC's MotionLibSMPL expects.

    Key steps
    ---------
    1. Pad to 24 SMPL joints  (add zero hand joints 22-23)
    2. Downsample 60 fps → 30 fps
    3. Reorder joints: SMPL canonical → Mujoco joint order
    4. axis-angle → quaternion (local)
    5. Build SkeletonState → extract global rotations
    6. Compute root_trans_offset (adds pelvis offset from skeleton tree)
    """
    T = root_orient.shape[0]

    # Step 1 — full 24-joint axis-angle [T, 72]
    pose_aa = np.concatenate(
        [root_orient, pose_body, np.zeros((T, 6))], axis=1
    )

    # Step 2 — downsample to 30 fps
    skip    = max(1, source_fps // 30)
    pose_aa = pose_aa[::skip]
    trans   = trans[::skip]
    N       = pose_aa.shape[0]

    # Step 3 — reorder joints
    pose_aa_mj = pose_aa.reshape(N, 24, 3)[:, SMPL_2_MUJOCO]

    # Step 4 — axis-angle → local quaternion [N, 24, 4]
    pose_quat = (
        sRot.from_rotvec(pose_aa_mj.reshape(-1, 3))
        .as_quat()
        .reshape(N, 24, 4)
    )

    # Step 5 — root translation offset (pelvis height adjustment)
    root_trans_offset = (
        torch.from_numpy(trans) + skeleton_tree.local_translation[0]
    )

    # Step 6 — forward kinematics via SkeletonState
    sk_state = SkeletonState.from_rotation_and_root_translation(
        skeleton_tree,
        torch.from_numpy(pose_quat),
        root_trans_offset,
        is_local=True,
    )

    # Step 7 — apply upright_start rotation so data matches PHC's "humanoid faces +X" frame.
    # Matches convert_amass_data.py --upright_start. PHC's smpl_humanoid has has_upright_start=True,
    # so the policy was trained on data in this frame.
    # Right-multiply each global rotation by the inverse upright rotation (same as the reference script).
    upright_rot_inv = sRot.from_quat([0.5, 0.5, 0.5, 0.5]).inv()
    pose_quat_global = (
        sRot.from_quat(sk_state.global_rotation.reshape(-1, 4).numpy()) * upright_rot_inv
    ).as_quat().reshape(N, -1, 4)

    sk_state = SkeletonState.from_rotation_and_root_translation(
        skeleton_tree,
        torch.from_numpy(pose_quat_global),
        root_trans_offset,
        is_local=False,
    )

    return {
        "pose_quat_global":  sk_state.global_rotation.numpy(),   # [N, 24, 4]  ndarray
        "pose_quat":         sk_state.local_rotation.numpy(),    # [N, 24, 4]  ndarray
        "trans_orig":        trans,                               # [N, 3]      ndarray
        "root_trans_offset": root_trans_offset,                  # [N, 3]      Tensor  (PHC calls .clone() on this)
        "beta":              np.zeros(16),
        "gender":            "neutral",
        "pose_aa":           pose_aa,                            # [N, 72]     ndarray
        "fps":               30.0,
    }


def load_avatarposer(opt_path: str = "options/test_avatarposer.json"):
    """Initialise the AvatarPoser model and test dataloader."""
    opt = option.parse(opt_path, is_train=True)

    # Locate latest checkpoint
    from utils.utils_option import find_last_checkpoint
    _, init_path_G = find_last_checkpoint(opt["path"]["models"], net_type="G")
    opt["path"]["pretrained_netG"] = init_path_G
    opt = option.dict_to_nonedict(opt)

    test_set    = define_Dataset(opt["datasets"]["test"])
    test_loader = DataLoader(
        test_set, batch_size=1, shuffle=False,
        num_workers=0, drop_last=False,
    )

    model = define_Model(opt)
    model.init_test()

    return model, test_loader


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seq_idx", type=int, nargs="+", default=[0],
                        help="Test-set sequence index (or multiple indices).")
    parser.add_argument("--out", type=str, default=None,
                        help="Output pkl path. Default: phc/sample_data/avatarposer_seq<idx>.pkl")
    parser.add_argument("--use_gt", action="store_true",
                        help="Use ground-truth body params instead of AvatarPoser predictions "
                             "(sanity check — PHC should track GT perfectly).")
    parser.add_argument("--opt", type=str, default="options/test_avatarposer.json",
                        help="AvatarPoser options JSON.")
    args = parser.parse_args()

    os.chdir(REPO_ROOT)

    # ── build neutral skeleton tree once ──────────────────────────────────────
    print("Building SMPL robot skeleton tree …")
    smpl_robot    = LocalRobot(ROBOT_CFG, data_dir=os.path.join(PHC_ROOT, "data/smpl"))
    skeleton_tree = build_skeleton_tree(smpl_robot)

    # ── load AvatarPoser ───────────────────────────────────────────────────────
    print("Loading AvatarPoser model …")
    model, test_loader = load_avatarposer(args.opt)

    # ── convert requested sequences ───────────────────────────────────────────
    target_indices = set(args.seq_idx)
    motion_dict    = {}

    for idx, test_data in enumerate(test_loader):
        if idx not in target_indices:
            continue

        filepath = test_data["filepath"][0]
        seq_name = filepath.replace("/", "_").replace(".npz", "")
        label    = "gt" if args.use_gt else "pred"
        key      = f"{label}_{idx:04d}_{seq_name}"

        print(f"  [{idx}] {filepath}")

        if args.use_gt:
            # GT body params are in test_data["P"] with shape [1, T, *]
            bp = {k: v.squeeze(0) for k, v in test_data["P"].items()}
        else:
            model.feed_data(test_data, test=True)
            model.test()
            bp = model.current_prediction()

        root_orient = bp["root_orient"]
        pose_body   = bp["pose_body"]
        trans       = bp["trans"]

        if isinstance(root_orient, torch.Tensor):
            root_orient = root_orient.detach().cpu().numpy()
        if isinstance(pose_body, torch.Tensor):
            pose_body   = pose_body.detach().cpu().numpy()
        if isinstance(trans, torch.Tensor):
            trans       = trans.detach().cpu().numpy()

        entry = convert_body_params(
            root_orient, pose_body, trans,
            skeleton_tree, source_fps=60,
        )

        if entry is None or entry["pose_aa"].shape[0] < 10:
            print(f"    → skipped (too short after downsampling)")
            continue

        T_out = entry["pose_aa"].shape[0]
        print(f"    → {T_out} frames at 30 fps  |  key: {key}")
        motion_dict[key] = entry

        if idx >= max(target_indices):
            break

    if not motion_dict:
        print("No sequences converted.")
        return

    # ── save ──────────────────────────────────────────────────────────────────
    if args.out:
        out_path = args.out
    elif len(args.seq_idx) == 1:
        out_path = os.path.join(PHC_ROOT, f"sample_data/avatarposer_seq{args.seq_idx[0]}.pkl")
    else:
        idxs     = "_".join(str(i) for i in args.seq_idx)
        out_path = os.path.join(PHC_ROOT, f"sample_data/avatarposer_seqs_{idxs}.pkl")

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    joblib.dump(motion_dict, out_path)
    print(f"\nSaved {len(motion_dict)} sequence(s) → {out_path}")
    print("\nTo visualise in PHC viewer, run from the phc/ directory:")
    rel = os.path.relpath(out_path, PHC_ROOT)
    print(f"  python phc/run_hydra.py learning=im_mcp exp_name=phc_kp_mcp_iccv test=True \\")
    print(f"    env=env_im_getup_mcp robot=smpl_humanoid robot.freeze_hand=True robot.box_body=False \\")
    print(f"    env.z_activation=relu \\")
    print(f"    \"env.motion_file={rel}\" \\")
    print(f"    \"env.models=['output/HumanoidIm/phc_kp_pnn_iccv/Humanoid.pth']\" \\")
    print(f"    env.num_envs=1 env.obs_v=7 headless=False epoch=-1 no_virtual_display=True")


if __name__ == "__main__":
    main()
