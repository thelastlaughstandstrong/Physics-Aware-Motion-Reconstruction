"""
Convert AvatarPoser's OWN predicted Quest2 poses (using only the 3 sparse HMD-tracked points as
input -- head + 2 wrists, no ground truth) into PHC-compatible motion pkls for the quest_humanoid
skeleton, one pkl per sequence, in exactly the schema motion_lib_quest.py expects (pose_quat,
root_trans_offset, fps -- see phc/phc/utils/motion_lib_quest.py's module docstring).

This is the Quest2 counterpart to convert_avatarposer_to_phc.py, which only handles the SMPL/AMASS
pkl format -- and the missing piece for running the SMPL pipeline's "naive baseline" experiment on
Quest2: feed AvatarPoser's own kinematic reconstruction straight to the frozen, general-purpose PHC
teacher (phc_prim_quest), with NO compressed distillation student in between. Every prior Quest2
physics run (Phase B, Phase C, and the explicit_pose_action_input variant) either tracked ground
truth or routed through a trained student -- this produces the first motion set that lets the
frozen teacher track AvatarPoser's own predictions instead.

Reuses main_test_avatarposer_quest2.py's exact prediction + world-position reconstruction logic
(sliding-window prediction -> FK at root-origin -> rigid shift onto the known/tracked head
position) rather than re-deriving it, so the pkl this produces is provably the same "prediction"
main_test_avatarposer_quest2.py already evaluates kinematically -- only the physics stage is new.

Usage (run from AvatarPoser repo root, conda activate avatarposer):
    python scripts/convert_avatarposer_quest2_to_phc.py
    python scripts/convert_avatarposer_quest2_to_phc.py --max_seqs 5     # smoke test, few sequences
    python scripts/convert_avatarposer_quest2_to_phc.py --split train

Then run the frozen phc_prim_quest teacher against the output (from phc/ directory):
    python phc/run_hydra.py learning=im_quest exp_name=phc_prim_quest env=env_quest_im_basic \\
      robot=quest_humanoid env.motion_file=data/quest2_synthetic/test_seg_avatarposer_pred \\
      env.num_envs=128 env.cycle_motion=False env.min_length=5 has_eval=True epoch=-1 test=True \\
      im_eval=True headless=True disable_multiprocessing=False

Then evaluate the same way as the Phase A/B comparison (results directly comparable to
results/baselines/phc_prim_quest_full_test_1072.json and avatarposer_quest2_phase_a.json):
    python scripts/evaluate_phc_metrics_quest.py \\
      --pos_dump output/HumanoidIm/phc_prim_quest/phc_positions_for_eval.pkl
"""
import argparse
import os
import sys

import joblib
import numpy as np
import torch
from torch.utils.data import DataLoader

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PHC_ROOT = os.path.join(REPO_ROOT, "phc")
for _p in (REPO_ROOT, PHC_ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from poselib.poselib.skeleton.skeleton3d import SkeletonState, SkeletonTree

from data.select_dataset import define_Dataset
from models.select_model import define_G
from models.quest_body_model import QUEST_HEAD_IDX, QUEST_HUMANOID_MJCF, QUEST_NUM_JOINTS
from utils import utils_option as option
from main_test_avatarposer_quest2 import sixd_to_quat_np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-opt', type=str, default='options/train_avatarposer_quest2.json')
    parser.add_argument('--ckpt', type=str, default=None,
                         help="Explicit checkpoint path (bypasses find_last_checkpoint, which "
                              "chokes on non-numeric filenames like latest_G.pth).")
    parser.add_argument('--split', type=str, default='test', choices=['train', 'test'],
                         help="Which dataset split to run AvatarPoser on and convert.")
    parser.add_argument('--out_dir', type=str, default=None,
                         help="Defaults to data/quest2_synthetic/<split>_seg_avatarposer_pred")
    parser.add_argument('--max_seqs', type=int, default=0,
                         help="Stop after this many sequences (0 = convert all). Useful for a "
                              "quick smoke test before converting the full 1072-sequence set.")
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

    dataset = define_Dataset(opt['datasets'][args.split])
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    window_size = opt['datasets'][args.split]['window_size']

    out_dir = args.out_dir or os.path.join(
        REPO_ROOT, "data", "quest2_synthetic", f"{args.split}_seg_avatarposer_pred")
    os.makedirs(out_dir, exist_ok=True)
    print(f"Writing predicted-motion pkls to {out_dir}")

    n_written, n_skipped = 0, 0
    with torch.no_grad():
        for data in loader:
            if args.max_seqs and n_written >= args.max_seqs:
                break

            key = data['filepath'][0]  # DataLoader wraps the length-1-batch str in a list
            L = data['L'].squeeze(0).to(device)                             # [T, 54]
            head_trans_global = data['Head_trans_global'].squeeze(0).numpy()  # [T,4,4]

            T = L.shape[0]
            if T <= window_size + 1:
                n_skipped += 1
                continue

            # Sliding windows -> predict last-frame pose for every valid window, same as
            # main_test_avatarposer_quest2.py.
            N = T - window_size
            windows = torch.stack([L[s:s + window_size + 1] for s in range(N)], dim=0)  # [N,W+1,54]
            global_orientation, joint_rotation = netG(windows, do_fk=False)             # [N,6],[N,144]

            pred_6d = torch.cat([global_orientation, joint_rotation], dim=-1).cpu().numpy()  # [N,150]
            pred_quat = sixd_to_quat_np(pred_6d.reshape(N, QUEST_NUM_JOINTS, 6))              # [N,25,4] local

            known_head_pos = head_trans_global[window_size - 1: window_size - 1 + N, :3, 3]  # [N,3]

            # Head-anchored world reconstruction, identical to main_test_avatarposer_quest2.py:
            # FK the predicted rotations with the root pinned at the origin, then rigidly shift
            # the whole predicted skeleton so its own head lands on the known/tracked head
            # position -- exactly what a real deployed system would do, since it only ever knows
            # head/wrist tracked poses, never the root/pelvis directly.
            sk_pred = SkeletonState.from_rotation_and_root_translation(
                skeleton_tree, torch.from_numpy(pred_quat.astype(np.float32)),
                torch.zeros(N, 3), is_local=True,
            )
            pred_pos_rel = sk_pred.global_translation.numpy()          # [N,25,3], root at origin
            shift = known_head_pos - pred_pos_rel[:, QUEST_HEAD_IDX, :]
            pred_pos = pred_pos_rel + shift[:, None, :]                # [N,25,3], world-anchored

            # root_trans_offset is quest_humanoid's direct SkeletonState root-translation argument
            # (joint 0 = b_root, no separate pelvis-offset convention the way the SMPL pkl spec
            # needs -- verified self-consistent: re-feeding pred_pos[:,0,:] as `trans` alongside
            # pred_quat into the same SkeletonState call reproduces pred_pos exactly).
            entry = {
                "pose_quat": torch.from_numpy(pred_quat.astype(np.float32)),
                "root_trans_offset": torch.from_numpy(pred_pos[:, 0, :].astype(np.float32)),
                "fps": 30,
            }
            joblib.dump({key: entry}, os.path.join(out_dir, f"{key}.pkl"))
            n_written += 1

            if n_written % 100 == 0:
                print(f"  ... {n_written} sequences written")

    print(f"Done: {n_written} sequences written to {out_dir}, {n_skipped} skipped "
          f"(too short for one window_size={window_size} prediction).")


if __name__ == '__main__':
    main()
