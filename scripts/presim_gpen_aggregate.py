"""
Aggregate pre-sim (backbone-only, before physics) gpen over the full 1072-clip Quest2 test set.

Same procedure that produced results/baselines/avatarposer_quest2_presim_*.json and the pre-sim
table in results_comparison.md: extract a run's `backbone` sub-dict, run it open-loop over the test
set, and measure ground penetration on the predicted kinematic pose. Valid open-loop because the
HMD input features come from the dataset (always derived from the ground-truth reference, never the
simulator's state), so a standalone forward pass reproduces the q_base a closed-loop rollout saw.

Reports the per-sequence empirical ground reference (Phase A's convention) -- the convention every
"Pre-sim gpen" figure in results_comparison.md uses -- and GROUND_Z=0 alongside for reference.

Usage: python scripts/presim_gpen_aggregate.py --backbones <label>=<ckpt> [...] --out <json>
"""
import argparse
import json
import os
import sys

import numpy as np
import torch
from scipy.spatial.transform import Rotation as sRot

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "phc"))
sys.path.insert(0, os.path.join(REPO, "scripts"))

from utils import utils_option as option                              # noqa: E402
from data.select_dataset import define_Dataset                        # noqa: E402
from models.select_model import define_G                              # noqa: E402
from models.quest_body_model import (QUEST_HEAD_IDX, QUEST_HUMANOID_MJCF,  # noqa: E402
                                     QUEST_NUM_JOINTS)
from poselib.poselib.skeleton.skeleton3d import SkeletonState, SkeletonTree  # noqa: E402
from evaluate_phc_metrics_quest import FOOT_JOINTS, GROUND_Z, compute_gpen   # noqa: E402

QUEST_MJCF = os.path.join(REPO, "phc/phc/data/assets/mjcf/quest_humanoid.xml")



def sixd_to_quat_np(sixd):
    v = sixd.reshape(-1, 6)
    c1, c2 = v[:, 0:3], v[:, 3:6]
    rot = np.stack([c1, c2, np.cross(c1, c2)], axis=-1)
    return sRot.from_matrix(rot).as_quat().reshape(*sixd.shape[:-1], 4)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbones", nargs="+", required=True, help="label=checkpoint_path")
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    opt = option.dict_to_nonedict(option.parse(
        os.path.join(REPO, "options/train_avatarposer_quest2.json"), is_train=True))
    test_set = define_Dataset(opt["datasets"]["test"])
    window = opt["datasets"]["test"]["window_size"]
    sk = SkeletonTree.from_mjcf(QUEST_MJCF)
    n_total = len(test_set) if args.limit <= 0 else min(args.limit, len(test_set))
    print(f"device={device}  sequences={n_total}  window={window}")

    results = {}
    for spec in args.backbones:
        label, ckpt = spec.split("=", 1)
        net = define_G(opt).to(device)
        net.load_state_dict(torch.load(os.path.join(REPO, ckpt), map_location=device), strict=True)
        net.eval()
        m_z0, r_z0, m_sq, x_sq, r_sq, perr = [], [], [], [], [], []
        for i in range(n_total):
            d = test_set[i]
            L = d["L"].to(device)
            H = d["H"].numpy()
            head = d["Head_trans_global"].numpy()
            N = L.shape[0] - window
            if N < 4:
                continue
            win = torch.stack([L[s:s + window + 1] for s in range(N)], dim=0)
            with torch.no_grad():
                go, jr = net(win, do_fk=False)
            pred6 = torch.cat([go, jr], dim=-1).cpu().numpy()
            pq = sixd_to_quat_np(pred6.reshape(N, QUEST_NUM_JOINTS, 6))
            st = SkeletonState.from_rotation_and_root_translation(
                sk, torch.from_numpy(pq.astype(np.float32)), torch.zeros(N, 3), is_local=True)
            rel = st.global_translation.numpy()
            known_head = head[window - 1: window - 1 + N, :3, 3]
            pred_pos = rel + (known_head - rel[:, QUEST_HEAD_IDX, :])[:, None, :]

            gq = sixd_to_quat_np(H.reshape(-1, QUEST_NUM_JOINTS, 6))[window - 1: window - 1 + N]
            trans = d["P"]["root_trans_offset"].numpy()[2:][window - 1: window - 1 + N]
            gt = SkeletonState.from_rotation_and_root_translation(
                sk, torch.from_numpy(gq.astype(np.float32)),
                torch.from_numpy(trans.astype(np.float32)), is_local=True).global_translation.numpy()

            a, _, c = compute_gpen(pred_pos, GROUND_Z)
            m_z0.append(a); r_z0.append(c)
            per_seq = float(gt[:, FOOT_JOINTS, 2].min())
            a2, b2, c2 = compute_gpen(pred_pos, per_seq)
            m_sq.append(a2); x_sq.append(b2); r_sq.append(c2)
            perr.append(float(np.linalg.norm(pred_pos - gt, axis=-1).mean() * 1000))
            if (i + 1) % 100 == 0:
                print(f"  {label}: {i+1}/{n_total}  running gpen@seq {np.mean(m_sq):.2f}mm", flush=True)

        results[label] = {
            "checkpoint": ckpt, "num_sequences": len(m_sq),
            "gpen_mean_mm": float(np.mean(m_sq)), "gpen_max_mm": float(np.mean(x_sq)),
            "gpen_rate": float(np.mean(r_sq)),
            "gpen_mean_mm_z0": float(np.mean(m_z0)), "gpen_rate_z0": float(np.mean(r_z0)),
            "pos_err_mm": float(np.mean(perr)),
            "ground_convention": "per-sequence empirical GT min foot height (Phase A convention)",
        }
        r = results[label]
        print(f"[{label}] n={r['num_sequences']}  gpen_mean {r['gpen_mean_mm']:.2f} mm  "
              f"rate {r['gpen_rate']:.3f}  (Z=0: {r['gpen_mean_mm_z0']:.2f} mm / {r['gpen_rate_z0']:.3f})"
              f"  pos_err {r['pos_err_mm']:.2f} mm", flush=True)

    json.dump(results, open(args.out, "w"), indent=2)
    print("wrote", args.out)


if __name__ == "__main__":
    sys.exit(main())
