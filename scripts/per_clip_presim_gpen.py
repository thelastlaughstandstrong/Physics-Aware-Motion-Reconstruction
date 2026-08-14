"""
Pre-sim (backbone-only, before physics) gpen + pose error for specific clips.

"Pre-sim" = the kinematic pose the AvatarPoser backbone predicts (q_base), before PHC's physics
simulation ever runs. Valid to compute open-loop because the 54-D HMD input features are always
derived from the ground-truth reference motion, never from the simulator's state -- so a
standalone forward pass reproduces exactly what q_base would have been during a closed-loop
rollout (same argument as the aggregate pre-sim table in results_comparison.md).

Reports gpen under BOTH ground conventions so it can be compared in either direction:
  - GROUND_Z = 0.0            : the real Isaac Gym floor, matching post-sim gpen
  - per-sequence GT foot min  : Phase A's own convention, matching avatarposer_quest2_phase_a.json

NOTE: only applies to models that HAVE a kinematic backbone. SimXR maps sensors directly to
actions with no intermediate pose ("no intermediate representations" is the paper's central
design claim), so pre-sim gpen is undefined for it -- not merely unmeasured.

Usage: python scripts/per_clip_presim_gpen.py
"""
import os
import sys

import numpy as np
import torch
from scipy.spatial.transform import Rotation as sRot

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "phc"))
sys.path.insert(0, os.path.join(REPO, "scripts"))

from poselib.poselib.skeleton.skeleton3d import SkeletonState, SkeletonTree
from data.select_dataset import define_Dataset
from models.select_model import define_G
from models.quest_body_model import QUEST_HEAD_IDX, QUEST_HUMANOID_MJCF, QUEST_NUM_JOINTS
from utils import utils_option as option
from evaluate_phc_metrics_quest import FOOT_JOINTS, GROUND_Z, compute_gpen

CLIPS = {
    "upper-body / planted":    "SUBJECT072.gestures.pkl_1281_1707.pkl",
    "full-body / transition":  "SUBJECT057.yoga.pkl_1643_2052.pkl",
    "lower-body / locomotion": "SUBJECT084.walking.pkl_0_426.pkl",
}
BACKBONES = {
    "finetuned_backbone@440k": "model_zoo/quest_backbone_finetuned_backbone_extracted.pth",
    "option2_scratch@400k":    "model_zoo/quest_backbone_option2_extracted.pth",
}


def sixd_to_quat_np(sixd):
    v = sixd.reshape(-1, 6)
    c1, c2 = v[:, 0:3], v[:, 3:6]
    rot = np.stack([c1, c2, np.cross(c1, c2)], axis=-1)
    return sRot.from_matrix(rot).as_quat().reshape(*sixd.shape[:-1], 4)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    opt = option.dict_to_nonedict(option.parse(
        os.path.join(REPO, "options/train_avatarposer_quest2.json"), is_train=True))
    test_set = define_Dataset(opt["datasets"]["test"])
    window = opt["datasets"]["test"]["window_size"]
    sk = SkeletonTree.from_mjcf(QUEST_HUMANOID_MJCF)
    names = [os.path.basename(p) for p in test_set.filename_list]

    out = {}
    for label, ckpt in BACKBONES.items():
        net = define_G(opt).to(device)
        net.load_state_dict(torch.load(os.path.join(REPO, ckpt), map_location=device), strict=True)
        net.eval()
        out[label] = {}
        for cat, clip in CLIPS.items():
            i = names.index(clip)
            d = test_set[i]
            L = d["L"].to(device)
            H = d["H"].numpy()
            head = d["Head_trans_global"].numpy()
            T = L.shape[0]
            N = T - window
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

            m0, x0, r0 = compute_gpen(pred_pos, GROUND_Z)
            per_seq = float(gt[:, FOOT_JOINTS, 2].min())
            m1, x1, r1 = compute_gpen(pred_pos, per_seq)
            out[label][cat] = dict(
                frames=N,
                pos_err_mm=float(np.linalg.norm(pred_pos - gt, axis=-1).mean() * 1000),
                gpen_mm_z0=m0, gpen_rate_z0=r0,
                gpen_mm_seq=m1, gpen_rate_seq=r1,
            )

    hdr = f"{'model':<26}{'category':<26}{'frames':>7}{'q_base err':>12}{'gpen@Z=0':>10}{'rate':>7}{'gpen@seq':>10}{'rate':>7}"
    print(hdr)
    for label, per in out.items():
        for cat, m in per.items():
            print(f"{label:<26}{cat:<26}{m['frames']:>7}{m['pos_err_mm']:>12.1f}"
                  f"{m['gpen_mm_z0']:>10.2f}{m['gpen_rate_z0']:>7.3f}"
                  f"{m['gpen_mm_seq']:>10.2f}{m['gpen_rate_seq']:>7.3f}")


if __name__ == "__main__":
    main()
