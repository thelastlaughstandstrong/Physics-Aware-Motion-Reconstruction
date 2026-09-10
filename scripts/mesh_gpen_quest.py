"""
Simulator-independent mesh-level ground penetration for Quest2 methods.

Every stream goes through ONE identical pipeline:

    25-joint quest_humanoid world positions
      -> validated Step 4/5 position-based Quest->SMPL IK (retarget_quest_to_smpl.fit_single_frame,
         warm-started frame to frame, unchanged)
      -> SMPL-H NEUTRAL mesh (betas = 0)
      -> vertex penetration against the common Z = 0 floor

The penetration definition is utils.physical_metrics.compute_ground_penetration, imported and used
as-is -- the same function the AvatarPoser baseline uses:
    GPen Mean (mm) : mean depth over PENETRATING (vertex, frame) pairs only
    GPen Rate      : fraction of FRAMES with at least one vertex below the floor
    GPen Max  (mm) : worst single-vertex depth
    tolerance      : 1e-4 m (0.1 mm), floor at ground_z = 0.0

Why joint-based gpen is not used: PHC/Isaac Gym enforces contact on collision bodies, so its
rigid-body penetration is ~0 by construction. A mesh sole sits below the toe JOINT, so the two
quantities differ by geometry, not by error.

No per-method vertical alignment, no foot-to-floor fitting, no smoothing, no coordinate correction:
all streams already live in the same native Z-up world with the floor at Z=0 (verified in the
coordinate audit). Neutral betas throughout, matching the retargeting's own rest offsets.

Usage:
  python scripts/mesh_gpen_quest.py --clips <clip> [...] --streams gt=<> simxr=<dump> ... --out <json>
"""
import argparse
import json
import os
import sys
import time

import joblib
import numpy as np
import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "scripts"))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "phc"))

import retarget_quest_to_smpl as M                                      # noqa: E402
from poselib.poselib.skeleton.skeleton3d import SkeletonTree            # noqa: E402
from human_body_prior.body_model.body_model import BodyModel            # noqa: E402
from utils.physical_metrics import compute_ground_penetration           # noqa: E402

CORR = os.path.join(REPO, "scripts/quest_smpl_correspondence.json")
CAL = os.path.join(REPO, "scripts/quest_smpl_frame_calibration.json")
BM_NEUTRAL = os.path.join(REPO, "support_data/body_models/smplh/neutral/model.npz")
DMPL_NEUTRAL = os.path.join(REPO, "support_data/body_models/dmpls/neutral/model.npz")
CACHE = os.path.join(REPO, "data/quest_smpl_retarget/mesh_gpen_cache")


class IKArgs:
    """Same IK settings the validated Step 5 sequence run used."""
    iters = 3000
    iters_first = 3000
    iters_warm = 600
    tol = 1e-6
    lr = 0.05
    w_group_a = 1.0
    w_group_b = 0.3
    w_reg = 1e-3
    w_root = 0.25
    world_frame = "native"
    arm_init = False          # matches the validated Step 5 sequence run


def quest_positions(stream_kind, path, clip, tree):
    """25-joint world positions for one clip from one stream, in the native Z-up world."""
    if stream_kind == "gt":
        d = joblib.load(os.path.join(REPO, f"data/quest2_synthetic/test_seg_motion/{clip}.pkl"))
        e = d[list(d.keys())[0]]
        pq = np.asarray(e["pose_quat"], dtype=np.float32)
        rt = np.asarray(e["root_trans_offset"], dtype=np.float32)
        P = np.stack([M.quest_fk_frame(tree, pq, rt, t) for t in range(pq.shape[0])], 0)
        # Apply PHC's OWN loader convention so GT shares the vertical frame the three simulated
        # streams already inherit (motion_lib_quest.MotionLibQuest.fix_trans_height): shift Z so
        # the minimum joint height over the FIRST 30 FRAMES equals 0.0871 m -- quest_humanoid's
        # empirical foot-joint height above the floor. This is a fixed 30-frame calibration, NOT a
        # per-clip foot-to-floor fit, so it cannot erase penetration later in the sequence.
        P[..., 2] -= float(P[:30, :, 2].min()) - 0.0871
        return P
    d = joblib.load(path)
    keys = [str(k) for k in d["motion_keys"]]
    idx = [i for i, k in enumerate(keys) if clip in k]
    if not idx:
        return None
    return np.asarray(d["pred_pos_all"][idx[0]], dtype=np.float64)


def run_ik(Qt, soff, smpl_par, direct, groups, qn, dev):
    """Warm-started per-frame position IK -- identical call pattern to Step 5."""
    args = IKArgs()
    recs, prev = [], None
    for t in range(len(Qt)):
        r = M.fit_single_frame(
            Qt[t], soff, smpl_par, direct, groups, qn, args, dev,
            init=prev,
            max_iters=(args.iters_warm if prev is not None else args.iters_first),
            tol=args.tol)
        prev = {"root_orient": r["root_orient_aa"], "body_pose": r["body_pose_aa"],
                "trans": r["trans"]}
        recs.append(r)
    return recs


def mesh_gpen(recs, bm, device):
    """SMPL params -> SMPL-H neutral mesh -> penetration, using the repo's own definition."""
    root = torch.tensor(np.array([r["root_orient_aa"] for r in recs]), dtype=torch.float32)
    body = torch.tensor(np.array([r["body_pose_aa"] for r in recs]), dtype=torch.float32)
    trans = torch.tensor(np.array([r["trans"] for r in recs]), dtype=torch.float32)
    pose_body = body[:, :21].reshape(len(recs), 63)          # SMPL-H body = 21 joints; hands excluded
    verts = []
    with torch.no_grad():
        for s in range(0, len(recs), 256):
            e = min(s + 256, len(recs))
            out = bm(pose_body=pose_body[s:e].to(device),
                     root_orient=root[s:e].to(device),
                     trans=trans[s:e].to(device))
            verts.append(out.v.cpu())
    V = torch.cat(verts, 0)
    mean_mm, max_mm, rate = compute_ground_penetration(V, ground_z=0.0)
    lowest = float(V[..., 2].min())
    return float(mean_mm), float(max_mm), float(rate), lowest, V.shape[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clips", nargs="+", required=True)
    ap.add_argument("--streams", nargs="+", required=True, help="label=gt | label=<dump path>")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    os.makedirs(CACHE, exist_ok=True)
    dev = torch.device("cpu")                      # IK is float64 CPU, as validated
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    corr = json.load(open(CORR))
    smpl_par = corr["smpl_parents"]
    direct = {int(k): v for k, v in corr["direct"].items()}
    groups = corr["validation_groups"]
    qn = corr["quest_joint_names"]
    soff = M.load_smpl_rest_offsets(corr)
    tree = SkeletonTree.from_mjcf(M.MJCF)
    bm = BodyModel(bm_fname=BM_NEUTRAL, num_betas=16, num_dmpls=8,
                   dmpl_fname=DMPL_NEUTRAL).to(device)
    print(f"body model: SMPL-H NEUTRAL, betas=0 | IK device cpu | mesh device {device}")

    results = {}
    for clip in args.clips:
        results[clip] = {}
        for spec in args.streams:
            label, src = spec.split("=", 1)
            kind = "gt" if src == "gt" else "dump"
            ck = os.path.join(CACHE, f"{clip}__{label}.pkl")
            t0 = time.time()
            if os.path.isfile(ck):
                recs = joblib.load(ck)
                cached = True
            else:
                Qt = quest_positions(kind, None if kind == "gt" else src, clip, tree)
                if Qt is None:
                    print(f"  {clip} / {label}: CLIP ABSENT in dump — skipped")
                    results[clip][label] = {"status": "absent"}
                    continue
                recs = run_ik(Qt, soff, smpl_par, direct, groups, qn, dev)
                joblib.dump(recs, ck)
                cached = False
            mean_mm, max_mm, rate, lowest, nv = mesh_gpen(recs, bm, device)
            res = {
                "frames": len(recs),
                "ik_all_mean_mm": float(np.mean([r["all_mean"] for r in recs])),
                "ik_group_a_mean_mm": float(np.mean([r["group_a_mean"] for r in recs])),
                "gpen_mean_mm": mean_mm, "gpen_rate": rate, "gpen_max_mm": max_mm,
                "lowest_vertex_z_m": lowest, "n_vertices": int(nv),
                "wall_s": round(time.time() - t0, 1), "cached": cached,
            }
            results[clip][label] = res
            print(f"  {clip[:34]:34s} {label:20s} IK {res['ik_all_mean_mm']:6.1f}mm | "
                  f"gpen mean {mean_mm:7.2f} rate {rate:6.3f} max {max_mm:7.2f} | "
                  f"lowest vtx {lowest*1000:+7.1f}mm | {res['wall_s']}s"
                  + (" [cached]" if cached else ""), flush=True)
        json.dump(results, open(args.out, "w"), indent=2)
    json.dump(results, open(args.out, "w"), indent=2)
    print("wrote", args.out)


if __name__ == "__main__":
    sys.exit(main())
