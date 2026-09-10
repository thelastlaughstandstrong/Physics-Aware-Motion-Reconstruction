"""
Export the QUEST GROUND TRUTH stream for Blender, aligned into the same world as the
pre-/post-PHC SMPL exports.

Quest GT is NOT an SMPL body -- quest_humanoid is a genuinely different 25-joint rig (Step 2), so
there is no SMPL mesh to drive and no pose vector to hand the SMPL-X addon. It is therefore
exported as joint POSITIONS for a marker cloud / bone-less skeleton overlay, plus the parent array
so Blender can draw the segments, plus the 17 joints that were actually IK targets (the rest are
structurally unmatched and should not be read as "error" if they diverge).

Alignment: PHC's loader applied a rigid fix_trans_height Z shift and samples the reference with a
small frame lag. Both are properties of the loader, so they are solved once against the PRE-sim
reference (exactly as the characterization study does) and the same (shift, dz) is applied here,
so all three streams share one world and one clock.

Run:
    python scripts/export_quest_gt_for_blender.py
"""
import json
import os
import sys

import joblib
import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "scripts"))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "phc"))

import isaacgym  # noqa: F401,E402
import numpy as np  # noqa: E402,F811

from evaluate_phc_metrics import MUJOCO_2_SMPL_REORDER as M2S  # noqa: E402
from analyze_step6e import quest_targets  # noqa: E402
from charstudy_analyze import gt_mpjpe  # noqa: E402

EXPORT = os.path.join(REPO, "data/quest_smpl_retarget/blender_export")
DUMP = os.path.join(REPO, "output/HumanoidIm/quest_smpl_charstudy/phc_positions_for_eval.pkl")
MANIFEST = os.path.join(REPO, "data/quest_smpl_retarget/charstudy/manifest.json")
CORR = os.path.join(REPO, "scripts/quest_smpl_correspondence.json")
CLIPS = [("sports", 9), ("gestures", 8), ("martialarts", 0)]


def zup_to_yup(p):
    x, y, z = p[..., 0], p[..., 1], p[..., 2]
    return np.stack([x, z, -y], axis=-1)


def main():
    from poselib.poselib.skeleton.skeleton3d import SkeletonTree
    import retarget_quest_to_smpl as M

    d = joblib.load(DUMP)
    man = json.load(open(MANIFEST))
    corr = json.load(open(CORR))
    direct = {int(k): v for k, v in corr["direct"].items()}
    qi = sorted(direct)
    muj = [M2S[direct[q]] for q in qi]
    tree = SkeletonTree.from_mjcf(M.MJCF)
    qparents = tree.parent_indices.tolist()
    qnames = list(tree.node_names)

    for tag, idx in CLIPS:
        key = [str(k) for k in d["motion_keys"]][idx]
        meta = man[key]
        pre = np.asarray(d["ref_pos_all"][idx], dtype=np.float64)
        Q = quest_targets(meta["source_clip"], corr)

        # same (shift, dz) solve as charstudy_analyze -> one shared world and clock
        T = len(pre)
        best = None
        for sh in range(-3, 4):
            a = slice(max(0, sh), T + min(0, sh))
            b = slice(max(0, -sh), len(Q) + min(0, -sh))
            m = min(len(range(*a.indices(T))), len(range(*b.indices(len(Q)))))
            cand = float(np.median(pre[a][:m][:, muj][..., 2] - Q[b][:m][:, qi][..., 2]))
            e = gt_mpjpe(pre, Q, muj, qi, cand, sh)
            if best is None or e < best[0]:
                best = (e, sh, cand)
        err, sh, dz = best

        # resample Quest onto the exported clips' frame indexing, then lift by dz so it sits in
        # the same world the PHC streams live in
        a = slice(max(0, sh), T + min(0, sh))
        b = slice(max(0, -sh), len(Q) + min(0, -sh))
        m = min(len(range(*a.indices(T))), len(range(*b.indices(len(Q)))))
        idx_src = np.arange(len(Q))[b][:m]
        aligned = np.full((T, Q.shape[1], 3), np.nan)
        aligned[np.arange(T)[a][:m]] = Q[idx_src] + np.array([0, 0, dz])
        # frames with no Quest counterpart (edges of the shift) hold the nearest valid frame
        valid = ~np.isnan(aligned[:, 0, 0])
        first, last = np.argmax(valid), T - 1 - np.argmax(valid[::-1])
        aligned[:first] = aligned[first]
        aligned[last + 1:] = aligned[last]

        out = os.path.join(EXPORT, f"quest_gt_{tag}.npz")
        np.savez(out,
                 positions_zup=aligned.astype(np.float32),
                 positions_yup=zup_to_yup(aligned).astype(np.float32),
                 parents=np.asarray(qparents, dtype=np.int32),
                 joint_names=np.asarray(qnames),
                 ik_target_joints=np.asarray(qi, dtype=np.int32),
                 fps=np.float32(30.0), frames=np.int32(T),
                 frame_shift=np.int32(sh), z_offset=np.float32(dz),
                 motion_key=key, source_clip=meta["source_clip"])
        print(f"{tag:12s} T={T} joints={Q.shape[1]} shift={sh:+d} dz={dz*1000:+.1f}mm "
              f"pre-sim fit={err:.1f}mm -> {os.path.relpath(out, REPO)}")


if __name__ == "__main__":
    sys.exit(main())
