"""
STEP 1 of the Quest->SMPL retargeting work: verify that the Quest2 motion files' rotation
conventions are interpreted correctly, BEFORE any retargeting is attempted.

Verifies the chain:
    pose_quat (local) + root translation -> Quest FK -> global rotations / positions
against the file's own stored `pose_quat_global` and its stored reference joint positions
(`head_fk_gt_3d`, `head_gt_3d`).

If the assumed convention is right, reconstruction error should be ~numerical zero. If it is
wrong, the sweep at the end identifies which of the likely failure modes (quaternion ordering,
local vs global, root handling, fixed base-frame rotation) actually applies.

No SMPL code, no frame calibration, no export -- deliberately out of scope for this step.

Usage:
    python scripts/verify_quest_rotation_convention.py [--n-clips 3] [--report <path.md>]
"""
import argparse
import os
import sys
import glob
import re

import numpy as np
import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "phc"))

import joblib  # noqa: E402
from poselib.poselib.skeleton.skeleton3d import SkeletonState, SkeletonTree  # noqa: E402

MJCF = os.path.join(REPO, "phc/phc/data/assets/mjcf/quest_humanoid.xml")
TEST_DIR = os.path.join(REPO, "data/quest2_synthetic/test_seg_motion")


def t(x):
    """-> float32 torch tensor, from tensor or array."""
    if torch.is_tensor(x):
        return x.detach().float()
    return torch.tensor(np.asarray(x), dtype=torch.float32)


def quat_angle_deg(q1, q2):
    """Geodesic angle between two xyzw quaternion sets, in degrees. abs() on the dot handles
    the double cover (q and -q are the same rotation)."""
    d = (q1 * q2).sum(-1).abs().clamp(max=1.0)
    return torch.rad2deg(2.0 * torch.acos(d))


def wxyz_to_xyzw(q):
    return torch.cat([q[..., 1:], q[..., 0:1]], dim=-1)


def quat_mul(a, b):
    x1, y1, z1, w1 = a.unbind(-1)
    x2, y2, z2, w2 = b.unbind(-1)
    return torch.stack([
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
    ], dim=-1)


def reconstruct(tree, quat_local, root_trans, is_local=True):
    st = SkeletonState.from_rotation_and_root_translation(tree, quat_local, root_trans, is_local=is_local)
    return st.global_rotation, st.global_translation


def _bone_lengths(X, par):
    """Mean bone length per non-root joint. X: [T,J,3] tensor."""
    return torch.stack([(X[:, j] - X[:, par[j]]).norm(dim=-1).mean() for j in range(1, X.shape[1])])


def _procrustes_mean_mm(A, B, stride=20):
    """Mean residual (mm) after the best-fit per-frame rigid transform (rotation+translation,
    no scale). ~0 means the two point sets are the same shape in different frames."""
    errs = []
    for f in range(0, A.shape[0], stride):
        a, b = A[f] - A[f].mean(0), B[f] - B[f].mean(0)
        U, S, Vt = np.linalg.svd(a.T @ b)
        R = U @ Vt
        if np.linalg.det(R) < 0:
            U[:, -1] *= -1
            R = U @ Vt
        errs.append(np.linalg.norm(a @ R - b, axis=-1).mean())
    return float(np.mean(errs) * 1000.0)


def summarize(name, err, unit, per_joint_names=None):
    """err: [T, J] tensor of per-frame per-joint error."""
    mean, mx = err.mean().item(), err.max().item()
    idx = torch.argmax(err)
    f, j = divmod(int(idx), err.shape[1])
    pj = err.mean(0)
    worst_j = int(torch.argmax(pj))
    line = (f"{name:<38} mean {mean:9.5f} {unit}   max {mx:9.5f} {unit}   "
            f"worst frame {f} joint {j}")
    if per_joint_names:
        line += f" ({per_joint_names[j]})"
    return line, mean, mx, pj, worst_j, f, j


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-clips", type=int, default=3)
    ap.add_argument("--per-subject", type=int, default=0,
                    help="Instead of the first N clips (which are all one subject), take ONE clip "
                         "from each of the first N distinct SUBJECT ids. 0 = off.")
    ap.add_argument("--report", default=os.path.join(REPO, "scripts/quest_rotation_convention_report.md"))
    args = ap.parse_args()

    tree = SkeletonTree.from_mjcf(MJCF)
    names = tree.node_names
    par = tree.parent_indices.tolist()
    out = []

    def P(s=""):
        print(s)
        out.append(s)

    P("# Quest Rotation Convention Verification (Step 1)")
    P("")
    P(f"Skeleton: `{os.path.relpath(MJCF, REPO)}` — {tree.num_joints} joints")
    P(f"Parents: `{tree.parent_indices.tolist()}`")
    P("")

    all_clips = sorted(glob.glob(os.path.join(TEST_DIR, "*.pkl")))
    if args.per_subject:
        # One clip per distinct subject -- the default first-N-clips selection is alphabetical and
        # therefore samples a single subject, which cannot detect per-subject convention drift.
        seen, clips = set(), []
        for c in all_clips:
            m = re.match(r"(SUBJECT\d+)\.", os.path.basename(c))
            sid = m.group(1) if m else None
            if sid and sid not in seen:
                seen.add(sid)
                clips.append(c)
            if len(clips) >= args.per_subject:
                break
    else:
        clips = all_clips[: args.n_clips]
    baseline_stats = []

    for ci, path in enumerate(clips):
        d = joblib.load(path)
        key = list(d.keys())[0]
        e = d[key]
        P(f"## Clip {ci+1}: `{os.path.basename(path)}`")
        P("")
        P("### Stored fields")
        P("")
        P("| key | shape / value | dtype |")
        P("|---|---|---|")
        for k, v in (e.items() if ci == 0 else []):
            if torch.is_tensor(v) or isinstance(v, np.ndarray):
                P(f"| `{k}` | {tuple(np.asarray(v).shape)} | {np.asarray(v).dtype} |")
            else:
                P(f"| `{k}` | {v} | {type(v).__name__} |")
        P("")

        pq_l = t(e["pose_quat"])            # [T,25,4] assumed LOCAL, xyzw
        pq_g = t(e["pose_quat_global"])     # [T,25,4] assumed GLOBAL, xyzw
        root = t(e["root_trans_offset"])    # [T,3]
        T_ = pq_l.shape[0]

        # quaternion ordering sanity: which component is closest to unit for a near-identity rot?
        nrm = pq_l.norm(dim=-1)
        P(f"- frames: **{T_}**, fps: **{e.get('fps')}**")
        P(f"- `pose_quat` norms: mean {nrm.mean():.6f} (unit-quaternion check)")
        P(f"- component means over all frames/joints: "
          f"[{pq_l[...,0].mean():.4f}, {pq_l[...,1].mean():.4f}, "
          f"{pq_l[...,2].mean():.4f}, {pq_l[...,3].mean():.4f}] "
          f"— a dominant LAST component implies **xyzw** (w last)")
        P("")

        # ---------- primary hypothesis: is_local=True, xyzw ----------
        grot, gpos = reconstruct(tree, pq_l, root, is_local=True)
        ang = quat_angle_deg(grot, pq_g)                     # [T,25]
        line, a_mean, a_max, a_pj, a_wj, a_f, a_j = summarize(
            "rot: is_local=True, xyzw", ang, "deg", names)
        P("### Primary hypothesis — `is_local=True`, xyzw, root inside `pose_quat[:,0]`")
        P("")
        P("```")
        P(line)

        pos_lines = []
        for ref_key in ("head_fk_gt_3d", "head_gt_3d"):
            if ref_key in e:
                ref = t(e[ref_key])
                if ref.shape == gpos.shape:
                    perr = (gpos - ref).norm(dim=-1) * 1000.0    # mm, WORLD frame
                    l2, p_raw, p_rawmax, _, _, _, _ = summarize(
                        f"pos vs {ref_key} (raw/world)", perr, "mm", names)
                    P(l2)
                    # These reference fields are stored head-anchored (head_fk_gt_3d[:,HEAD]==0),
                    # i.e. in a per-frame rotated+translated frame -- so the raw world-frame
                    # number above is a FRAME mismatch, not an FK error. The meaningful checks
                    # are (a) do the bone lengths agree (same skeleton?) and (b) does a per-frame
                    # rigid transform reconcile them exactly (same FK?).
                    bl_ours = _bone_lengths(gpos, par)
                    bl_ref = _bone_lengths(ref, par)
                    bl_diff = (bl_ours - bl_ref).abs().max().item()
                    rig = _procrustes_mean_mm(gpos.numpy().astype(np.float64),
                                              ref.numpy().astype(np.float64))
                    P(f"{'  -> max bone-length difference':<38} {bl_diff*1000:9.5f} mm "
                      f"(0 => identical skeleton)")
                    P(f"{'  -> after per-frame rigid align':<38} {rig:9.5f} mm "
                      f"(0 => identical FK, frame-only difference)")
                    pos_lines.append((ref_key, p_raw, p_rawmax, bl_diff * 1000, rig))
        P("```")
        P("")

        if ci == 0:
            P("Per-joint mean angular error (deg):")
            P("")
            P("| idx | joint | mean deg |")
            P("|---|---|---|")
            for j in range(len(names)):
                P(f"| {j} | `{names[j]}` | {a_pj[j].item():.6f} |")
            P("")
        else:
            P(f"Per-joint mean angular error: min {a_pj.min():.6f}, max {a_pj.max():.6f} deg "
              f"(worst joint `{names[a_wj]}`)")
            P("")
        baseline_stats.append((os.path.basename(path), a_mean, a_max,
                               pos_lines[0][4] if pos_lines else None,
                               pos_lines[0][3] if pos_lines else None))

        # ---------- failure-mode sweep (only meaningful if primary is wrong, but always run) ----------
        if ci == 0:
            P("### Failure-mode sweep (clip 1)")
            P("")
            P("Alternative interpretations, to confirm the primary one is not accidentally close:")
            P("")
            P("```")
            variants = {}

            g2, _ = reconstruct(tree, pq_l, root, is_local=False)
            variants["is_local=False, xyzw"] = quat_angle_deg(g2, pq_g)

            g3, _ = reconstruct(tree, wxyz_to_xyzw(pq_l), root, is_local=True)
            variants["is_local=True, stored read as wxyz"] = quat_angle_deg(g3, pq_g)

            ident = torch.zeros_like(pq_l); ident[..., 3] = 1.0
            pq_noroot = pq_l.clone(); pq_noroot[:, 0, :] = ident[:, 0, :]
            g4, _ = reconstruct(tree, pq_noroot, root, is_local=True)
            variants["root forced to identity"] = quat_angle_deg(g4, pq_g)

            # fixed base-frame rotations applied to the root before FK
            for nm, bq in [("upright [.5,.5,.5,.5]", torch.tensor([0.5, 0.5, 0.5, 0.5])),
                           ("quest->smpl [-.5,-.5,-.5,.5]", torch.tensor([-0.5, -0.5, -0.5, 0.5]))]:
                pq_b = pq_l.clone()
                pq_b[:, 0, :] = quat_mul(pq_b[:, 0, :], bq.expand(T_, 4))
                g5, _ = reconstruct(tree, pq_b, root, is_local=True)
                variants[f"root x {nm}"] = quat_angle_deg(g5, pq_g)

            base = f"{'is_local=True, xyzw (PRIMARY)':<38} mean {a_mean:9.5f} deg   max {a_max:9.5f} deg"
            P(base)
            for nm, err in variants.items():
                P(f"{nm:<38} mean {err.mean():9.5f} deg   max {err.max():9.5f} deg")
            P("```")
            P("")

            # is pose_quat_global itself consistent with the tree (independent check)?
            g6, gpos6 = reconstruct(tree, pq_g, root, is_local=False)
            same = quat_angle_deg(g6, pq_g)
            P(f"Sanity: feeding `pose_quat_global` back with `is_local=False` reproduces itself "
              f"to {same.mean():.6f} deg mean — confirms the comparison target is self-consistent.")
            P("")

    # ---------- verdict ----------
    P("## Verdict")
    P("")
    P("| clip | mean rot err (deg) | max rot err (deg) | pos err after rigid align (mm) | max bone-len diff (mm) |")
    P("|---|---|---|---|---|")
    for nm, am, ax, pm, px in baseline_stats:
        pm_s = f"{pm:.5f}" if pm is not None else "n/a"
        px_s = f"{px:.5f}" if px is not None else "n/a"
        P(f"| `{nm}` | {am:.6f} | {ax:.6f} | {pm_s} | {px_s} |")
    P("")
    worst_ang = max(s[1] for s in baseline_stats)
    worst_pos = max((s[3] for s in baseline_stats if s[3] is not None), default=None)
    ok_ang = worst_ang < 0.1
    ok_pos = worst_pos is not None and worst_pos < 1.0
    P(f"- Pass condition mean angular < 0.1 deg: **{'PASS' if ok_ang else 'FAIL'}** "
      f"(worst clip mean {worst_ang:.6f} deg)")
    if worst_pos is not None:
        P(f"- Pass condition mean position < 1 mm: **{'PASS' if ok_pos else 'FAIL'}** "
          f"(worst clip {worst_pos:.6f} mm, measured after removing the reference field's "
          f"per-frame rigid transform -- see note below)")
    P("")

    with open(args.report, "w") as f:
        f.write("\n".join(out) + "\n")
    print(f"\n[report written] {args.report}")


if __name__ == "__main__":
    main()
