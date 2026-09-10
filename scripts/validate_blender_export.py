"""
Blender-export validation: reconstruct each clip's post-PHC (and pre-PHC) motion from the
EXPORTED AMASS npz ALONE -- the exact file Blender consumes -- and compare against the PHC
simulated rigid-body trajectory saved in the rollout dump.

This is deliberately end-to-end: it re-derives world joint positions from (trans, poses, betas)
through the same interpretation the SMPL-X addon uses, so a mistake anywhere in
export_blender_mesh_motion.py -> export_amass_for_addon.py is caught, not just the last hop.

Coordinate chain under test:
    PHC / Isaac Gym  : Z-up, right-handed, metres, MUJOCO body order, xyzw GLOBAL quats,
                       expressed in PHC's upright_start frame
      -> mujoco -> SMPL_BONE_ORDER_NAMES reorder
      -> undo upright_start (right-multiply by quat[.5,.5,.5,.5])
      -> global -> local (parent-relative)
      -> Z-up -> Y-up for trans and the ROOT rotation only (Q_Z2Y = -90 deg about X);
         every other joint rotation is parent-relative and therefore frame-agnostic
    Blender/SMPL-X   : Y-up, right-handed, metres, SMPLX_JOINT_NAMES order, axis-angle

Run:
    python scripts/validate_blender_export.py
"""
import json
import os
import sys

import joblib
import numpy as np
from scipy.spatial.transform import Rotation as sRot

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "scripts"))
sys.path.insert(0, REPO)

from smpl_sim.smpllib.smpl_joint_names import SMPL_BONE_ORDER_NAMES as JN, SMPL_MUJOCO_NAMES as MJ

EXPORT = os.path.join(REPO, "data/quest_smpl_retarget/blender_export")
DUMP = os.path.join(REPO, "output/HumanoidIm/quest_smpl_charstudy/phc_positions_for_eval.pkl")
BODY = os.path.join(REPO, "support_data/body_models/smplh/neutral/model.npz")
OUT_MD = os.path.join(REPO, "scripts/quest_smpl_blender_export_report.md")
OUT_JSON = os.path.join(REPO, "scripts/quest_smpl_blender_export_report.json")
CLIPS = [("sports", 9), ("gestures", 8), ("martialarts", 0)]

PARENTS = [-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19, 20, 21]
MJ_2_SMPL = [MJ.index(q) for q in JN if q in MJ]
Q_Z2Y = sRot.from_rotvec([-np.pi / 2, 0, 0])
_ROWS = list(range(22)) + [22, 37]          # SMPL-H -> SMPL-24 (Step 6A fix)
SMPLX = ("pelvis", "left_hip", "right_hip", "spine1", "left_knee", "right_knee", "spine2",
         "left_ankle", "right_ankle", "spine3", "left_foot", "right_foot", "neck",
         "left_collar", "right_collar", "head", "left_shoulder", "right_shoulder",
         "left_elbow", "right_elbow", "left_wrist", "right_wrist")
FAILS, LINES = [], []


def PR(s=""):
    print(s)
    LINES.append(s)


def check(name, ok, detail=""):
    PR(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  — {detail}" if detail else ""))
    if not ok:
        FAILS.append(name)


def rest_offsets():
    b = np.load(BODY, allow_pickle=True)
    J = np.array(b["J_regressor"])
    J = J[_ROWS] if J.shape[0] == 52 else J[:24]
    j = J @ np.array(b["v_template"])
    o = j.copy()
    for i in range(1, 24):
        o[i] = j[i] - j[PARENTS[i]]
    return o


def reconstruct(npz, off):
    """AMASS npz -> world joint positions in PHC's Z-up frame, SMPL order. Blender's own
    interpretation, run backwards: undo the Y-up remap, rebuild globals down the chain, FK."""
    trans_yup = np.asarray(npz["trans"], dtype=np.float64)
    poses = np.asarray(npz["poses"], dtype=np.float64)
    T = len(trans_yup)
    # Y-up (x,y,z) came from Z-up (x, z, -y); invert it
    trans = np.stack([trans_yup[:, 0], -trans_yup[:, 2], trans_yup[:, 1]], axis=-1)

    Rl = np.repeat(np.eye(3)[None, None], 24, 1).repeat(T, 0)     # (T,24,3,3), identity default
    Rl[:, 0] = (Q_Z2Y.inv() * sRot.from_rotvec(poses[:, 0:3])).as_matrix()
    for smpl_idx in range(1, 22):
        j = SMPLX.index(SMPLX[smpl_idx])                          # first 22 are 1:1 with SMPL
        Rl[:, smpl_idx] = sRot.from_rotvec(poses[:, j * 3:(j + 1) * 3]).as_matrix()

    Rg = np.empty_like(Rl)
    Rg[:, 0] = Rl[:, 0]
    pos = np.zeros((T, 24, 3))
    pos[:, 0] = trans
    for i in range(1, 24):
        p = PARENTS[i]
        Rg[:, i] = Rg[:, p] @ Rl[:, i]
        pos[:, i] = pos[:, p] + np.einsum("tij,j->ti", Rg[:, p], off[i])
    return pos, Rg


def main():
    d = joblib.load(DUMP)
    off = rest_offsets()
    rendered = list(range(22))                # joints the addon actually drives
    hands = [22, 23]
    out = {}

    PR("# Blender export — validation")
    PR("")
    PR("Each clip reconstructed from the exported AMASS npz alone and compared against the PHC "
       "simulated rigid-body trajectory in `phc_positions_for_eval.pkl`.")

    for tag, idx in CLIPS:
        PR("")
        PR(f"## {tag}")
        PR("")
        res = {}
        for stream, arr_pos, arr_rot, fn in [
                ("post-PHC (pred)", "pred_pos_all", "pred_rot_all", f"amass_pred_cs_{tag}.npz"),
                ("pre-PHC (retarget)", "ref_pos_all", "gt_rot_all", f"amass_gt_cs_{tag}.npz")]:
            npz = np.load(os.path.join(EXPORT, fn), allow_pickle=True)
            ref = np.asarray(d[arr_pos][idx], dtype=np.float64)[:, MJ_2_SMPL]     # SMPL order, Z-up
            rec, Rg = reconstruct(npz, off)
            T = min(len(ref), len(rec))
            ref, rec, Rg = ref[:T], rec[:T], Rg[:T]
            e = np.linalg.norm(rec - ref, axis=-1)

            fps = float(npz["mocap_frame_rate"])
            PR(f"### {stream} — `{fn}`")
            PR("")
            PR(f"- frames **{T}** (dump {len(d[arr_pos][idx])}), fps **{fps:g}**, "
               f"duration **{(T-1)/fps:.3f} s**, betas {np.asarray(npz['betas']).shape[0]} "
               f"(all zero: {bool(np.all(np.asarray(npz['betas'])==0))}), "
               f"gender `{npz['gender']}`")
            PR(f"- joint position error, 22 rendered joints: "
               f"mean **{e[:, rendered].mean()*1000:.3f} mm**, max **{e[:, rendered].max()*1000:.3f} mm**")
            PR(f"- hands (not driven by the addon): mean {e[:, hands].mean()*1000:.1f} mm "
               f"— PHC's mjcf hand bone differs from real SMPL's by 40.8/38.0 mm")
            PR(f"- root position error: mean **{e[:, 0].mean()*1000:.4f} mm**, "
               f"max {e[:, 0].max()*1000:.4f} mm")

            # root orientation: compare reconstructed global root against the dump's own
            ref_rot = np.asarray(d[arr_rot][idx], dtype=np.float64)[:T][:, MJ_2_SMPL]
            und = (sRot.from_quat(ref_rot[:, 0]) * sRot.from_quat([.5, .5, .5, .5]))
            ang = np.degrees((sRot.from_matrix(Rg[:, 0]).inv() * und).magnitude())
            PR(f"- root orientation error: mean **{ang.mean():.3e}°**, max {ang.max():.3e}°")

            check(f"{tag}/{stream}: rendered-joint reconstruction < 1 mm",
                  e[:, rendered].max() < 1e-3, f"max {e[:, rendered].max()*1000:.3f} mm")
            check(f"{tag}/{stream}: root position exact",
                  e[:, 0].max() < 1e-5, f"max {e[:,0].max()*1000:.4f} mm")
            check(f"{tag}/{stream}: root orientation exact", ang.max() < 1e-3,
                  f"max {ang.max():.2e}°")
            check(f"{tag}/{stream}: frame count and fps preserved",
                  T == len(d[arr_pos][idx]) and fps == 30.0)
            check(f"{tag}/{stream}: finite everywhere", np.isfinite(rec).all())

            # joint order: a permutation would show cm-scale error on the worst joint
            worst = int(e[:, rendered].mean(0).argmax())
            check(f"{tag}/{stream}: joint ordering correct",
                  e[:, rendered].mean(0).max() < 1e-3,
                  f"worst joint {JN[worst]} {e[:, rendered].mean(0)[worst]*1000:.3f} mm")

            # left/right: structural pairs, in the body's own lateral frame
            lat = rec[:, 1] - rec[:, 2]
            lat /= np.linalg.norm(lat, axis=-1, keepdims=True)
            s = np.einsum("tjc,tc->tj", rec - rec[:, 0:1], lat)
            lr_ok = all((s[:, l] - s[:, r] > 0).all() for l, r in [(1, 2), (16, 17)])
            check(f"{tag}/{stream}: left/right correct (hips & shoulders never cross)", lr_ok)

            # temporal continuity of the RECONSTRUCTION, vs the source trajectory
            # rendered joints only: the two hand leaves carry PHC's own 40.8/38.0 mm hand bone,
            # so their per-frame step legitimately differs by a few mm. They are not driven by
            # the addon and cannot appear in a render, so including them here measures the known
            # skeleton difference rather than any discontinuity the export introduced.
            st_rec = np.linalg.norm(np.diff(rec[:, rendered], axis=0), axis=-1).max(1)
            st_ref = np.linalg.norm(np.diff(ref[:, rendered], axis=0), axis=-1).max(1)
            PR(f"- temporal continuity: max per-frame joint step "
               f"{st_rec.max()*1000:.1f} mm (source {st_ref.max()*1000:.1f} mm)")
            check(f"{tag}/{stream}: no discontinuity introduced by the export",
                  abs(st_rec.max() - st_ref.max()) < 1e-3,
                  f"Δ {abs(st_rec.max()-st_ref.max())*1000:.3f} mm")
            PR("")
            res[stream] = dict(frames=int(T), fps=fps,
                               mean_mm=float(e[:, rendered].mean() * 1000),
                               max_mm=float(e[:, rendered].max() * 1000),
                               hands_mm=float(e[:, hands].mean() * 1000),
                               root_mm=float(e[:, 0].mean() * 1000),
                               root_rot_deg=float(ang.mean()),
                               max_step_mm=float(st_rec.max() * 1000))
        out[tag] = res

    PR("")
    if FAILS:
        PR(f"**VALIDATION FAILED — {len(FAILS)} check(s):** {FAILS}")
    else:
        PR("**VALIDATION PASSED** — every export reconstructs the PHC trajectory to sub-millimetre.")
    json.dump({"clips": out, "passed": not FAILS, "failed": FAILS},
              open(OUT_JSON, "w"), indent=2, default=float)
    open(OUT_MD, "w").write("\n".join(LINES) + "\n")
    return 1 if FAILS else 0


if __name__ == "__main__":
    sys.exit(main())
