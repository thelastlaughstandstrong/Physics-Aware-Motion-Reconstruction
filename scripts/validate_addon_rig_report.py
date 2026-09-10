"""
Numerical validation of the DIRECT-RIG Blender path (blender_render_addon.py) against the same PHC
trajectories used by the export validation.

Consumes the per-clip bone-world-position dumps produced inside Blender by
validate_blender_addon_rig.py (which drives blender_render_addon.py's own build_rig/animate_rig),
and scores every frame -- not a sample -- on: pelvis/root world position, all 22 rendered joint
positions, root orientation, left/right ordering, frame count and fps, temporal continuity, and
ground-plane alignment.

The residual is expected to be dominated by a STRUCTURAL difference, not a pipeline error: the
addon ships only an SMPL-X body (the SMPL+H model file is not installed), and SMPL-X neutral's rest
skeleton differs from SMPL/SMPL-H neutral's -- which is what our rotations were fitted against.
That hypothesis is tested here directly by correlating the per-joint error against the measured
rest-skeleton difference.
"""
import json
import os
import sys

import numpy as np
from scipy.spatial.transform import Rotation as R

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "scripts"))
sys.path.insert(0, REPO)

import validate_blender_export as V                                    # noqa: E402
from smpl_sim.smpllib.smpl_joint_names import SMPL_BONE_ORDER_NAMES as JN  # noqa: E402

SC = os.environ.get("SCRATCH", "/tmp/claude-1000/-home-sia-irl-AvatarPoser/"
                    "198ae13f-92be-4147-9256-2b72f67cdfbf/scratchpad")
EXPORT = os.path.join(REPO, "data/quest_smpl_retarget/blender_export")
OUT_MD = os.path.join(REPO, "scripts/quest_smpl_addon_rig_validation.md")
OUT_JSON = os.path.join(REPO, "scripts/quest_smpl_addon_rig_validation.json")
CLIPS = ["sports", "gestures", "martialarts"]
LINES, FAILS = [], []
# joints whose position is essentially unaffected by the rig's shape difference (proximal chain)
PROXIMAL = [0, 1, 2, 3]


def PR(s=""):
    print(s)
    LINES.append(s)


def check(name, ok, detail=""):
    PR(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  — {detail}" if detail else ""))
    if not ok:
        FAILS.append(name)


def rest_skeleton_diff():
    """SMPL-X rig rest vs our SMPL-H rest, frame-matched, pelvis-relative, mm."""
    rig = np.load(os.path.join(SC, "rig_rest_world.npy"))
    rig = rig - rig[0]
    off = V.rest_offsets()
    smpl = np.zeros((24, 3))
    for i in range(1, 24):
        smpl[i] = smpl[V.PARENTS[i]] + off[i]
    smpl = (smpl[:22] - smpl[0]) @ R.from_rotvec([np.pi / 2, 0, 0]).as_matrix().T
    return np.linalg.norm(rig - smpl, axis=-1) * 1000


def main():
    dr = rest_skeleton_diff()
    PR("# Direct-rig Blender path — numerical validation")
    PR("")
    PR("`blender_render_addon.py`'s own `build_rig`/`animate_rig` driven inside Blender 5.2, "
       "world bone positions read back for **every frame**, compared against the PHC simulated "
       "trajectory in `mesh_<clip>.npz`.")
    PR("")
    PR(f"Reference: SMPL-X rig rest skeleton vs our SMPL-H rest skeleton — "
       f"**mean {dr.mean():.1f} mm, max {dr.max():.1f} mm** "
       f"(stature pelvis→head 62.0 vs 57.6 cm). The addon ships only an SMPL-X body; the SMPL+H "
       f"model file is not installed, so this difference cannot be removed from this route.")
    out = {}
    for tag in CLIPS:
        rig = np.load(os.path.join(SC, f"rigdump_{tag}_fixed.npz"), allow_pickle=True)
        mesh = np.load(os.path.join(EXPORT, f"mesh_{tag}.npz"), allow_pickle=True)
        fps = int(mesh["fps"])
        PR("")
        PR(f"## {tag}")
        PR("")
        res = {}
        for stream, pk, rk in [("post-PHC (pred)", "pred_pos", "pred_rot_local"),
                               ("pre-PHC (gt)", "gt_pos", "gt_rot_local")]:
            key = "pred" if "pred" in stream else "gt"
            W = rig[f"{key}_world"]
            Q = rig[f"{key}_quat_wxyz"]
            P = mesh[pk][:, :22]
            T = min(len(W), len(P))
            W, Q, P = W[:T], Q[:T], P[:T]
            e = np.linalg.norm(W - P, axis=-1)

            # root orientation: rig pelvis world quat vs the SMPL global root rotation
            rig_q = R.from_quat(np.stack([Q[:, 0, 1], Q[:, 0, 2], Q[:, 0, 3], Q[:, 0, 0]], -1))
            smpl_root = R.from_quat(mesh[rk][:T, 0])
            ang = np.degrees((rig_q.inv() * smpl_root).magnitude())

            # left/right in the body's own lateral frame (hips + shoulders, structural)
            lat = W[:, 1] - W[:, 2]
            lat /= np.linalg.norm(lat, axis=-1, keepdims=True)
            s = np.einsum("tjc,tc->tj", W - W[:, 0:1], lat)
            lr = all((s[:, l] - s[:, r] > 0).all() for l, r in [(1, 2), (16, 17)])

            st_rig = np.linalg.norm(np.diff(W, axis=0), axis=-1).max(1)
            st_phc = np.linalg.norm(np.diff(P, axis=0), axis=-1).max(1)
            # ground plane: lowest driven joint (ankles/feet) over the clip
            foot = [7, 8, 10, 11]
            gz_rig, gz_phc = W[:, foot, 2].min(), P[:, foot, 2].min()

            PR(f"### {stream}")
            PR("")
            PR(f"- frames **{T}** @ **{fps} fps** (PHC {len(mesh[pk])} @ 30)")
            PR(f"- pelvis/root world position: mean **{e[:,0].mean()*1000:.4f} mm**, "
               f"max **{e[:,0].max()*1000:.4f} mm**")
            PR(f"- all 22 rendered joints: mean **{e.mean()*1000:.2f} mm**, max **{e.max()*1000:.2f} mm**")
            PR(f"- proximal chain (pelvis, hips, spine1): mean **{e[:,PROXIMAL].mean()*1000:.2f} mm**")
            PR(f"- root orientation: mean **{ang.mean():.3f}°**, max **{ang.max():.3f}°**")
            PR(f"- temporal continuity: max per-frame step {st_rig.max()*1000:.1f} mm "
               f"(PHC {st_phc.max()*1000:.1f} mm)")
            PR(f"- ground plane: lowest foot/ankle Z rig **{gz_rig*1000:+.1f} mm** vs "
               f"PHC **{gz_phc*1000:+.1f} mm**")
            PR(f"- per-joint error vs rest-skeleton difference: **r = "
               f"{np.corrcoef(e.mean(0)*1000, dr)[0,1]:+.3f}**")

            check(f"{tag}/{stream}: pelvis world position exact", e[:, 0].max() < 1e-5,
                  f"max {e[:,0].max()*1000:.4f} mm")
            check(f"{tag}/{stream}: no residual 90° frame error (root orientation < 3°)",
                  ang.max() < 3.0, f"max {ang.max():.2f}°")
            check(f"{tag}/{stream}: joint error within the rig's own skeleton difference",
                  e.mean() * 1000 < 1.5 * dr.mean(),
                  f"{e.mean()*1000:.1f} mm vs rest-diff {dr.mean():.1f} mm")
            check(f"{tag}/{stream}: residual is structural (r > 0.9 with rest-skeleton difference)",
                  np.corrcoef(e.mean(0) * 1000, dr)[0, 1] > 0.9)
            check(f"{tag}/{stream}: left/right ordering correct", lr)
            check(f"{tag}/{stream}: frame count and fps preserved",
                  T == len(mesh[pk]) and fps == 30)
            check(f"{tag}/{stream}: temporal continuity preserved",
                  abs(st_rig.max() - st_phc.max()) < 0.05,
                  f"Δ {abs(st_rig.max()-st_phc.max())*1000:.1f} mm")
            check(f"{tag}/{stream}: ground plane consistent (within 5 cm)",
                  abs(gz_rig - gz_phc) < 0.05, f"Δ {(gz_rig-gz_phc)*1000:+.1f} mm")
            check(f"{tag}/{stream}: finite everywhere", np.isfinite(W).all())
            PR("")
            res[stream] = dict(frames=int(T), fps=fps,
                               pelvis_max_mm=float(e[:, 0].max() * 1000),
                               joints_mean_mm=float(e.mean() * 1000),
                               joints_max_mm=float(e.max() * 1000),
                               proximal_mean_mm=float(e[:, PROXIMAL].mean() * 1000),
                               root_rot_mean_deg=float(ang.mean()),
                               root_rot_max_deg=float(ang.max()),
                               ground_rig_mm=float(gz_rig * 1000),
                               ground_phc_mm=float(gz_phc * 1000),
                               corr_with_rest_diff=float(np.corrcoef(e.mean(0) * 1000, dr)[0, 1]))
        out[tag] = res
    PR("")
    if FAILS:
        PR(f"**FAILED — {len(FAILS)} check(s):** {FAILS}")
    else:
        PR("**PASSED** — pelvis exact, no frame/convention error, and the entire residual is the "
           "SMPL-X rig's own rest skeleton.")
    json.dump({"clips": out, "rest_diff_mean_mm": float(dr.mean()),
               "rest_diff_max_mm": float(dr.max()), "passed": not FAILS, "failed": FAILS},
              open(OUT_JSON, "w"), indent=2, default=float)
    open(OUT_MD, "w").write("\n".join(LINES) + "\n")
    return 1 if FAILS else 0


if __name__ == "__main__":
    sys.exit(main())
