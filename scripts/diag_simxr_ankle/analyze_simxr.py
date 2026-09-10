"""Command-space and realised-motion comparison: PHC teacher vs Finetuned Backbone vs SimXR.

Reuses the exact DOF grouping, percentile definitions and clip set from the 49-clip FB diagnostic
(scripts/diag_fb_multiclip), so the SimXR column drops into the same table without re-deriving
anything.

STREAM DISCIPLINE -- five streams, each tagged by whose states it was measured in:
  teacher_cf_fb   frozen PHC teacher mu, queried in the FINETUNED BACKBONE's states (GT refs)
  teacher_own     same frozen weights driving THEMSELVES on AvatarPoser predictions (Naive PHC)
  fb_own          Finetuned Backbone, its own rollout, GT refs
  simxr_own       SimXR student, its own rollout, GT refs (same 56 clips as fb_own)
  (no teacher_cf_simxr: SimXR's env computes its BC target only when `not flags.test`
   -- humanoid_ego_img.py:421 -- so it cannot be harvested from an eval rollout.)

fb_own and simxr_own share the same reference motions and the same robot MJCF (md5-identical), so
their command statistics are directly comparable. They do NOT share reset criteria or observation
layout, so their STATE distributions differ -- that is the object of study, not a nuisance.
"""
import json, os, sys
import joblib, numpy as np
from scipy.spatial.transform import Rotation as sRot

REPO = "/home/sia-irl/AvatarPoser"
D = f"{REPO}/scripts/diag_simxr_ankle"
FBD = f"{REPO}/scripts/diag_fb_multiclip"
sys.path.insert(0, f"{REPO}/scripts")
from evaluate_phc_metrics_quest import FOOT_JOINTS, FPS, H_THRESH, compute_foot_skate  # noqa: E402

SP = ("/tmp/claude-1000/-home-sia-irl-AvatarPoser/"
      "198ae13f-92be-4147-9256-2b72f67cdfbf/scratchpad")
SIMXR_DUMP = "/home/sia-irl/SimXR/output/HumanoidIm/diag_simxr_ankle/phc_positions_for_eval.pkl"
SIMXR_FAILED = "/home/sia-irl/SimXR/output/HumanoidIm/diag_simxr_ankle/failed.pkl"
GRP = {"hip": ["b_l_upleg", "b_r_upleg"], "knee": ["b_l_leg", "b_r_leg"],
       "ankle": ["b_l_foot_twist", "b_r_foot_twist"]}
ANKLE_BODIES = [21, 24]
PCTS = [50, 90, 95, 99]


def group_idx(dn):
    return {g: np.concatenate([np.arange(dn.index(b) * 3, dn.index(b) * 3 + 3) for b in bs])
            for g, bs in GRP.items()}


def dstats(a):
    d = np.abs(np.diff(a, axis=0))
    return {f"p{p}": float(np.percentile(d, p)) for p in PCTS}


def basename(k):
    b = os.path.basename(str(k))
    return b[:-4] if b.endswith(".pkl") else b


def load_npz(p, key):
    z = np.load(p, allow_pickle=True)
    dn = [str(x) for x in z["dof_names"]]
    keys = [basename(k) for k in z["keys"]]
    ns = np.asarray(z["num_steps"]).astype(int)
    out = {}
    for i, k in enumerate(keys):
        n = max(int(ns[i]) - 1, 2)          # PHC's own slicing convention
        out[k] = z[key][:n, i, :]
    return out, dn


def realised(pred_pos, pred_rot, ref_pos):
    gz = float(ref_pos[:, FOOT_JOINTS, 2].min())
    skate = compute_foot_skate(pred_pos, gz) * 100.0
    jerk = (pred_pos[3:] - 3 * pred_pos[2:-1] + 3 * pred_pos[1:-2] - pred_pos[:-3]) * (FPS ** 3)
    jit = float(np.linalg.norm(jerk, axis=-1).mean())
    q = pred_rot[:, ANKLE_BODIES, :].reshape(-1, 4)
    r = sRot.from_quat(q).as_matrix().reshape(pred_rot.shape[0], len(ANKLE_BODIES), 3, 3)
    rel = np.einsum("tjab,tjcb->tjac", r[1:], r[:-1])
    tr = np.clip((np.trace(rel, axis1=2, axis2=3) - 1.0) / 2.0, -1.0, 1.0)
    ang = np.degrees(np.arccos(tr)) * FPS
    footv = np.linalg.norm(np.diff(pred_pos[:, FOOT_JOINTS, :], axis=0), axis=-1) * FPS * 100
    h = pred_pos[:, FOOT_JOINTS, 2] - gz
    c = h < H_THRESH
    trans = float(np.abs(np.diff(c.astype(np.int8), axis=0)).sum()) / (len(h) / FPS)
    return dict(foot_skate_cm_s=skate, jitter_m_s3=jit,
                ankle_angvel_p95=float(np.percentile(ang, 95)),
                foot_vel_p95_cm_s=float(np.percentile(footv, 95)),
                contact_frac=float(c.mean()), contact_trans_per_s=trans)


def main():
    S, dn = load_npz(f"{SP}/mc_simxr.npz", "simxr")
    gi = group_idx(dn)
    fb_rows = {r["clip"]: r for r in json.load(open(f"{FBD}/per_clip.json"))}

    sx_failed = set()
    if os.path.isfile(SIMXR_FAILED):
        sx_failed = {basename(k) for k in joblib.load(SIMXR_FAILED)}
        print(f"[info] {len(sx_failed)} clip(s) terminated in the SimXR run -> excluded")

    # realised kinematics from the SimXR rollout dump (same run, same clips)
    RS = {}
    if os.path.isfile(SIMXR_DUMP):
        d = joblib.load(SIMXR_DUMP)
        for i, k in enumerate(d["motion_keys"]):
            RS[basename(k)] = realised(np.asarray(d["pred_pos_all"][i]),
                                       np.asarray(d["pred_rot_all"][i]),
                                       np.asarray(d["ref_pos_all"][i]))
        print(f"[info] SimXR realised kinematics for {len(RS)} clips")
    else:
        print("[warn] no SimXR rollout dump; realised columns will be missing")

    rows = []
    for clip, a in S.items():
        fb = fb_rows.get(clip)
        if fb is None:
            continue
        r = dict(clip=clip, activity=fb["activity"], category=fb["category"],
                 anchor=fb["anchor"], fb_terminated=fb.get("terminated", False),
                 simxr_terminated=clip in sx_failed, frames=int(a.shape[0]))
        for g, ix in gi.items():
            st = dstats(a[:, ix])
            for p in PCTS:
                r[f"sx_{g}_p{p}"] = st[f"p{p}"]
            r[f"sx_mag_{g}"] = float(np.abs(a[:, ix]).mean())
            v = np.abs(a[:, ix])
            r[f"sx_sat08_{g}"] = float((v > 0.8).mean())
            r[f"sx_sat095_{g}"] = float((v > 0.95).mean())
            r[f"sx_max_{g}"] = float(v.max())
            # carry the FB-side numbers across for the joint table
            for pre, src in (("fb", "s"), ("tc", "t"), ("nv", "n")):
                k = f"{src}_{g}_p95"
                if k in fb:
                    r[f"{pre}_{g}_p95"] = fb[k]
        for p in PCTS:
            r[f"sx_ak_p{p}"] = (r[f"sx_ankle_p{p}"] / r[f"sx_knee_p{p}"]
                                if r[f"sx_knee_p{p}"] > 0 else np.nan)
        r["fb_ak_p95"] = fb.get("s_ak_p95", np.nan)
        r["tc_ak_p95"] = fb.get("t_ak_p95", np.nan)
        r["nv_ak_p95"] = fb.get("n_ak_p95", np.nan)
        for k in ("foot_skate_cm_s", "jitter_m_s3", "ankle_angvel_p95", "contact_trans_per_s"):
            if k in fb:
                r[f"fb_{k}"] = fb[k]
        if clip in RS:
            for k, v in RS[clip].items():
                r[f"sx_{k}"] = v
        rows.append(r)

    json.dump(rows, open(f"{D}/per_clip_simxr.json", "w"), indent=1, default=float)
    print(f"{len(rows)} clips -> per_clip_simxr.json")
    return rows


if __name__ == "__main__":
    main()
