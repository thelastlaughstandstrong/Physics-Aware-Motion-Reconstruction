"""Multi-clip analysis of the ankle action-delta tail (diagnostic only).

THREE action streams are kept strictly separate:
  student   FB (trackerr@380k) control, on GT motions, in the states FB itself created.
  teacher*  frozen PHC teacher mu, evaluated on the SAME observations the student consumed.
            COUNTERFACTUAL -- what the teacher would do in the student's states. This is the
            correct comparator for "does the student imitate the teacher's tail".
  naive     the SAME frozen PHC weights rolling out on AvatarPoser predictions, i.e. the Naive
            PHC baseline along ITS OWN trajectory, its own states, its own clip lengths (42
            frames shorter). NEVER pooled with `teacher*` -- different states, different motions.

Realised kinematics come from the existing full-test FB dump, using the definitions in
evaluate_phc_metrics_quest.py (FOOT_JOINTS=[21,24], H_THRESH=0.033, FPS=30, per-sequence
empirical ground) so they are comparable with results_comparison.md.
"""
import json, os, sys
import joblib, numpy as np
from scipy.spatial.transform import Rotation as sRot

REPO = "/home/sia-irl/AvatarPoser"
D = f"{REPO}/scripts/diag_fb_multiclip"
sys.path.insert(0, f"{REPO}/scripts")
from evaluate_phc_metrics_quest import (FOOT_JOINTS, FPS, H_THRESH, GROUND_Z,  # noqa: E402
                                        compute_foot_skate)

SP = sys.argv[1] if len(sys.argv) > 1 else \
    "/tmp/claude-1000/-home-sia-irl-AvatarPoser/198ae13f-92be-4147-9256-2b72f67cdfbf/scratchpad"
FB_DUMP = f"{REPO}/output/HumanoidIm/quest_distill_trackerr/phc_positions_for_eval_0380160.pkl"
GRP = {"hip": ["b_l_upleg", "b_r_upleg"], "knee": ["b_l_leg", "b_r_leg"],
       "ankle": ["b_l_foot_twist", "b_r_foot_twist"]}
ANKLE_BODIES = [21, 24]   # b_l_foot_twist, b_r_foot_twist (body idx = dof idx + 1)
PCTS = [50, 90, 95, 99]


def group_idx(dof_names):
    return {g: np.concatenate([np.arange(dof_names.index(b) * 3, dof_names.index(b) * 3 + 3)
                               for b in bs]) for g, bs in GRP.items()}


def dstats(a):
    """|Δa| percentiles over (frame, dim) pairs of one group."""
    d = np.abs(np.diff(a, axis=0))
    return {f"p{p}": float(np.percentile(d, p)) for p in PCTS}


def load_actions(npz):
    z = np.load(npz, allow_pickle=True)
    dn = [str(x) for x in z["dof_names"]]
    # clip names contain ".pkl" MID-STRING (SUBJECT086.walking.pkl_2889_3300), so strip only
    # the trailing extension -- .replace(".pkl","") silently mangles every key.
    keys = [os.path.basename(str(k)) for k in z["keys"]]
    keys = [k[:-4] if k.endswith(".pkl") else k for k in keys]
    ns = np.asarray(z["num_steps"]).astype(int)
    out = {}
    for i, k in enumerate(keys):
        n = max(int(ns[i]) - 1, 2)          # PHC's own slicing: arr[:(num_steps-1), idx]
        e = {"student": z["student"][:n, i, :]}
        if "teacher" in z.files:
            e["teacher"] = z["teacher"][:n, i, :]
        out[k] = e
    return out, dn


def realised(pred_pos, pred_rot, ref_pos):
    """Realised kinematics for one clip, definitions from evaluate_phc_metrics_quest."""
    gz = float(ref_pos[:, FOOT_JOINTS, 2].min())
    skate = compute_foot_skate(pred_pos, gz) * 100.0                       # cm/s
    jerk = (pred_pos[3:] - 3 * pred_pos[2:-1] + 3 * pred_pos[1:-2] - pred_pos[:-3]) * (FPS ** 3)
    jit = float(np.linalg.norm(jerk, axis=-1).mean())
    # realised ankle angular speed (deg/s) from consecutive global body rotations
    q = pred_rot[:, ANKLE_BODIES, :].reshape(-1, 4)
    r = sRot.from_quat(q).as_matrix().reshape(pred_rot.shape[0], len(ANKLE_BODIES), 3, 3)
    rel = np.einsum("tjab,tjcb->tjac", r[1:], r[:-1])
    tr = np.clip((np.trace(rel, axis1=2, axis2=3) - 1.0) / 2.0, -1.0, 1.0)
    ang = np.degrees(np.arccos(tr)) * FPS
    # foot-contact transitions per second (per-sequence empirical ground + H_THRESH)
    h = pred_pos[:, FOOT_JOINTS, 2] - gz
    c = h < H_THRESH
    trans = float(np.abs(np.diff(c.astype(np.int8), axis=0)).sum()) / (len(h) / FPS)
    return dict(foot_skate_cm_s=skate, jitter_m_s3=jit,
                ankle_angvel_p95=float(np.percentile(ang, 95)),
                ankle_angvel_mean=float(ang.mean()),
                contact_frac=float(c.mean()), contact_trans_per_s=trans)


def main():
    fb_np, nv_np = f"{SP}/mc_fb.npz", f"{SP}/mc_naive.npz"
    A, dn = load_actions(fb_np)
    gi = group_idx(dn)
    N = None
    if os.path.isfile(nv_np):
        N, dn_n = load_actions(nv_np)
        assert dn_n == dn, "dof order differs between runs"
    else:
        print("[warn] naive npz missing -- own-trajectory stream will be absent")

    # A clip that terminates early has its post-reset frames appended by PHC's dump slicing, which
    # would inject a spurious |da| discontinuity. Read the run's OWN failed-key record and exclude
    # those clips from the aggregate rather than averaging over a reset.
    failed = set()
    fp = f"{REPO}/output/HumanoidIm/diag_mc_fb/failed.pkl"
    if os.path.isfile(fp):
        for k in joblib.load(fp):
            b = os.path.basename(str(k))
            failed.add(b[:-4] if b.endswith(".pkl") else b)
        print(f"[info] {len(failed)} clip(s) terminated early in this run -> excluded: {sorted(failed)}")
    else:
        print("[warn] no failed.pkl found; cannot exclude early-terminating clips")

    # The naive run has its OWN failure set (different policy states, different motions). A clip
    # that FB tracks fine may still fail for naive; drop only that clip's NAIVE columns.
    nv_failed = set()
    nfp = f"{REPO}/output/HumanoidIm/diag_mc_naive/failed.pkl"
    if os.path.isfile(nfp):
        for k in joblib.load(nfp):
            b = os.path.basename(str(k))
            nv_failed.add(b[:-4] if b.endswith(".pkl") else b)
        print(f"[info] {len(nv_failed)} clip(s) terminated in the NAIVE run -> naive columns "
              f"dropped for: {sorted(nv_failed)}")

    meta = {r["clip"]: r for r in json.load(open(f"{D}/selected_clips.json"))}
    d = joblib.load(FB_DUMP)
    dk = {os.path.basename(str(k))[:-4]: i for i, k in enumerate(d["motion_keys"])}

    rows = []
    for clip, e in A.items():
        m = meta.get(clip)
        if m is None:
            print(f"[warn] no metadata for {clip}"); continue
        r = dict(clip=clip, activity=m["activity"], category=m["category"],
                 anchor=bool(m.get("anchor", False)), terminated=clip in failed,
                 frames=int(e["student"].shape[0]))
        for g, ix in gi.items():
            s, t = e["student"][:, ix], e["teacher"][:, ix]
            ss, ts = dstats(s), dstats(t)
            for p in PCTS:
                r[f"s_{g}_p{p}"] = ss[f"p{p}"]
                r[f"t_{g}_p{p}"] = ts[f"p{p}"]
                r[f"ratio_{g}_p{p}"] = ss[f"p{p}"] / ts[f"p{p}"] if ts[f"p{p}"] > 0 else np.nan
            r[f"imit_err_{g}"] = float(np.abs(s - t).mean())
            r[f"mag_{g}"] = float(np.abs(s).mean())
            if N and clip in N and clip not in nv_failed:
                r[f"n_{g}_p95"] = dstats(N[clip]["student"][:, ix])["p95"]
                r[f"n_{g}_p99"] = dstats(N[clip]["student"][:, ix])["p99"]
        for p in PCTS:
            r[f"s_ak_p{p}"] = r[f"s_ankle_p{p}"] / r[f"s_knee_p{p}"] if r[f"s_knee_p{p}"] > 0 else np.nan
            r[f"t_ak_p{p}"] = r[f"t_ankle_p{p}"] / r[f"t_knee_p{p}"] if r[f"t_knee_p{p}"] > 0 else np.nan
        r["naive_terminated"] = clip in nv_failed
        if N and clip in N and clip not in nv_failed:
            r["n_ak_p95"] = r["n_ankle_p95"] / r["n_knee_p95"] if r.get("n_knee_p95", 0) > 0 else np.nan
        if clip in dk:
            i = dk[clip]
            r.update(realised(np.asarray(d["pred_pos_all"][i]), np.asarray(d["pred_rot_all"][i]),
                              np.asarray(d["ref_pos_all"][i])))
        rows.append(r)

    json.dump(rows, open(f"{D}/per_clip.json", "w"), indent=1, default=float)
    print(f"{len(rows)} clips analysed -> per_clip.json")
    return rows


if __name__ == "__main__":
    main()
