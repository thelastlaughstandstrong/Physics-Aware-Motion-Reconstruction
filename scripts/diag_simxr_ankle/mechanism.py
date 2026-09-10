"""Mechanism measurements behind the SimXR-vs-FB foot-skate gap. Diagnosis only.

Answers, with measurements rather than inference:
  1. Are SimXR's larger ankle |da| values net motion, or self-cancelling chatter?
     -> lag-1 autocorrelation of the delta signal (strongly negative = frame-to-frame reversal).
  2. Is SimXR's lower foot skate an artifact of its feet spending less time near the ground?
     -> contact fraction, using the same band as the foot-skate metric.
  3. Does either policy inherit skate from the reference motion?
     -> foot skate of ref_pos itself.
  4. Where does each policy spend its tracking accuracy?
     -> foot-joint error vs whole-body error.
"""
import json, os, sys
import joblib, numpy as np

REPO = "/home/sia-irl/AvatarPoser"
D = f"{REPO}/scripts/diag_simxr_ankle"
SP = ("/tmp/claude-1000/-home-sia-irl-AvatarPoser/"
      "198ae13f-92be-4147-9256-2b72f67cdfbf/scratchpad")
sys.path.insert(0, f"{REPO}/scripts")
from evaluate_phc_metrics_quest import FOOT_JOINTS, compute_foot_skate  # noqa: E402

GRP = {"hip": ["b_l_upleg", "b_r_upleg"], "knee": ["b_l_leg", "b_r_leg"],
       "ankle": ["b_l_foot_twist", "b_r_foot_twist"]}


def bn(k):
    b = os.path.basename(str(k))
    return b[:-4] if b.endswith(".pkl") else b


def load(p, key):
    z = np.load(p, allow_pickle=True)
    dn = [str(x) for x in z["dof_names"]]
    ks = [bn(k) for k in z["keys"]]
    ns = np.asarray(z["num_steps"]).astype(int)
    return {k: z[key][:max(int(ns[i]) - 1, 2), i, :] for i, k in enumerate(ks)}, dn


def ac1(a):
    d = np.diff(a, axis=0); d = d - d.mean(0, keepdims=True)
    num = (d[1:] * d[:-1]).sum(0)
    den = np.sqrt((d[1:] ** 2).sum(0) * (d[:-1] ** 2).sum(0)) + 1e-12
    return float(np.mean(num / den))


def main():
    SX, dn = load(f"{SP}/mc_simxr.npz", "simxr")
    FB, _ = load(f"{SP}/mc_fb.npz", "student")
    gi = {g: np.concatenate([np.arange(dn.index(b) * 3, dn.index(b) * 3 + 3) for b in bs])
          for g, bs in GRP.items()}
    rows = json.load(open(f"{D}/per_clip_simxr.json"))
    ok = [r["clip"] for r in rows
          if not r["anchor"] and not r["fb_terminated"] and not r["simxr_terminated"]]

    out = {"n_clips": len(ok), "command": {}, "realised": {}}
    for g, ix in gi.items():
        e = {}
        for nm, DD in (("fb", FB), ("simxr", SX)):
            A = [DD[c][:, ix] for c in ok if c in DD]
            e[nm] = dict(
                mean_abs_a=float(np.median([np.abs(x).mean() for x in A])),
                dp95=float(np.median([np.percentile(np.abs(np.diff(x, axis=0)), 95) for x in A])),
                lag1_ac_of_delta=float(np.median([ac1(x) for x in A])))
        out["command"][g] = e

    sx = joblib.load("/home/sia-irl/SimXR/output/HumanoidIm/diag_simxr_ankle/phc_positions_for_eval.pkl")
    fb = joblib.load(f"{REPO}/output/HumanoidIm/quest_distill_trackerr/phc_positions_for_eval_0380160.pkl")
    IX = {"simxr": {bn(k): i for i, k in enumerate(sx["motion_keys"])},
          "fb": {bn(k): i for i, k in enumerate(fb["motion_keys"])}}
    acc = {k: [] for k in ("fb_foot_err", "simxr_foot_err", "fb_body_err", "simxr_body_err",
                           "fb_skate", "simxr_skate", "ref_skate",
                           "fb_contact_frac", "simxr_contact_frac")}
    from evaluate_phc_metrics_quest import H_THRESH
    for c in ok:
        if c not in IX["simxr"] or c not in IX["fb"]:
            continue
        for tag, d in (("fb", fb), ("simxr", sx)):
            i = IX[tag][c]
            P = np.asarray(d["pred_pos_all"][i]); R = np.asarray(d["ref_pos_all"][i])
            n = min(len(P), len(R)); P, R = P[:n], R[:n]
            gz = float(R[:, FOOT_JOINTS, 2].min())
            acc[f"{tag}_foot_err"].append(
                float(np.linalg.norm(P[:, FOOT_JOINTS] - R[:, FOOT_JOINTS], axis=-1).mean() * 100))
            acc[f"{tag}_body_err"].append(float(np.linalg.norm(P - R, axis=-1).mean() * 100))
            acc[f"{tag}_skate"].append(float(compute_foot_skate(P, gz) * 100))
            acc[f"{tag}_contact_frac"].append(
                float(((P[:, FOOT_JOINTS, 2] - gz) < H_THRESH).mean()))
            if tag == "fb":
                acc["ref_skate"].append(float(compute_foot_skate(R, gz) * 100))
    out["realised"] = {k: float(np.median(v)) for k, v in acc.items() if v}
    json.dump(out, open(f"{D}/mechanism.json", "w"), indent=1)

    print(f"n = {out['n_clips']} clips\n")
    print(f"{'group':7s} {'stream':7s} {'mean|a|':>9s} {'|da|p95':>9s} {'lag-1 AC(da)':>13s}")
    for g in GRP:
        for nm in ("fb", "simxr"):
            e = out["command"][g][nm]
            print(f"{g:7s} {nm:7s} {e['mean_abs_a']:9.4f} {e['dp95']:9.4f} "
                  f"{e['lag1_ac_of_delta']:13.3f}")
    r = out["realised"]
    print(f"\n{'':36s} {'FB':>8s} {'SimXR':>8s}")
    for lbl, a, b in [("foot-joint tracking error (cm)", "fb_foot_err", "simxr_foot_err"),
                      ("whole-body tracking error (cm)", "fb_body_err", "simxr_body_err"),
                      ("foot skate (cm/s)", "fb_skate", "simxr_skate"),
                      ("contact fraction", "fb_contact_frac", "simxr_contact_frac")]:
        print(f"  {lbl:34s} {r[a]:8.2f} {r[b]:8.2f}")
    print(f"\n  foot skate of the REFERENCE itself: {r['ref_skate']:.2f} cm/s")
    print(f"-> {D}/mechanism.json")


if __name__ == "__main__":
    main()
