"""Score saved validation rollouts for every arm/checkpoint. Reporting only.

MPJPE / MPJRE / jitter / foot skate / success come from evaluate_phc_metrics_quest.py -- the same
module that produced every other Quest2 number -- invoked as a subprocess per dump so the
definitions cannot drift. Post-sim foot-joint error and the ankle |da| statistics are added here.

Per PROTOCOL.md the arms are compared as the MEAN +- SD over the 8 predefined checkpoints, never a
single best checkpoint.
"""
import glob, json, os, subprocess, sys
import joblib, numpy as np

REPO = "/home/sia-irl/AvatarPoser"
D = f"{REPO}/scripts/ablation_footweight"
EVALPY = f"{REPO}/scripts/evaluate_phc_metrics_quest.py"
PY = "/home/sia-irl/anaconda3/envs/rlgpu/bin/python"
FOOT_BODIES = [21, 24]
GRP = {"knee": ["b_l_leg", "b_r_leg"], "ankle": ["b_l_foot_twist", "b_r_foot_twist"]}
STEPS = [330000, 340000, 350000, 360000, 370000, 380000, 390000, 400000]
ARMS = [("w=1.0 (baseline)", "footw1"), ("w=2.0", "footw2"), ("w=4.0", "footw4")]


def bn(k):
    b = os.path.basename(str(k))
    return b[:-4] if b.endswith(".pkl") else b


def foot_error_cm(dump):
    d = joblib.load(dump)
    out = []
    for i in range(len(d["motion_keys"])):
        P = np.asarray(d["pred_pos_all"][i]); R = np.asarray(d["ref_pos_all"][i])
        n = min(len(P), len(R))
        out.append(float(np.linalg.norm(P[:n, FOOT_BODIES] - R[:n, FOOT_BODIES], axis=-1).mean() * 100))
    return float(np.mean(out))


def ankle_stats(npz):
    z = np.load(npz, allow_pickle=True)
    dn = [str(x) for x in z["dof_names"]]
    ns = np.asarray(z["num_steps"]).astype(int)
    gi = {g: np.concatenate([np.arange(dn.index(b) * 3, dn.index(b) * 3 + 3) for b in bs])
          for g, bs in GRP.items()}
    per = {g: [] for g in gi}
    for i in range(len(ns)):
        n = max(int(ns[i]) - 1, 2)
        for g, ix in gi.items():
            per[g].append(float(np.percentile(np.abs(np.diff(z["student"][:n, i, ix], axis=0)), 95)))
    ap, kp = float(np.median(per["ankle"])), float(np.median(per["knee"]))
    return ap, (ap / kp if kp > 0 else float("nan"))


def score_one(prefix):
    dump, npz = f"{prefix}_dump.pkl", f"{prefix}_actions.npz"
    if not (os.path.isfile(dump) and os.path.isfile(npz)):
        return None
    j = f"{prefix}_metrics.json"
    if not os.path.isfile(j):
        subprocess.run([PY, EVALPY, "--pos_dump", dump, "--out", j],
                       check=True, capture_output=True)
    m = json.load(open(j))
    m = m.get("metrics", m)          # evaluate_phc_metrics_quest nests them under "metrics"
    ap, ak = ankle_stats(npz)
    return dict(foot_err_cm=foot_error_cm(dump),
                foot_skate_cm_s=m.get("foot_skate_cm_s"),
                mpjpe_mm=m.get("eg_mpjpe_mm", m.get("pos_error_cm", np.nan) * 10),
                mpjre_deg=m.get("rot_error_deg"), jitter=m.get("jitter_m_s3"),
                success=m.get("success_rate_phc_survival"),
                ankle_dp95=ap, ankle_knee_ratio=ak)


KEYS = [("foot_err_cm", "foot err (cm)", 2), ("foot_skate_cm_s", "foot skate (cm/s)", 2),
        ("mpjpe_mm", "MPJPE (mm)", 2), ("mpjre_deg", "MPJRE (deg)", 2),
        ("jitter", "jitter", 1), ("success", "success", 4),
        ("ankle_dp95", "ankle |da| p95", 4), ("ankle_knee_ratio", "ankle/knee", 2)]


def main():
    res = {}
    for label, tag in ARMS:
        rows = []
        for s in STEPS:
            r = score_one(f"{D}/evals/{tag}_{s}")
            if r: r["step"] = s; rows.append(r)
        if rows: res[label] = rows
        print(f"{label:18s} {len(rows)}/{len(STEPS)} checkpoints scored")
    if not res:
        print("\nNo rollouts scored yet — run eval_checkpoint.py first."); return 1
    json.dump(res, open(f"{D}/arm_scores.json", "w"), indent=1, default=float)

    print("\n" + "=" * 92)
    print(f"{'metric':20s}" + "".join(f"{l:>24s}" for l, _ in ARMS if l in res))
    print("=" * 92)
    for k, lbl, dp in KEYS:
        line = f"{lbl:20s}"
        for label, _ in ARMS:
            if label not in res: continue
            v = np.array([r[k] for r in res[label] if r.get(k) is not None], dtype=float)
            line += f"{v.mean():>16.{dp}f} ±{v.std(ddof=1):<7.{dp}f}" if len(v) > 1 else f"{'n/a':>24s}"
        print(line)
    print("=" * 92)
    print("mean ± SD over the 8 predefined checkpoints (330k–400k), 90 validation clips.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
