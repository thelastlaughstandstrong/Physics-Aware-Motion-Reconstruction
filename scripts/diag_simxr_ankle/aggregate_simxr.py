"""Build the PHC vs FB vs SimXR comparison tables. Reporting only."""
import json, os
import numpy as np

REPO = "/home/sia-irl/AvatarPoser"
D = f"{REPO}/scripts/diag_simxr_ankle"
CATS = ["upper-body", "lower-body", "full-body/dynamic", "other"]
L = []
def PR(s=""):
    print(s); L.append(s)

def q(v, p):
    v = np.asarray([x for x in v if np.isfinite(x)])
    return float(np.percentile(v, p)) if len(v) else float("nan")

def sign_test(vals, ref=1.0):
    from math import comb
    v = [x for x in vals if np.isfinite(x)]
    k = sum(1 for x in v if x > ref); n = len(v)
    if n == 0: return 0, 0, float("nan")
    p = sum(comb(n, i) for i in range(0, min(k, n - k) + 1)) / 2 ** n * 2
    return k, n, min(p, 1.0)

def main():
    rows = json.load(open(f"{D}/per_clip_simxr.json"))
    # a clip enters the command comparison only if BOTH systems tracked it to the end
    ok = [r for r in rows if not r["anchor"] and not r["fb_terminated"] and not r["simxr_terminated"]]
    sx_only = [r for r in rows if not r["anchor"] and not r["simxr_terminated"]]

    PR("# Does SimXR suppress PHC's ankle-heavy commands, or realise them better?")
    PR("")
    PR("Diagnosis only — nothing retrained, smoothed or retuned.")
    PR("")
    PR(f"**{len(ok)} clips** where the Finetuned Backbone and SimXR both survive to the end of the "
       f"clip (of {len(sx_only)} SimXR survivors in the shared 56-clip set). Same reference motions, "
       f"same `quest_humanoid` MJCF (md5-identical across the two repositories), same DOF grouping "
       f"and percentile definitions as the 49-clip FB diagnostic.")
    PR("")

    # ---------- command space ----------
    PR("## Command space — |Δa| per joint group")
    PR("")
    PR("Median across clips. `teacher*` = frozen PHC mu in **FB's** states; `naive` = the same "
       "frozen weights on **their own** rollout; `FB` and `SimXR` are each along **their own** "
       "rollout. Reference motions are identical for teacher*, FB and SimXR; `naive` tracks "
       "AvatarPoser predictions instead.")
    PR("")
    for g in ["hip", "knee", "ankle"]:
        PR(f"### {g} — |Δa| p95")
        PR("")
        PR("| category | n | teacher* | naive | FB | **SimXR** |")
        PR("|---|---|---|---|---|---|")
        for c in CATS:
            sub = [r for r in ok if r["category"] == c]
            if not sub: continue
            PR(f"| {c} | {len(sub)} | {q([r.get(f'tc_{g}_p95',np.nan) for r in sub],50):.3f} "
               f"| {q([r.get(f'nv_{g}_p95',np.nan) for r in sub],50):.3f} "
               f"| {q([r.get(f'fb_{g}_p95',np.nan) for r in sub],50):.3f} "
               f"| **{q([r[f'sx_{g}_p95'] for r in sub],50):.3f}** |")
        PR(f"| **all** | {len(ok)} | {q([r.get(f'tc_{g}_p95',np.nan) for r in ok],50):.3f} "
           f"| {q([r.get(f'nv_{g}_p95',np.nan) for r in ok],50):.3f} "
           f"| {q([r.get(f'fb_{g}_p95',np.nan) for r in ok],50):.3f} "
           f"| **{q([r[f'sx_{g}_p95'] for r in ok],50):.3f}** |")
        PR("")

    PR("### SimXR percentile sweep (all categories)")
    PR("")
    PR("| group | p50 | p90 | p95 | p99 | mean \\|a\\| | max \\|a\\| | >0.8 | >0.95 |")
    PR("|---|---|---|---|---|---|---|---|---|")
    for g in ["hip", "knee", "ankle"]:
        PR(f"| {g} | " + " | ".join(f"{q([r[f'sx_{g}_p{p}'] for r in ok],50):.3f}" for p in [50,90,95,99])
           + f" | {q([r[f'sx_mag_{g}'] for r in ok],50):.3f}"
           + f" | {q([r[f'sx_max_{g}'] for r in ok],50):.3f}"
           + f" | {q([r[f'sx_sat08_{g}'] for r in ok],50):.4f}"
           + f" | {q([r[f'sx_sat095_{g}'] for r in ok],50):.4f} |")
    PR("")

    # ---------- the headline ----------
    PR("## Ankle ÷ knee |Δa| p95 — the asymmetry, per policy")
    PR("")
    PR("| category | n | teacher* | naive | FB | **SimXR** |")
    PR("|---|---|---|---|---|---|")
    for c in CATS + ["ALL"]:
        sub = ok if c == "ALL" else [r for r in ok if r["category"] == c]
        if not sub: continue
        PR(f"| {c} | {len(sub)} | {q([r['tc_ak_p95'] for r in sub],50):.2f} "
           f"| {q([r['nv_ak_p95'] for r in sub],50):.2f} "
           f"| {q([r['fb_ak_p95'] for r in sub],50):.2f} "
           f"| **{q([r['sx_ak_p95'] for r in sub],50):.2f}** |")
    k, n, p = sign_test([r["sx_ak_p95"] for r in ok])
    PR("")
    PR(f"SimXR's ankle |Δa| p95 exceeds its own knee on **{k}/{n} clips** (sign test p = {p:.2e}).")
    PR("")
    PR("Distribution of SimXR ankle/knee across clips:")
    PR("")
    v = [r["sx_ak_p95"] for r in ok]
    PR("| min | p10 | p25 | median | p75 | p90 | max |")
    PR("|---|---|---|---|---|---|---|")
    PR("| " + " | ".join(f"{q(v,p):.2f}" for p in [0,10,25,50,75,90,100]) + " |")
    PR("")

    # ---------- realised ----------
    PR("## Realised physical behaviour — same clips, same metric definitions")
    PR("")
    PR("| metric | FB | **SimXR** | ratio SimXR/FB |")
    PR("|---|---|---|---|")
    for lbl, fk, sk in [("ankle angular velocity p95 (°/s)", "fb_ankle_angvel_p95", "sx_ankle_angvel_p95"),
                        ("foot linear velocity p95 (cm/s)", None, "sx_foot_vel_p95_cm_s"),
                        ("contact transitions / s", "fb_contact_trans_per_s", "sx_contact_trans_per_s"),
                        ("foot skate (cm/s)", "fb_foot_skate_cm_s", "sx_foot_skate_cm_s"),
                        ("jitter (m/s³)", "fb_jitter_m_s3", "sx_jitter_m_s3")]:
        s = q([r.get(sk, np.nan) for r in ok], 50)
        if fk is None:
            PR(f"| {lbl} | — | **{s:.1f}** | — |")
        else:
            f = q([r.get(fk, np.nan) for r in ok], 50)
            PR(f"| {lbl} | {f:.1f} | **{s:.1f}** | {s/f:.2f} |" if np.isfinite(f) and f != 0
               else f"| {lbl} | {f:.1f} | **{s:.1f}** | — |")
    PR("")
    PR("Per-clip paired comparison (SimXR − FB), median of the per-clip difference:")
    PR("")
    PR("| metric | median Δ | SimXR lower in |")
    PR("|---|---|---|")
    for lbl, fk, sk in [("foot skate (cm/s)", "fb_foot_skate_cm_s", "sx_foot_skate_cm_s"),
                        ("jitter (m/s³)", "fb_jitter_m_s3", "sx_jitter_m_s3"),
                        ("ankle angvel p95 (°/s)", "fb_ankle_angvel_p95", "sx_ankle_angvel_p95"),
                        ("contact transitions / s", "fb_contact_trans_per_s", "sx_contact_trans_per_s")]:
        d = [r[sk] - r[fk] for r in ok if np.isfinite(r.get(sk, np.nan)) and np.isfinite(r.get(fk, np.nan))]
        if not d: continue
        PR(f"| {lbl} | {np.median(d):+.2f} | {np.mean([x < 0 for x in d])*100:.0f}% of clips |")
    PR("")

    json.dump(dict(n=len(ok)), open(f"{D}/aggregate_simxr.json", "w"), indent=1)
    open(f"{D}/SIMXR_FINDINGS.md", "w").write("\n".join(L) + "\n")
    print(f"\n-> {D}/SIMXR_FINDINGS.md")


if __name__ == "__main__":
    main()
