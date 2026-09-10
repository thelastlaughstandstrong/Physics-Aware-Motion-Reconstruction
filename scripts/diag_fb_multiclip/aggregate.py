"""Aggregate the multi-clip ankle-tail diagnostic into category stats, distributions,
examples and correlations. Reporting only -- computes nothing new about the models."""
import json, os, sys
import numpy as np

REPO = "/home/sia-irl/AvatarPoser"
D = f"{REPO}/scripts/diag_fb_multiclip"
CATS = ["upper-body", "lower-body", "full-body/dynamic", "other"]
L = []
def PR(s=""):
    print(s); L.append(s)


def q(v, p):
    v = np.asarray([x for x in v if np.isfinite(x)])
    return float(np.percentile(v, p)) if len(v) else float("nan")


def _rank(v):
    return np.argsort(np.argsort(np.asarray(v, float))).astype(float)


def partial_spearman(x, y, z):
    """Rank correlation of x and y after linearly removing z (on ranks).

    Needed because clip-level motion intensity drives BOTH the action tail and the realised
    kinematics: a raw x-y correlation mostly restates 'vigorous clips are vigorous'. Controlling
    for the TEACHER's ankle tail isolates the student-specific excess.
    """
    x, y, z = np.asarray(x, float), np.asarray(y, float), np.asarray(z, float)
    m = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
    x, y, z = x[m], y[m], z[m]
    n = len(x)
    if n < 10:
        return float("nan"), float("nan"), n
    rx, ry, rz = _rank(x), _rank(y), _rank(z)
    def resid(a):
        A = np.vstack([np.ones_like(rz), rz]).T
        return a - A @ np.linalg.lstsq(A, a, rcond=None)[0]
    ex, ey = resid(rx), resid(ry)
    r = float(np.corrcoef(ex, ey)[0, 1])
    rng = np.random.RandomState(0)
    null = np.array([np.corrcoef(ex, rng.permutation(ey))[0, 1] for _ in range(2000)])
    return r, float((np.abs(null) >= abs(r)).mean()), n


def sign_test(vals, ref=1.0):
    """Two-sided exact binomial test that vals are not centred on ref."""
    from math import comb
    v = [x for x in vals if np.isfinite(x)]
    k = sum(1 for x in v if x > ref); n = len(v)
    if n == 0:
        return 0, 0, float("nan")
    p = sum(comb(n, i) for i in range(0, min(k, n - k) + 1)) / 2 ** n * 2
    return k, n, min(p, 1.0)


def spearman(x, y):
    """Rank correlation + a permutation p-value (no scipy.stats dependency assumptions)."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    m = np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]
    n = len(x)
    if n < 8:
        return float("nan"), float("nan"), n
    rx = np.argsort(np.argsort(x)).astype(float)
    ry = np.argsort(np.argsort(y)).astype(float)
    r = float(np.corrcoef(rx, ry)[0, 1])
    rng = np.random.RandomState(0)
    null = np.array([np.corrcoef(rx, rng.permutation(ry))[0, 1] for _ in range(2000)])
    p = float((np.abs(null) >= abs(r)).mean())
    return r, p, n


def main():
    rows = json.load(open(f"{D}/per_clip.json"))
    main_rows = [r for r in rows if not r["anchor"] and not r.get("terminated", False)]
    dropped = [r for r in rows if not r["anchor"] and r.get("terminated", False)]
    anchor = [r for r in rows if r["anchor"]]
    has_naive = any("n_ankle_p95" in r for r in rows)

    PR("# Multi-clip validation: is the ankle action-delta tail inherited from the PHC teacher?")
    PR("")
    PR(f"Diagnostic only — nothing retrained, no loss/reward/simulator/metric/retargeting change.")
    PR(f"**{len(main_rows)} clips** from the Quest2 test set where BOTH the Finetuned Backbone "
       f"(`trackerr`@380k) and the Naive PHC baseline survive to the end of the clip, plus the "
       f"single-clip anchor from the earlier diagnostic (reported separately, never pooled).")
    if dropped:
        PR(f"*{len(dropped)} selected clip(s) terminated early in this batched run despite "
           f"surviving the full-test run and were excluded from the aggregate "
           f"({', '.join('`'+r['clip']+'`' for r in dropped)}); PHC appends post-reset frames to a "
           f"terminated episode, which would inject a spurious |Δa| jump.*")
    PR("")
    PR("**Three action streams, kept separate throughout:**")
    PR("")
    PR("| stream | policy | motions | states visited |")
    PR("|---|---|---|---|")
    PR("| `student` | FB `trackerr`@380k | ground-truth Quest | its own |")
    PR("| `teacher*` (counterfactual) | frozen PHC teacher | ground-truth Quest | **the student's** — mu queried on the same obs |")
    PR("| `naive` (own trajectory) | frozen PHC teacher, same weights | AvatarPoser predictions | **its own**, 42 frames shorter |")
    PR("")
    PR("`teacher*` answers *does the student imitate the teacher's tail*. `naive` answers *what "
       "does this policy do when it drives itself*. They are different states and are never pooled.")
    PR("")

    # ---------- Q1/Q2: per-category aggregates ----------
    PR("## Aggregate by motion category")
    PR("")
    PR("Median across clips (IQR in brackets). `|Δa| p95`, per joint group.")
    PR("")
    for g in ["hip", "knee", "ankle"]:
        PR(f"### {g}")
        PR("")
        PR("| category | n | student p95 | teacher* p95 | student/teacher | naive p95 |")
        PR("|---|---|---|---|---|---|")
        for c in CATS:
            sub = [r for r in main_rows if r["category"] == c]
            if not sub: continue
            s = [r[f"s_{g}_p95"] for r in sub]; t = [r[f"t_{g}_p95"] for r in sub]
            rt = [r[f"ratio_{g}_p95"] for r in sub]
            nv = [r.get(f"n_{g}_p95", np.nan) for r in sub]
            nvs = f"{q(nv,50):.3f} [{q(nv,25):.3f}–{q(nv,75):.3f}]" if has_naive else "—"
            PR(f"| {c} | {len(sub)} | {q(s,50):.3f} [{q(s,25):.3f}–{q(s,75):.3f}] "
               f"| {q(t,50):.3f} [{q(t,25):.3f}–{q(t,75):.3f}] "
               f"| **{q(rt,50):.2f}** [{q(rt,25):.2f}–{q(rt,75):.2f}] | {nvs} |")
        s = [r[f"s_{g}_p95"] for r in main_rows]; t = [r[f"t_{g}_p95"] for r in main_rows]
        rt = [r[f"ratio_{g}_p95"] for r in main_rows]
        PR(f"| **all** | {len(main_rows)} | {q(s,50):.3f} | {q(t,50):.3f} | **{q(rt,50):.2f}** | |")
        PR("")

    # ---------- Q1: teacher ankle vs knee/hip ----------
    PR("## Q1 — does the teacher itself have a heavier ankle tail than knee/hip?")
    PR("")
    PR("Teacher* ankle/knee and ankle/hip |Δa| p95 ratio, per clip.")
    PR("")
    PR("| category | n | teacher ankle/knee | >1 in | student ankle/knee | naive ankle/knee |")
    PR("|---|---|---|---|---|---|")
    for c in CATS + ["ALL"]:
        sub = main_rows if c == "ALL" else [r for r in main_rows if r["category"] == c]
        if not sub: continue
        tk = [r["t_ak_p95"] for r in sub]; sk = [r["s_ak_p95"] for r in sub]
        nk = [r.get("n_ak_p95", np.nan) for r in sub]
        frac = float(np.mean([x > 1 for x in tk if np.isfinite(x)]))
        nks = f"{q(nk,50):.2f}" if has_naive else "—"
        PR(f"| {c} | {len(sub)} | **{q(tk,50):.2f}** [{q(tk,25):.2f}–{q(tk,75):.2f}] "
           f"| {frac*100:.0f}% | {q(sk,50):.2f} | {nks} |")
    k, n, pv = sign_test([r["t_ak_p95"] for r in main_rows])
    PR("")
    PR(f"Teacher ankle |Δa| p95 exceeds its own knee on **{k}/{n} clips** "
       f"(exact two-sided sign test p = {pv:.2e}).")
    PR("")

    # ---------- fairness: counterfactual teacher vs naive own-trajectory ----------
    if has_naive:
        PR("## The two teacher streams, side by side (kept separate on purpose)")
        PR("")
        PR("Same frozen weights. `teacher*` is evaluated in the **student's** states on "
           "ground-truth motions; `naive` drives **itself** on AvatarPoser predictions. If the "
           "ankle-heavy tail were an artefact of the states FB visits, `naive` would not show it.")
        PR("")
        PR("| category | n | teacher* ankle/knee | naive ankle/knee | teacher* ankle p95 | naive ankle p95 |")
        PR("|---|---|---|---|---|---|")
        for c in CATS + ["ALL"]:
            sub = main_rows if c == "ALL" else [r for r in main_rows if r["category"] == c]
            sub = [r for r in sub if np.isfinite(r.get("n_ak_p95", np.nan))]
            if not sub: continue
            PR(f"| {c} | {len(sub)} | {q([r['t_ak_p95'] for r in sub],50):.2f} "
               f"| **{q([r['n_ak_p95'] for r in sub],50):.2f}** "
               f"| {q([r['t_ankle_p95'] for r in sub],50):.3f} "
               f"| {q([r['n_ankle_p95'] for r in sub],50):.3f} |")
        nk, nn, npv = sign_test([r["n_ak_p95"] for r in main_rows if np.isfinite(r.get("n_ak_p95", np.nan))])
        PR("")
        PR(f"Naive's own ankle |Δa| p95 exceeds its own knee on **{nk}/{nn} clips** "
           f"(sign test p = {npv:.2e}).")
        PR("")

    # ---------- Q2: does the student reproduce it ----------
    PR("## Q2 — does the student reproduce the teacher's ankle tail?")
    PR("")
    PR("student/teacher* |Δa| ratio at each percentile (median across clips, all categories).")
    PR("")
    PR("| group | p50 | p90 | p95 | p99 |")
    PR("|---|---|---|---|---|")
    for g in ["hip", "knee", "ankle"]:
        PR(f"| {g} | " + " | ".join(
            f"{q([r[f'ratio_{g}_p{p}'] for r in main_rows], 50):.2f}" for p in [50, 90, 95, 99]) + " |")
    PR("")
    within = [r for r in main_rows if abs(r["ratio_ankle_p95"] - 1) <= 0.15]
    PR(f"Student ankle p95 within ±15% of the teacher on **{len(within)}/{len(main_rows)} clips** "
       f"({len(within)/len(main_rows)*100:.0f}%).")
    PR("")

    # ---------- Q5: imitation error ----------
    PR("## Q5 — is the student's extra ankle imitation error large enough to matter?")
    PR("")
    PR("| category | n | ankle err | knee err | ankle/knee err | ankle/knee after magnitude-normalising |")
    PR("|---|---|---|---|---|---|")
    for c in CATS + ["ALL"]:
        sub = main_rows if c == "ALL" else [r for r in main_rows if r["category"] == c]
        if not sub: continue
        ea = [r["imit_err_ankle"] for r in sub]; ek = [r["imit_err_knee"] for r in sub]
        rat = [r["imit_err_ankle"] / r["imit_err_knee"] for r in sub if r["imit_err_knee"] > 0]
        nrm = [(r["imit_err_ankle"] / r["imit_err_knee"]) / (r["mag_ankle"] / r["mag_knee"])
               for r in sub if r["imit_err_knee"] > 0 and r["mag_knee"] > 0 and r["mag_ankle"] > 0]
        PR(f"| {c} | {len(sub)} | {q(ea,50):.4f} | {q(ek,50):.4f} | {q(rat,50):.2f} | {q(nrm,50):.2f} |")
    PR("")

    # ---------- Q4: correlations ----------
    PR("## Q4 — do larger ankle tails go with higher realised ankle velocity, jitter, foot skate?")
    PR("")
    PR("Spearman rank correlation across clips, with a 2000-sample permutation p-value.")
    PR("")
    PR("| x | y | rho | p | n |")
    PR("|---|---|---|---|---|")
    corrs = {}
    for xn, xk in [("student ankle |Δa| p95", "s_ankle_p95"),
                   ("teacher* ankle |Δa| p95", "t_ankle_p95"),
                   ("student/teacher ankle ratio", "ratio_ankle_p95"),
                   ("ankle imitation error", "imit_err_ankle")]:
        for yn, yk in [("realised ankle angvel p95", "ankle_angvel_p95"),
                       ("jitter", "jitter_m_s3"), ("foot skate", "foot_skate_cm_s"),
                       ("contact transitions/s", "contact_trans_per_s")]:
            x = [r.get(xk, np.nan) for r in main_rows]; y = [r.get(yk, np.nan) for r in main_rows]
            r_, p_, n_ = spearman(x, y)
            corrs[f"{xk}|{yk}"] = dict(rho=r_, p=p_, n=n_)
            star = " **" if (np.isfinite(p_) and p_ < 0.05) else ""
            PR(f"| {xn} | {yn} | {r_:+.2f}{star} | {p_:.3f} | {n_} |")
    PR("")
    PR("`**` = p < 0.05 (uncorrected; 16 tests, so treat isolated marks with caution).")
    PR("")
    PR("The raw rows above are **confounded by motion intensity** — a vigorous clip has both a "
       "bigger action tail and faster realised motion, for the student and the teacher alike "
       "(note the teacher\u2019s rows are nearly as strong as the student\u2019s). The question that "
       "actually bears on a student-side intervention is whether the student\u2019s tail predicts "
       "realised instability *beyond what the teacher\u2019s own tail already predicts*:")
    PR("")
    PR("| x (controlling for teacher* ankle p95) | y | partial rho | p | n |")
    PR("|---|---|---|---|---|")
    z = [r["t_ankle_p95"] for r in main_rows]
    for yn, yk in [("realised ankle angvel p95", "ankle_angvel_p95"), ("jitter", "jitter_m_s3"),
                   ("foot skate", "foot_skate_cm_s"), ("contact transitions/s", "contact_trans_per_s")]:
        x = [r["s_ankle_p95"] for r in main_rows]; y = [r.get(yk, np.nan) for r in main_rows]
        r_, p_, n_ = partial_spearman(x, y, z)
        corrs[f"partial|s_ankle_p95|{yk}"] = dict(rho=r_, p=p_, n=n_)
        star = " **" if (np.isfinite(p_) and p_ < 0.05) else ""
        PR(f"| student ankle |Δa| p95 | {yn} | {r_:+.2f}{star} | {p_:.3f} | {n_} |")
    PR("")

    # ---------- distribution + examples ----------
    PR("## Distribution across clips (not just the mean)")
    PR("")
    PR("student/teacher* ankle |Δa| p95 ratio:")
    PR("")
    rt = sorted(r["ratio_ankle_p95"] for r in main_rows)
    PR("| min | p10 | p25 | median | p75 | p90 | max |")
    PR("|---|---|---|---|---|---|---|")
    PR("| " + " | ".join(f"{q(rt,p):.2f}" for p in [0, 10, 25, 50, 75, 90, 100]) + " |")
    PR("")
    PR("teacher* ankle/knee |Δa| p95 ratio:")
    PR("")
    tk = sorted(r["t_ak_p95"] for r in main_rows)
    PR("| min | p10 | p25 | median | p75 | p90 | max |")
    PR("|---|---|---|---|---|---|---|")
    PR("| " + " | ".join(f"{q(tk,p):.2f}" for p in [0, 10, 25, 50, 75, 90, 100]) + " |")
    PR("")

    PR("## Representative clips")
    PR("")
    srt = sorted(main_rows, key=lambda r: r["t_ak_p95"])
    picks = [("lowest teacher ankle/knee", srt[0]), ("", srt[1]),
             ("median", srt[len(srt)//2]), ("", srt[len(srt)//2 + 1]),
             ("highest teacher ankle/knee", srt[-2]), ("", srt[-1])]
    PR("| | clip | category | teacher ankle/knee | student ankle/knee | s/t ankle p95 | realised ankle angvel p95 | skate |")
    PR("|---|---|---|---|---|---|---|---|")
    for lbl, r in picks:
        PR(f"| {lbl} | `{r['clip']}` | {r['category']} | {r['t_ak_p95']:.2f} | {r['s_ak_p95']:.2f} "
           f"| {r['ratio_ankle_p95']:.2f} | {r.get('ankle_angvel_p95',float('nan')):.0f}°/s "
           f"| {r.get('foot_skate_cm_s',float('nan')):.1f} |")
    PR("")
    if anchor:
        a = anchor[0]
        PR("### Anchor clip (earlier single-clip diagnostic, reported separately)")
        PR("")
        PR(f"`{a['clip']}` — teacher ankle/knee **{a['t_ak_p95']:.2f}**, "
           f"student/teacher ankle p95 **{a['ratio_ankle_p95']:.2f}**. "
           f"Published single-clip values were 2.05 and 1.03.")
        PR("")
        PR("> Note: FB **terminates early** on this clip in the full test run, so it was not a "
           "representative choice — it is retained only as the pipeline's reproducibility check.")
        PR("")

    json.dump(dict(n_clips=len(main_rows), corrs=corrs), open(f"{D}/aggregate.json", "w"),
              indent=1, default=float)
    open(f"{D}/MULTICLIP_FINDINGS.md", "w").write("\n".join(L) + "\n")
    print(f"\n-> {D}/MULTICLIP_FINDINGS.md")


if __name__ == "__main__":
    main()
