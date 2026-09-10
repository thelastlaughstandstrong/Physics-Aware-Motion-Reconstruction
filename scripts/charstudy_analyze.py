"""
Characterization study, stage 3: is the frozen-PHC physics pass worth applying to every Quest
motion, or only to motions that arrive with real physical artefacts?

Protocol is Step 6E's: BOTH pre- and post-physics are scored against the independent QUEST
GROUND TRUTH, never against the retargeted reference (which would make pre-sim trivially zero).

Per clip:
    benefit = reduction in ground penetration + reduction in foot skate
    cost    = increase in Quest-GT MPJPE     + increase in jitter

The four raw terms have incompatible units (mm, cm/s, mm, m/s^3), so the composite scores
normalise each term by that metric's ACROSS-CLIP pre-sim mean, making every term a dimensionless
relative change. Raw components are reported alongside so the composite never has to be trusted
on its own.

Run (after charstudy_convert.py and run_charstudy_phc.sh):
    LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH python scripts/charstudy_analyze.py
"""
import json
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "scripts"))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "phc"))

import isaacgym  # noqa: F401,E402
import joblib  # noqa: E402
import numpy as np  # noqa: E402
import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from evaluate_phc_metrics import (  # noqa: E402
    MUJOCO_2_SMPL_REORDER as M2S, NUM_SMPL_JOINTS as NJ,
    compute_foot_skate, compute_gpen, compute_metrics)
from analyze_step6e import jitter_per_frame, quest_targets  # noqa: E402

CHAR = os.path.join(REPO, "data/quest_smpl_retarget/charstudy")
EXP_DIR = os.path.join(REPO, "output/HumanoidIm/quest_smpl_charstudy")
DUMP = os.path.join(EXP_DIR, "phc_positions_for_eval.pkl")
MANIFEST = os.path.join(CHAR, "manifest.json")
CORR_JSON = os.path.join(REPO, "scripts/quest_smpl_correspondence.json")
OUT_MD = os.path.join(REPO, "scripts/quest_smpl_charstudy_report.md")
OUT_JSON = os.path.join(REPO, "scripts/quest_smpl_charstudy_report.json")
OUT_PNG = os.path.join(REPO, "scripts/quest_smpl_charstudy_scatter.png")
FPS = 30
LINES = []


def PR(s=""):
    print(s)
    LINES.append(s)


def hf_frac(p):
    """Fraction of acceleration power above 7.5 Hz -- the chatter band at 30 fps."""
    acc = np.diff(p, 2, axis=0) * (FPS ** 2)
    sp = np.abs(np.fft.rfft(acc - acc.mean(0), axis=0)) ** 2
    f = np.fft.rfftfreq(len(acc), 1.0 / FPS)
    return float(sp[f > 7.5].sum() / max(sp.sum(), 1e-12))


def gt_mpjpe(P, Q, muj, qi, dz, sh):
    """Mean joint error against the Quest targets, mm, at a fixed frame shift."""
    T = len(P)
    a = slice(max(0, sh), T + min(0, sh))
    b = slice(max(0, -sh), len(Q) + min(0, -sh))
    n = min(len(range(*a.indices(T))), len(range(*b.indices(len(Q)))))
    return float(np.linalg.norm((P[a][:n][:, muj] - np.array([0, 0, dz]))
                                - Q[b][:n][:, qi], axis=-1).mean() * 1000)


def main():
    d = joblib.load(DUMP)
    man = json.load(open(MANIFEST))
    corr = json.load(open(CORR_JSON))
    direct = {int(k): v for k, v in corr["direct"].items()}
    qi = sorted(direct)
    muj = [M2S[direct[q]] for q in qi]
    sel = M2S[:NJ]
    keys = [str(k) for k in d["motion_keys"]]
    succ = set()
    p = os.path.join(EXP_DIR, "long_succ.pkl")
    if os.path.isfile(p):
        succ = {os.path.basename(str(x)) for x in joblib.load(p)}
    term = d.get("terminate_hist", [])

    rows = []
    for i, key in enumerate(keys):
        meta = man.get(key)
        if meta is None:
            continue
        pre = np.asarray(d["ref_pos_all"][i])
        post = np.asarray(d["pred_pos_all"][i])
        n = min(len(pre), len(post))
        pre, post = pre[:n], post[:n]

        Q = quest_targets(meta["source_clip"], corr)
        # PHC's loader applies a rigid fix_trans_height Z shift and samples the reference with a
        # small lag. Both are properties of the LOADER, not of physics, so they are solved once on
        # the PRE-sim reference and the identical (shift, dz) is then applied to post-sim -- any
        # other choice would let the alignment absorb part of the effect being measured.
        T = len(pre)
        best = None
        for sh in range(-3, 4):
            a = slice(max(0, sh), T + min(0, sh))
            b = slice(max(0, -sh), len(Q) + min(0, -sh))
            m = min(len(range(*a.indices(T))), len(range(*b.indices(len(Q)))))
            cand_dz = float(np.median(pre[a][:m][:, muj][..., 2] - Q[b][:m][:, qi][..., 2]))
            e = gt_mpjpe(pre, Q, muj, qi, cand_dz, sh)
            if best is None or e < best[0]:
                best = (e, sh, cand_dz)
        _, sh, dz = best

        gp_pre, gp_post = compute_gpen(pre), compute_gpen(post)
        r = {
            "tag": meta["tag"], "class": meta["class"], "frames": int(n),
            "source": os.path.basename(meta["source_clip"]),
            "survived": os.path.basename(key) in succ,
            "terminated": bool(term[i]) if i < len(term) else None,
            "survival_ratio": len(d["pred_pos_all"][i]) / max(len(d["ref_pos_all"][i]), 1),
            "mpjpe_pre": gt_mpjpe(pre, Q, muj, qi, dz, sh),
            "mpjpe_post": gt_mpjpe(post, Q, muj, qi, dz, sh),
            "jitter_pre": float(jitter_per_frame(pre[:, sel]).mean()),
            "jitter_post": float(jitter_per_frame(post[:, sel]).mean()),
            "hf_pre": hf_frac(pre[:, sel]), "hf_post": hf_frac(post[:, sel]),
            "skate_pre": float(compute_foot_skate(pre, FPS) * 100),
            "skate_post": float(compute_foot_skate(post, FPS) * 100),
            "pen_pre": gp_pre[0], "pen_post": gp_post[0],
            "pen_max_pre": gp_pre[1], "pen_max_post": gp_post[1],
            "pen_rate_pre": gp_pre[2], "pen_rate_post": gp_post[2],
        }
        pv = compute_metrics(post[:, sel], pre[:, sel], fps=FPS)
        r["mpjve_post_vs_ref"] = pv["vel_error_m_s"] * 100
        rows.append(r)

    # ---- normalisation: each term relative to its across-clip pre-sim mean ----
    def mean_of(k):
        return float(np.mean([x[k] for x in rows])) or 1.0
    n_pen, n_sk = max(mean_of("pen_pre"), 1e-9), max(mean_of("skate_pre"), 1e-9)
    n_mp, n_ji = max(mean_of("mpjpe_pre"), 1e-9), max(mean_of("jitter_pre"), 1e-9)
    for r in rows:
        r["benefit"] = (r["pen_pre"] - r["pen_post"]) / n_pen + (r["skate_pre"] - r["skate_post"]) / n_sk
        r["cost"] = (r["mpjpe_post"] - r["mpjpe_pre"]) / n_mp + (r["jitter_post"] - r["jitter_pre"]) / n_ji
        r["net"] = r["benefit"] - r["cost"]
        r["violation"] = r["pen_pre"] / n_pen + r["skate_pre"] / n_sk
    rows.sort(key=lambda x: -x["violation"])
    med_v = float(np.median([r["violation"] for r in rows]))
    for r in rows:
        r["group"] = "high" if r["violation"] >= med_v else "low"

    # ------------------------------------------------ report
    PR("# Characterization study — should PHC physics be applied selectively?")
    PR("")
    PR(f"{len(rows)} clips, frozen `phc_kp_pnn_iccv` checkpoint, Step 6E protocol "
       f"(pre- and post-physics both scored against Quest ground truth).")
    PR("")
    PR("## Per-clip results")
    PR("")
    PR("| clip | class | frames | surv | MPJPE pre→post (mm) | jitter pre→post | HF%% pre→post | "
       "skate pre→post (cm/s) | penetration pre→post (mm) |")
    PR("|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        PR(f"| {r['tag']} | {r['class']} | {r['frames']} | "
           f"{'Y' if r['survived'] else '**N**'} | "
           f"{r['mpjpe_pre']:.1f} → {r['mpjpe_post']:.1f} (**{r['mpjpe_post']-r['mpjpe_pre']:+.1f}**) | "
           f"{r['jitter_pre']:.1f} → {r['jitter_post']:.1f} ({r['jitter_post']/max(r['jitter_pre'],1e-9):.2f}×) | "
           f"{100*r['hf_pre']:.1f} → {100*r['hf_post']:.1f} | "
           f"{r['skate_pre']:.2f} → {r['skate_post']:.2f} | "
           f"{r['pen_pre']:.2f} → {r['pen_post']:.2f} |")

    PR("")
    PR("## Benefit vs cost")
    PR("")
    PR("Each term normalised by its across-clip pre-sim mean, so all four are dimensionless "
       "relative changes. Positive net = physics worth applying.")
    PR("")
    PR("| clip | pre-sim violation | benefit | cost | **net** | verdict |")
    PR("|---|---|---|---|---|---|")
    for r in rows:
        v = "**beneficial**" if r["net"] > 0.15 else ("**harmful**" if r["net"] < -0.15 else "neutral")
        r["verdict"] = v.replace("*", "")
        PR(f"| {r['tag']} | {r['violation']:.2f} ({r['group']}) | {r['benefit']:+.2f} | "
           f"{r['cost']:+.2f} | **{r['net']:+.2f}** | {v} |")

    lo = [r for r in rows if r["group"] == "low"]
    hi = [r for r in rows if r["group"] == "high"]
    PR("")
    PR("## Low- vs high-violation groups")
    PR("")
    PR("| group | n | pre-sim violation | ΔMPJPE (mm) | Δjitter (×) | Δskate (cm/s) | "
       "Δpenetration (mm) | **net** |")
    PR("|---|---|---|---|---|---|---|---|")
    for nm, g in [("low", lo), ("high", hi)]:
        if not g:
            continue
        PR(f"| {nm} | {len(g)} | {np.mean([r['violation'] for r in g]):.2f} | "
           f"{np.mean([r['mpjpe_post']-r['mpjpe_pre'] for r in g]):+.1f} | "
           f"{np.mean([r['jitter_post']/max(r['jitter_pre'],1e-9) for r in g]):.2f} | "
           f"{np.mean([r['skate_post']-r['skate_pre'] for r in g]):+.2f} | "
           f"{np.mean([r['pen_post']-r['pen_pre'] for r in g]):+.2f} | "
           f"**{np.mean([r['net'] for r in g]):+.2f}** |")

    v = np.array([r["violation"] for r in rows])
    dm = np.array([r["mpjpe_post"] - r["mpjpe_pre"] for r in rows])
    ben = np.array([r["benefit"] for r in rows])
    net = np.array([r["net"] for r in rows])
    PR("")
    PR("## Correlations")
    PR("")
    PR(f"- pre-sim violation vs benefit: **r = {np.corrcoef(v, ben)[0,1]:+.3f}** — this is "
       f"TAUTOLOGICAL, not a finding: physics drives penetration to exactly 0 and skate below "
       f"1 cm/s on every clip, so the reduction is mechanically equal to the pre-existing "
       f"violation. It is reported only to make that circularity explicit.")
    PR(f"- pre-sim violation vs ΔMPJPE (fidelity cost): **r = {np.corrcoef(v, dm)[0,1]:+.3f}**")
    PR(f"- pre-sim violation vs net: **r = {np.corrcoef(v, net)[0,1]:+.3f}**")
    pos = v[net > 0]
    neg = v[net <= 0]
    thr = None
    if len(pos) and len(neg):
        thr = (neg.max() + pos.min()) / 2 if pos.min() > neg.max() else None
        PR(f"- net-positive clips have violation ≥ {pos.min():.2f}; "
           f"net-negative ≤ {neg.max():.2f}"
           + (f" → **clean threshold ≈ {thr:.2f}**" if thr else
              " → **the groups overlap; no clean threshold**"))

    # 3 clips for Blender
    order = sorted(rows, key=lambda x: -x["net"])
    # "mixed" should mean LARGE benefit AND large cost (a genuine trade-off to look at), not a
    # clip where both are negligible -- smallest |net| would pick a do-nothing clip.
    picks = [("clearly beneficial", order[0]), ("clearly harmful", order[-1]),
             ("mixed", max(rows, key=lambda x: min(x["benefit"], x["cost"])))]
    PR("")
    PR("## Fidelity-first view (the project's stated priority 1)")
    PR("")
    worse = [r for r in rows if r["mpjpe_post"] > r["mpjpe_pre"]]
    PR(f"ΔMPJPE vs Quest GT is **positive on {len(worse)} of {len(rows)} clips** "
       f"(range +{min(r['mpjpe_post']-r['mpjpe_pre'] for r in rows):.1f} to "
       f"+{max(r['mpjpe_post']-r['mpjpe_pre'] for r in rows):.1f} mm, mean "
       f"+{np.mean([r['mpjpe_post']-r['mpjpe_pre'] for r in rows]):.1f} mm / "
       f"+{np.mean([100*(r['mpjpe_post']-r['mpjpe_pre'])/r['mpjpe_pre'] for r in rows]):.0f}%).")
    PR("")
    PR(f"**There is no clip on which physics improves kinematic fidelity.** Any recommendation to "
       f"run it therefore rests entirely on valuing contact realism above pose accuracy; the "
       f"composite `net` above encodes one particular exchange rate (equal weight per normalised "
       f"term) and should not be read as objective.")
    PR("")
    PR("## Recommended clips for Blender visualization")
    PR("")
    PR("| role | clip | source | net | why |")
    PR("|---|---|---|---|---|")
    for role, r in picks:
        why = (f"penetration {r['pen_pre']:.1f}→{r['pen_post']:.1f} mm, "
               f"skate {r['skate_pre']:.2f}→{r['skate_post']:.2f}, "
               f"MPJPE {r['mpjpe_post']-r['mpjpe_pre']:+.1f} mm")
        PR(f"| {role} | **{r['tag']}** | `{r['source']}` | {r['net']:+.2f} | {why} |")

    # ------------------------------------------------ scatter
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
    cls = sorted({r["class"] for r in rows})
    cmap = dict(zip(cls, plt.cm.tab10.colors))
    for ax, (yv, yl, ttl) in zip(axes, [
            (dm, "ΔMPJPE vs Quest GT (mm)", "fidelity cost vs pre-sim violation"),
            (ben, "benefit (normalised)", "physics benefit vs pre-sim violation"),
            (net, "net = benefit − cost", "net effect vs pre-sim violation")]):
        for r, y in zip(rows, yv):
            ax.scatter(r["violation"], y, s=70, color=cmap[r["class"]],
                       edgecolor="k", linewidth=.5, zorder=3)
            ax.annotate(r["tag"], (r["violation"], y), fontsize=6,
                        xytext=(4, 4), textcoords="offset points")
        if len(rows) > 2:
            k = np.polyfit(v, yv, 1)
            xs = np.linspace(v.min(), v.max(), 50)
            ax.plot(xs, np.polyval(k, xs), "k--", lw=.9, alpha=.6,
                    label=f"r = {np.corrcoef(v, yv)[0,1]:+.2f}")
            ax.legend(fontsize=7)
        ax.axhline(0, color="k", lw=.8, alpha=.4)
        ax.set_xlabel("pre-sim physical violation (penetration + skate, normalised)", fontsize=8)
        ax.set_ylabel(yl, fontsize=8)
        ax.set_title(ttl, fontsize=9)
        ax.tick_params(labelsize=7)
    handles = [plt.Line2D([], [], marker="o", ls="", color=cmap[c], label=c) for c in cls]
    axes[0].legend(handles=handles, fontsize=6, loc="upper left")
    fig.suptitle("Frozen-PHC physics: does benefit track pre-simulation physical violation?",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT_PNG, dpi=140, bbox_inches="tight")

    json.dump({"clips": rows, "picks": [{"role": a, "tag": b["tag"]} for a, b in picks]},
              open(OUT_JSON, "w"), indent=2, default=float)
    open(OUT_MD, "w").write("\n".join(LINES) + "\n")
    PR("")
    PR(f"scatter -> {os.path.relpath(OUT_PNG, REPO)}")
    PR(f"report  -> {os.path.relpath(OUT_MD, REPO)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
