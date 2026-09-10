"""
Aggregate the full-dataset mesh-GPen results into the deliverable tables.

Clip-level metrics are computed first, then aggregated with EQUAL WEIGHT PER CLIP so long clips do
not dominate. A pooled frame-weighted figure is reported alongside as a secondary diagnostic only.
"""
import csv
import json
import os
import re
import statistics as st
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IN = os.path.join(REPO, "data/quest_smpl_retarget/dataset_gpen")
OUT_MD = os.path.join(REPO, "scripts/dataset_mesh_gpen_report.md")
OUT_JSON = os.path.join(REPO, "scripts/dataset_mesh_gpen_aggregate.json")
OUT_CSV = os.path.join(REPO, "scripts/dataset_mesh_gpen_per_clip.csv")

STREAM_LABEL = [
    ("gt",                 "Ground Truth (retarget control)", "pre"),
    ("phase_a",            "Phase A / Kinematics Only",        "pre"),
    ("fb_pre",             "Finetuned Backbone (q_base)",      "pre"),
    ("finetuned_backbone", "Finetuned Backbone",               "post"),
    ("naive",              "Naive PHC Baseline",               "post"),
    ("simxr",              "SimXR no-vision",                  "post"),
]
CATEGORY = {
    "upper-body": {"gestures", "whiteboard", "teaching", "drinks", "retail", "cooking"},
    "lower-body": {"walking", "jogging"},
    "full-body/dynamic": {"ROM", "sports", "dancing", "martialarts", "stretching", "workout",
                          "yoga", "interactions"},
}
L = []


def PR(s=""):
    print(s); L.append(s)


def activity(clip):
    m = re.match(r"SUBJECT\d+\.([A-Za-z]+)\.", clip)
    return m.group(1) if m else "?"


def category(a):
    for c, s in CATEGORY.items():
        if a in s:
            return c
    return "other"


def agg(vals):
    if not vals:
        return None
    q1, q3 = np.percentile(vals, [25, 75])
    return dict(mean=float(np.mean(vals)), median=float(np.median(vals)),
                sd=float(np.std(vals)), iqr=float(q3 - q1), n=len(vals))


def main():
    data = {}
    for key, _, _ in STREAM_LABEL:
        p = os.path.join(IN, f"{key}.json")
        if os.path.isfile(p):
            data[key] = json.load(open(p))
    if not data:
        print("no stream results yet"); return 2

    PR("# Full-dataset mesh-level Ground Penetration — Quest2 test set")
    PR("")
    PR("Metric: `utils.physical_metrics.compute_ground_penetration`, unchanged — GPen Mean is the")
    PR("mean depth over **penetrating (vertex, frame) pairs only**; GPen Rate is the fraction of")
    PR("**frames** with at least one vertex below the floor; tolerance 1e-4 m; floor Z = 0;")
    PR("SMPL-H neutral, betas = 0, 6890 vertices. Every stream goes through the identical")
    PR("Quest→SMPL-H position IK, the same vertical calibration, and no per-method alignment.")
    PR("")
    PR("Aggregation is **equal-weight per clip**. Pooled frame-weighted values appear only as a")
    PR("secondary diagnostic.")
    PR("")
    PR("## Main aggregate")
    PR("")
    PR("| Method | Stage | Clips | GPen Mean (mm) | Median | GPen Rate | % clips penetrating | IK residual (mm) |")
    PR("|---|---|---|---|---|---|---|---|")
    summary = {}
    for key, label, stage in STREAM_LABEL:
        if key not in data:
            PR(f"| {label} | {stage} | — | pending | — | — | — | — |")
            continue
        d = data[key]
        gm = [v["gpen_mean_mm"] for v in d.values()]
        gr = [v["gpen_rate"] for v in d.values()]
        ik = [v["ik_all_mean_mm"] for v in d.values()]
        pct = 100.0 * sum(1 for v in d.values() if v["gpen_rate"] > 0) / len(d)
        a_gm, a_gr, a_ik = agg(gm), agg(gr), agg(ik)
        summary[key] = dict(label=label, stage=stage, clips=len(d), gpen_mean=a_gm,
                            gpen_rate=a_gr, ik=a_ik, pct_penetrating=pct,
                            gpen_max_mean=float(np.mean([v["gpen_max_mm"] for v in d.values()])))
        PR(f"| {label} | {stage} | {len(d)} | **{a_gm['mean']:.2f}** | {a_gm['median']:.2f} | "
           f"**{a_gr['mean']:.3f}** | {pct:.1f}% | {a_ik['mean']:.1f} ± {a_ik['sd']:.1f} |")
    PR("")
    PR("Spread (equal-weight per clip):")
    PR("")
    PR("| Method | GPen Mean sd | GPen Mean IQR | GPen Rate median | mean GPen Max (mm) |")
    PR("|---|---|---|---|---|")
    for key, label, _ in STREAM_LABEL:
        if key not in summary: continue
        s = summary[key]
        PR(f"| {label} | {s['gpen_mean']['sd']:.2f} | {s['gpen_mean']['iqr']:.2f} | "
           f"{s['gpen_rate']['median']:.3f} | {s['gpen_max_mean']:.2f} |")

    # per-category
    PR("")
    PR("## By motion category (equal-weight per clip)")
    PR("")
    cats = ["upper-body", "lower-body", "full-body/dynamic"]
    PR("| Method | " + " | ".join(f"{c} GPen mean / rate" for c in cats) + " |")
    PR("|---|" + "---|" * len(cats))
    percat = {}
    for key, label, _ in STREAM_LABEL:
        if key not in data: continue
        row, percat[key] = [], {}
        for c in cats:
            v = [x for k, x in data[key].items() if category(activity(k)) == c]
            if v:
                m = float(np.mean([y["gpen_mean_mm"] for y in v]))
                r = float(np.mean([y["gpen_rate"] for y in v]))
                percat[key][c] = dict(gpen_mean=m, gpen_rate=r, n=len(v))
                row.append(f"{m:.2f} / {r:.3f}")
            else:
                row.append("—")
        PR(f"| {label} | " + " | ".join(row) + " |")

    # per-activity for the biggest activities
    PR("")
    PR("## By activity (largest categories)")
    PR("")
    acts = sorted({activity(k) for k in data[list(data)[0]]},
                  key=lambda a: -sum(1 for k in data[list(data)[0]] if activity(k) == a))[:8]
    PR("| Method | " + " | ".join(acts) + " |")
    PR("|---|" + "---|" * len(acts))
    for key, label, _ in STREAM_LABEL:
        if key not in data: continue
        row = []
        for a in acts:
            v = [x for k, x in data[key].items() if activity(k) == a]
            row.append(f"{np.mean([y['gpen_mean_mm'] for y in v]):.1f}" if v else "—")
        PR(f"| {label} | " + " | ".join(row) + " |")

    # pre vs post
    PR("")
    PR("## Pre- vs post-physics (same mesh metric, stages labelled)")
    PR("")
    PR("| Comparison | pre-sim GPen mean | post-sim GPen mean | change |")
    PR("|---|---|---|---|")
    for pre, post, name in [("fb_pre", "finetuned_backbone", "Finetuned Backbone"),
                            ("phase_a", "naive", "Naive PHC (Phase A → frozen teacher)")]:
        if pre in summary and post in summary:
            a, b = summary[pre]["gpen_mean"]["mean"], summary[post]["gpen_mean"]["mean"]
            PR(f"| {name} | {a:.2f} | {b:.2f} | {b-a:+.2f} mm "
               f"({'reduced' if b < a else 'increased'}) |")
    PR("| SimXR no-vision | N/A (no intermediate pose by design) | "
       + (f"{summary['simxr']['gpen_mean']['mean']:.2f}" if "simxr" in summary else "—") + " | — |")

    # per-clip CSV
    with open(OUT_CSV, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["clip", "activity", "category", "stream", "stage", "frames",
                    "ik_all_mean_mm", "gpen_mean_mm", "gpen_rate", "gpen_max_mm"])
        for key, label, stage in STREAM_LABEL:
            for c, v in data.get(key, {}).items():
                w.writerow([c, activity(c), category(activity(c)), label, stage, v["frames"],
                            f"{v['ik_all_mean_mm']:.3f}", f"{v['gpen_mean_mm']:.4f}",
                            f"{v['gpen_rate']:.4f}", f"{v['gpen_max_mm']:.4f}"])
    json.dump({"summary": summary, "per_category": percat}, open(OUT_JSON, "w"),
              indent=2, default=float)
    open(OUT_MD, "w").write("\n".join(L) + "\n")
    PR("")
    PR(f"per-clip CSV -> {os.path.relpath(OUT_CSV, REPO)}")
    PR(f"aggregate    -> {os.path.relpath(OUT_JSON, REPO)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
