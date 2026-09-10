"""Assemble the mesh-GPen deliverable from the diagnostics + four-way result jsons."""
import json, os, sys
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LABEL = {"gt": "Ground Truth", "simxr": "SimXR",
         "finetuned_backbone": "Finetuned Backbone", "naive": "Naive PHC Baseline"}
ORDER = ["gt", "simxr", "finetuned_backbone", "naive"]
SHORT = {"SUBJECT072.gestures.pkl_427_853": "gestures_427_853 (upper-body)",
         "SUBJECT086.walking.pkl_0_412": "walking_0_412 (lower-body)",
         "SUBJECT075.ROM.pkl_3820_4243": "ROM_3820_4243 (full-body)"}
L = []
def PR(s=""): print(s); L.append(s)

diag = json.load(open(f"{REPO}/scripts/mesh_gpen_diagnostics.json"))
four = json.load(open(f"{REPO}/scripts/mesh_gpen_fourway.json"))
old = json.load(open(f"{REPO}/scripts/old_joint_gpen_crosscheck.json"))

PR("# Mesh-level ground penetration — simulator-independent")
PR("")
PR("Definition (imported unchanged from `utils/physical_metrics.compute_ground_penetration`, the")
PR("same function the AvatarPoser baseline uses):")
PR("")
PR("- **GPen Mean (mm)** — mean depth over **penetrating (vertex, frame) pairs only**")
PR("- **GPen Rate** — fraction of **frames** with at least one vertex below the floor")
PR("- **GPen Max (mm)** — worst single-vertex depth")
PR("- tolerance 1e-4 m (0.1 mm) · floor `ground_z = 0.0` · SMPL-H **neutral**, betas = 0, 6890 verts")
PR("")
PR("## Step 1+2 — diagnostic clips and the GT retargeting floor")
PR("")
PR("| Clip | IK residual (mm) | GPen Mean (mm) | GPen Rate | GPen Max (mm) |")
PR("|---|---|---|---|---|")
for c, v in sorted(diag.items(), key=lambda kv: -kv[1]["gt"]["gpen_mean_mm"]):
    r = v["gt"]
    PR(f"| {c.replace('.pkl','')} | {r['ik_all_mean_mm']:.1f} | {r['gpen_mean_mm']:.2f} | "
       f"{r['gpen_rate']:.3f} | {r['gpen_max_mm']:.2f} |")

PR("")
PR("## Step 3 — four-way comparison")
PR("")
PR("| Motion | Method | IK Residual (mm) | GPen Mean (mm) | GPen Rate | GPen Max (mm) |")
PR("|---|---|---|---|---|---|")
for c in SHORT:
    if c not in four: continue
    for k in ORDER:
        r = four[c].get(k)
        if not r or r.get("status") == "absent":
            PR(f"| {SHORT[c]} | {LABEL[k]} | — | — | — | — |"); continue
        PR(f"| {SHORT[c]} | {LABEL[k]} | {r['ik_all_mean_mm']:.1f} | {r['gpen_mean_mm']:.2f} | "
           f"{r['gpen_rate']:.3f} | {r['gpen_max_mm']:.2f} |")

PR("")
PR("## Step 6 — old simulator GPen vs new mesh GPen")
PR("")
PR("| Motion | Method | Simulator GPen mean (mm) | Mesh GPen mean (mm) |")
PR("|---|---|---|---|")
for c in SHORT:
    for k in ORDER:
        if k == "gt" or c not in four: continue
        r = four[c].get(k)
        o = old.get(c, {}).get(LABEL[k])
        if not r or not o or r.get("status") == "absent": continue
        PR(f"| {SHORT[c]} | {LABEL[k]} | {o['mean_mm']:.3f} | {r['gpen_mean_mm']:.2f} |")

# fairness: flag any method whose IK residual is an outlier
PR("")
PR("## Fairness — IK residual spread per motion")
PR("")
PR("| Motion | min | max | spread | flag |")
PR("|---|---|---|---|---|")
for c in SHORT:
    if c not in four: continue
    vals = {k: v["ik_all_mean_mm"] for k, v in four[c].items() if "ik_all_mean_mm" in v}
    if not vals: continue
    lo, hi = min(vals.values()), max(vals.values())
    worst = max(vals, key=vals.get)
    flag = ("**" + LABEL[worst] + " +%.0f%% over best — GPen may be fit-contaminated**" %
            (100 * (hi - lo) / lo)) if (hi - lo) / lo > 0.25 else "none"
    PR(f"| {SHORT[c]} | {lo:.1f} | {hi:.1f} | {hi-lo:.1f} | {flag} |")

open(f"{REPO}/scripts/mesh_gpen_report.md", "w").write("\n".join(L) + "\n")
print("\nwrote scripts/mesh_gpen_report.md")
