"""Build a held-out VALIDATION subset for the foot-weight ablation.

Two design constraints drive this:

  1. Selection and reporting must not share clips. Every previous Quest2 checkpoint was chosen by
     scanning success rate on the FULL test set (checkpoint_scan.log), i.e. model selection on the
     reporting set. This splits the 1072 test clips into a validation slice used to pick the winning
     arm and a disjoint remainder reserved for the final number.

  2. The subset must contain the cases the intervention targets (clips where FB's feet are worst)
     AND clean controls (clips where FB's feet are already fine), so a regression on easy clips is
     visible rather than averaged away.

Strata: 4 categories x {high-foot-error, clean-control} + a random remainder, drawn only from clips
where the current FB checkpoint survives, so foot error is well defined.
"""
import csv, json, os
import joblib, numpy as np

REPO = "/home/sia-irl/AvatarPoser"
OUT = f"{REPO}/scripts/ablation_footweight"
FB_DUMP = f"{REPO}/output/HumanoidIm/quest_distill_trackerr/phc_positions_for_eval_0380160.pkl"
PER_CLIP = f"{REPO}/scripts/dataset_mesh_gpen_per_clip.csv"
FOOT = [21, 24]
N_HIGH, N_CLEAN, N_RAND = 12, 8, 10      # per category / total for the random tail
SEED = 0

cat, act = {}, {}
for r in csv.DictReader(open(PER_CLIP)):
    cat[r["clip"]] = r["category"]; act[r["clip"]] = r["activity"]

d = joblib.load(FB_DUMP)
term = np.asarray(d["terminate_hist"])
rows = []
for i, k in enumerate(d["motion_keys"]):
    c = os.path.basename(str(k))[:-4]
    if c not in cat or bool(term[i]):
        continue
    P = np.asarray(d["pred_pos_all"][i]); R = np.asarray(d["ref_pos_all"][i])
    n = min(len(P), len(R)); P, R = P[:n], R[:n]
    fe = float(np.linalg.norm(P[:, FOOT] - R[:, FOOT], axis=-1).mean() * 100)
    rows.append(dict(clip=c, category=cat[c], activity=act[c], fb_foot_err_cm=fe))
print(f"{len(rows)} surviving clips scored for FB foot error")

rng = np.random.RandomState(SEED)
sel, taken = [], set()
for c in sorted({r["category"] for r in rows}):
    sub = sorted([r for r in rows if r["category"] == c], key=lambda r: -r["fb_foot_err_cm"])
    for r in sub[:N_HIGH]:
        r["stratum"] = "high-foot-error"; sel.append(r); taken.add(r["clip"])
    for r in sub[-N_CLEAN:]:
        if r["clip"] in taken: continue
        r["stratum"] = "clean-control"; sel.append(r); taken.add(r["clip"])
rest = [r for r in rows if r["clip"] not in taken]
for r in rng.permutation(len(rest))[:N_RAND]:
    q = rest[int(r)]; q["stratum"] = "random"; sel.append(q); taken.add(q["clip"])

held_out = [r["clip"] for r in rows if r["clip"] not in taken]
json.dump(sel, open(f"{OUT}/val_subset.json", "w"), indent=1)
json.dump(sorted(taken), open(f"{OUT}/val_clips.json", "w"), indent=1)

print(f"\nvalidation subset: {len(sel)} clips   (reporting remainder: {len(held_out)} clips)")
print(f"{'category':20s} {'high':>5s} {'clean':>6s} {'rand':>5s}   foot-err cm (high / clean)")
for c in sorted({r["category"] for r in sel}):
    s = [r for r in sel if r["category"] == c]
    hi = [r["fb_foot_err_cm"] for r in s if r["stratum"] == "high-foot-error"]
    cl = [r["fb_foot_err_cm"] for r in s if r["stratum"] == "clean-control"]
    print(f"{c:20s} {len(hi):5d} {len(cl):6d} {sum(1 for r in s if r['stratum']=='random'):5d}   "
          f"{np.mean(hi) if hi else float('nan'):.1f} / {np.mean(cl) if cl else float('nan'):.1f}")
allv = [r["fb_foot_err_cm"] for r in sel]
print(f"\nsubset FB foot error: mean {np.mean(allv):.2f} cm, range {min(allv):.2f}-{max(allv):.2f}")
print(f"full-set FB foot error: mean {np.mean([r['fb_foot_err_cm'] for r in rows]):.2f} cm")
