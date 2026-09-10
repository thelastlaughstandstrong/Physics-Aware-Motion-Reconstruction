"""Select a stratified multi-clip sample for the ankle-tail generalisation check.

Uses the two EXISTING full-test dumps (no new rollouts) to find clips where BOTH the Finetuned
Backbone (trackerr@380k, tracking GT) and the Naive PHC baseline (same teacher weights, tracking
AvatarPoser predictions) survive to the end, so the comparison is not contaminated by early
termination. Category scheme is copied verbatim from dataset_mesh_gpen_per_clip.csv.
"""
import csv, json, os, joblib, numpy as np

REPO = "/home/sia-irl/AvatarPoser"
FB = f"{REPO}/output/HumanoidIm/quest_distill_trackerr/phc_positions_for_eval_0380160.pkl"
NV = f"{REPO}/output/HumanoidIm/phc_prim_quest/phc_positions_for_eval.pkl"
PER_CLIP = f"{REPO}/scripts/dataset_mesh_gpen_per_clip.csv"
N_PER = {"lower-body": 15, "upper-body": 15, "full-body/dynamic": 15, "other": 10}
SEED = 0

cat, act = {}, {}
for r in csv.DictReader(open(PER_CLIP)):
    cat[r["clip"]] = r["category"]; act[r["clip"]] = r["activity"]

def load(p):
    d = joblib.load(p)
    ks = [os.path.basename(str(k))[:-4] for k in d["motion_keys"]]
    th = np.asarray(d["terminate_hist"])
    return {k: dict(term=bool(th[i]), sim=len(d["pred_pos_all"][i]), ref=len(d["ref_pos_all"][i]))
            for i, k in enumerate(ks)}

fb, nv = load(FB), load(NV)
rows = []
for k in sorted(set(fb) & set(nv) & set(cat)):
    rows.append(dict(clip=k, activity=act[k], category=cat[k],
                     fb_term=fb[k]["term"], fb_sim=fb[k]["sim"], fb_ref=fb[k]["ref"],
                     nv_term=nv[k]["term"], nv_sim=nv[k]["sim"], nv_ref=nv[k]["ref"]))
print(f"{len(rows)} clips with all three records")
for c in sorted(N_PER):
    sub = [r for r in rows if r["category"] == c]
    both = [r for r in sub if not r["fb_term"] and not r["nv_term"]]
    print(f"  {c:18s} total {len(sub):4d} | FB survives {sum(not r['fb_term'] for r in sub):4d}"
          f" | Naive survives {sum(not r['nv_term'] for r in sub):4d} | BOTH {len(both):4d}")

rng = np.random.RandomState(SEED)
sel = []
for c in sorted(N_PER):
    pool = sorted([r for r in rows if r["category"] == c and not r["fb_term"] and not r["nv_term"]],
                  key=lambda r: r["clip"])
    if not pool:
        print(f"  !! no surviving clips for {c}"); continue
    # spread across activities within the category, then fill
    by_act = {}
    for r in pool: by_act.setdefault(r["activity"], []).append(r)
    order, acts = [], sorted(by_act)
    for a in acts: rng.shuffle(by_act[a])
    i = 0
    while len(order) < min(N_PER[c], len(pool)):
        a = acts[i % len(acts)]
        if by_act[a]: order.append(by_act[a].pop())
        i += 1
        if all(not v for v in by_act.values()): break
    for r in order: r["selected_for"] = c
    sel += order

print(f"\nselected {len(sel)} clips")
for c in sorted(N_PER):
    s = [r for r in sel if r["category"] == c]
    import collections
    print(f"  {c:18s} n={len(s):3d}  {dict(collections.Counter(r['activity'] for r in s))}")
json.dump(sel, open(f"{REPO}/scripts/diag_fb_multiclip/selected_clips.json", "w"), indent=1)
print("-> scripts/diag_fb_multiclip/selected_clips.json")
