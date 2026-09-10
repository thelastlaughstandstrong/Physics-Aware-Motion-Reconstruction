"""Build the two motion subsets for the multi-clip action logging.

  gt_subset.pkl    : ground-truth Quest motions  -> Finetuned Backbone rollout (student states)
  pred_dir/        : symlinks to AvatarPoser-predicted motions -> Naive PHC rollout (own trajectory)

The naive side must be a DIRECTORY, not a merged pkl: motion_lib_base.load_data() applies its
min_length / im_eval branches only in file mode, and both index 'pose_quat_global' -- a field the
AvatarPoser-pred motions do not carry (they have pose_quat + root_trans_offset + fps only).

Both keep the SAME clip order so env index i refers to the same underlying clip in both runs.
The anchor clip from the single-clip diagnostic is appended last, flagged, and is the pipeline's
correctness check -- it must reproduce the published ankle numbers.
"""
import json, os, joblib

REPO = "/home/sia-irl/AvatarPoser"
D = f"{REPO}/scripts/diag_fb_multiclip"
GT_DIR = f"{REPO}/data/quest2_synthetic/test_seg_motion"
PR_DIR = f"{REPO}/data/quest2_synthetic/test_seg_avatarposer_pred"
ANCHOR = "SUBJECT086.walking.pkl_2889_3300"

sel = json.load(open(f"{D}/selected_clips.json"))
for r in sel: r["anchor"] = False
if not any(r["clip"] == ANCHOR for r in sel):
    sel.append(dict(clip=ANCHOR, activity="walking", category="lower-body",
                    selected_for="anchor", anchor=True,
                    fb_term=True, nv_term=False, fb_sim=411, fb_ref=411, nv_sim=369, nv_ref=369))

gt, pr, order, missing = {}, {}, [], []
for r in sel:
    c = r["clip"]
    fg, fp = f"{GT_DIR}/{c}.pkl", f"{PR_DIR}/{c}.pkl"
    if not (os.path.isfile(fg) and os.path.isfile(fp)):
        missing.append((c, os.path.isfile(fg), os.path.isfile(fp))); continue
    dg, dp = joblib.load(fg), joblib.load(fp)
    assert list(dg.keys()) == [c] and list(dp.keys()) == [c], (list(dg.keys()), list(dp.keys()))
    gt[c] = dg[c]; pr[c] = dp[c]; order.append(c)

if missing:
    print("MISSING (skipped):"); [print("   ", m) for m in missing]
joblib.dump(gt, f"{D}/gt_subset.pkl")
os.makedirs(f"{D}/pred_dir", exist_ok=True)
for c in order:
    lnk = f"{D}/pred_dir/{c}.pkl"
    if not os.path.islink(lnk):
        os.symlink(f"{PR_DIR}/{c}.pkl", lnk)
json.dump([r for r in sel if r["clip"] in gt], open(f"{D}/selected_clips.json", "w"), indent=1)
json.dump(order, open(f"{D}/clip_order.json", "w"), indent=1)
print(f"{len(order)} clips -> gt_subset.pkl / pred_dir/  (anchor last: {order[-1]})")
import collections
print(dict(collections.Counter(r["category"] for r in sel if r["clip"] in gt)))
