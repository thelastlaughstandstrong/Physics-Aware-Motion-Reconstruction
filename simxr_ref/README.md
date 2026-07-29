# simxr_ref

Working copies of two files from the separate/upstream SimXR repo (`/home/sia-irl/SimXR/`,
local checkout, commit `39081b831aa1c1e52a4a974fb78a1ed93f7580c2`, no remote configured),
copied here so they can be patched without touching that repo's originals.

Purpose: enable option (b) from `results_comparison.md` footnote 7 / the
`simxr_ablation_replication` memory — get the full `MPJRE`/`Jitter`/`Foot Skate`/`gpen` metric
set for the SimXR no-vision ablation checkpoint (`ablation_results.md`), by porting this repo's
own `phc_positions_for_eval.pkl`-dump instrumentation (already working in
`phc/phc/learning/im_amp_players.py`, this repo's own PHC fork) into the equivalent files in
the SimXR repo, rather than trying to load that checkpoint into this repo's incompatible env
wiring.

Files, relative path preserved from the SimXR repo's own layout:
- `phc/env/tasks/humanoid_im.py` — needs `body_rot`/`body_rot_gt` added to `self.extras` in
  `post_physics_step` (SimXR's copy currently only exposes `body_pos`/`body_pos_gt`, no
  rotation — required for MPJRE).
- `phc/learning/im_amp_players.py` — needs the same `pred_rot`/`gt_rot` accumulation and final
  `phc_positions_for_eval.pkl` dump this repo's fork already has (diffed directly against
  `phc/phc/learning/im_amp_players.py` here — otherwise identical). Also has a live, unconditional
  `import ipdb; ipdb.set_trace()` mid-eval-loop that needs removing so a batch/automated eval run
  doesn't hang.

**To apply**: once patched here, copy these two files back over their originals at
`/home/sia-irl/SimXR/phc/env/tasks/humanoid_im.py` and
`/home/sia-irl/SimXR/phc/learning/im_amp_players.py`, then re-run the ablation checkpoint's
`test=True im_eval=True` eval pass there to produce `phc_positions_for_eval.pkl`, and run this
repo's `scripts/evaluate_phc_metrics_quest.py --pos_dump <that file>` on it.

Not wired into this repo's own build/imports — these are reference/patch-staging copies only,
not used by anything in `phc/` (this repo's actual PHC fork) or any training/eval script here.

## Status: patched, copied back, and run successfully (2026-07-27)

Both files above had the patch applied (diffed and syntax-checked against the SimXR originals
first). Per user request ("you do that"), copied over the SimXR checkout's originals at
`/home/sia-irl/SimXR/phc/env/tasks/humanoid_im.py` and
`/home/sia-irl/SimXR/phc/learning/im_amp_players.py`, then re-ran the peak (epoch 1,500)
checkpoint of `simxr_quest_novision_lrdecay` through `test=True im_eval=True`
(`env.num_envs=64`, reduced from the original 512 to avoid crashing on GPU memory contention
with an unrelated concurrently-running training job, `quest_distill_explicit_pose`) to produce
`phc_positions_for_eval.pkl`, then ran `scripts/evaluate_phc_metrics_quest.py` on it in this
repo. Result now recorded in `results_comparison.md`'s Quest2 table (footnote 7): MPJRE 29.76°,
MPJVE 40.93 cm/s, Jitter 704.29 m/s³, Foot Skate 14.39 cm/s, gpen_mean/rate 0.00/0.000 — success
rate (0.7155) and MPJPE (112.36mm) reproduced the original console numbers (0.717/112.5mm)
within noise, confirming the num_envs=64 substitution is valid.

Real gotcha hit and fixed along the way: the SimXR repo's `im_amp_players.py` had a live
`import ipdb; ipdb.set_trace()` that had been silently blocking this exact dump step in every
prior manual eval run (confirmed via `eval_peak_1500.log` — metrics printed fine, then hung at
an interactive debugger until manually killed). Removed in the patched copy (this directory)
before copying back.

These local copies (`simxr_ref/`) are left in place as the patch record / for future re-runs
(e.g. the epoch-6,105 final checkpoint, not yet done) — they are not wired into anything and can
be regenerated from the SimXR checkout's current (now-patched) files if needed.
