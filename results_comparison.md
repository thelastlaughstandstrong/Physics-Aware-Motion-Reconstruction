# Results Comparison

## PHC Physics Evaluation — Primary Results

Full student (backbone + action head) in IsaacGym physics simulation on PHC test set (536 sequences).
Results saved to `phc/output/HumanoidIm/<EXP>/phc_eval.log`.

| Model | Checkpoint | Success | MPJPE (cm) | qref_MPJPE (cm) | MPJVE (cm/s) | MPJRE (°) | Jitter (m/s³) | Foot Skate (cm/s) | gpen_mean (mm) | gpen_rate |
|---|---|---|---|---|---|---|---|---|---|---|
| `frozen_bb` — frozen backbone | 489.5k | 0.584 | 34.83 | — | 89.39 | 38.73 | 1188 | 2.66 | — | — |
| `v2`–`v6` — pre-v7 refinement-branch line (7 rows, collapsed)† | 431k–1.95M | 0.440–0.498 | 41.14–50.71 | 21.57–61.40 | 99.59–113.59 | 40.86–45.02 | 1345–1612 | 2.28–2.81 | 0.04–0.74 | 0.003–0.167 |
| **`v7` — action distillation (Stage E), post HMD-pipeline fix**¹ | 250k | 0.545 | 38.18 | 10.42 | 96.69 | 40.31 | 1337 | 2.41 | 2.46 | 0.745 |
| `v7_stager` — Stage R (reward PPO), 10k steps, coarse manual milestone | 260k | 0.644 | 30.84 | 10.72 | 82.88 | 37.35 | 1028 | 2.13 | 1.24 | 0.388 |
| `v7_stager` — Stage R, 30k steps (regressing) | 280k | 0.506 | 39.71 | 10.50 | 97.40 | 44.75 | 1275 | 1.64 | 1.30 | 0.375 |
| `v7_stager` — Stage R, 50k steps (COLLAPSED — do not use) | 300k | 0.062 | 53.95 | 10.12 | 161.63 | 59.68 | 2433 | 1.16 | 1.04 | 0.333 |
| **`v7_validate` — Stage R, 6k steps, found via collapse-detection script — BEST distilled result⁴** | 256k | **0.756** | **24.59** | 10.76 | 68.99 | 34.83 | 773 | 1.66 | 1.46 | 0.388 |
| `v8` — Stage E, explicit `q_base` input to action/value heads (negative result⁵) | 250k | 0.521 | 40.79 | 10.69 | 100.09 | 40.93 | 1370 | 2.36 | 1.23 | 0.382 |
| `v8_stager` — Stage R, 10k steps (v8's own peak, still worse than v7_validate) | 260k | 0.634 | 33.61 | 10.81 | 88.51 | 39.67 | 1210 | 1.91 | 1.31 | 0.380 |
| `goal-delta-input` — Stage E, full obs (self+task-obs) instead of self-obs only (negative result⁶) | 250k | 0.539 | 38.85 | 10.67 | 94.02 | 40.45 | 1235 | 2.41 | 1.23 | 0.378 |
| `goal-delta-input` — Stage R, 10k steps (peak, collapses fast — do not use) | 260k | 0.332 | 51.38 | 10.30 | 126.06 | 49.87 | 1665 | 1.85 | 1.05 | 0.355 |
| **Naive baseline — frozen PHC teacher tracks raw AvatarPoser output directly, no distillation²** | phc_kp_mcp_iccv | **0.912** | **14.06** | — | 43.39 | 18.93 | 318 | 0.88 | (23.79)³ | (0.654)³ |

> `qref_MPJPE` = kinematic pose accuracy before physics sim; `MPJPE` = physics sim trajectory accuracy. They measure different outputs.
> `gpen_mean` = mean penetration depth (mm) among penetrating joints, and `gpen_rate` = fraction of joints
> penetrating the ground plane — both **pre-sim** (measured on q_ref/q_base), for all rows above the
> naive-baseline row. Rows with no q_ref/backbone-reconstruction step (`frozen_bb`, `v2`, `v3`) have no
> pre-sim reference to measure against, hence `—`.
> AvatarPoser kinematic baseline: MPJPE 2.51 cm, Jitter 1357 m/s³, Foot Skate 33.67 cm/s, gpen_mean 23.79 mm, gpen_rate 0.654.
> † **Pre-v7 rows collapsed 2026-08-13.** All seven were superseded by `v7` and every one of their
> `qref_MPJPE` values was computed on the broken HMD input pipeline (footnote ¹), so none is
> comparable to anything below. What they established, in one line: v4 was the best of the line
> (peaking at 750k); v5's position-space `L_ref` improved qref_MPJPE 61→22cm but degraded physics
> and spiked gpen to 16.7% (L_ref conflicted with encoder training); v6's full gradient isolation
> protected physics but made the refinement head produce near-zero corrections. Full per-version
> numbers are in git history if ever needed.
>
> ¹ `v7`/`v7_stager` supersede all rows above them — six distinct bugs in the pre-v7 HMD
> feature construction were found and fixed (joint indices, rotation frame, feature layout,
> 6D rotation convention, velocity frame-rate, and position reconstruction via FK), taking
> `qref_MPJPE` from ~56-61cm down to ~10.3-10.7cm. Full writeup in dev memory
> (`distill_hmd_joint_bug.md`). Do not compare pre-v7 rows' `qref_MPJPE` against v7's —
> they were computed on a broken input pipeline.
>
> ² Naive baseline evaluated on the full 536-sequence test set 2026-07-07 via
> `evaluate_phc_metrics.py`'s own documented recipe (`convert_avatarposer_to_phc.py` on all
> 536 test sequences → `learning=im_mcp` frozen-teacher eval → metrics). **This beats every
> distillation/Stage R result above on every metric** (2x better MPJPE, 41% better success
> rate, 3x better jitter). It's not that the distillation pipeline is broken — it solves a
> strictly harder problem (compressing full-body tracking into a network that only ever sees
> a sparse 3-point HMD signal, vs. handing PHC's teacher its full intended per-frame kinematic
> input). Don't present the distillation work as an accuracy win over this baseline without
> closing this gap first, or reframe it around a latency/compute tradeoff instead.
>
> > **⚠ 2026-08-13: "it's not that the distillation pipeline is broken" is now an OPEN QUESTION,
> > not an established fact.** On Quest2 the pipeline *was* broken, in a way that produced this
> > exact signature — naive baseline far ahead of every distilled variant — because the student's
> > observations could not express its own tracking error (see the ⚠ banner in the Quest2 section).
> > Fixing that took Quest2 from 0.317 to 0.738 success and closed almost the whole gap to its
> > naive baseline (0.767). **The AMASS/SMPL pipeline here has NOT been checked for the same
> > defect**, and it is a different code path (`DistillAgent`, frozen backbone, SMPL rig), so the
> > Quest2 result does not transfer automatically. But `DistillAgent` feeds `action_head` the same
> > `[z, state_enc(self_obs)]` with the task-obs slice dropped, so the defect is *structurally
> > present* here too. Note footnote ⁶'s `goal-delta-input` experiment fed the delta and did not
> > help — that is the one piece of counter-evidence, and it deserves a re-read now that the
> > Quest2 mechanism is understood, since it conditioned on ground-truth task_obs rather than the
> > deployable HMD-vs-current delta. **Until someone runs it, treat this footnote's "harder
> > problem" framing as a hypothesis, not a conclusion.**
>
> ³ Naive baseline has no q_ref/backbone reconstruction step at all — it tracks AvatarPoser's
> raw predicted output directly. Its pre-sim reference gpen is therefore identical to the
> `AvatarPoser (original kinematic)` baseline (23.79mm / 0.654), not a new number. Separately,
> the naive run's **post-sim** gpen (physics-corrected, computed directly on PHC's simulated
> output since no `qref_gpen.log` exists for a non-distillation run) is 0.00mm / 0.000 —
> physics simulation eliminates penetration entirely for this reference. Don't read that 0.000
> as a "pre-sim" number the way the rest of this column is defined; it's measuring something
> different (see `evaluate_phc_metrics.py`'s fallback path when no `qref_gpen.log` is found).
>
> ⁴ Found by re-running Stage R from the same `v7` Stage E checkpoint (250k) through
> `scripts/train_stage_r_with_early_stop.sh`'s collapse-detection sweep (3000-step eval
> interval, patience=2) instead of the earlier coarse manual 10k/30k/50k-step checks that
> found `v7_stager`. Confirmed deterministic/reproducible (fixed seed) across 3 independent
> reruns, not a lucky draw. **This is the current best distilled checkpoint overall** — still
> well behind the naive baseline on every metric. `v7_stager`'s rows above remain as a record
> of Stage R's collapse-after-peak behavior, not as the best result anymore.
>
> ⁵ Hypothesis: giving `action_head`/`value_head` explicit access to `q_base` (rather than only
> implicitly through `z`) would close some of the gap to the naive baseline. Result: v8 is
> *worse* than v7 at every comparable checkpoint. Likely explanation: `q_base` is a
> deterministic function of `z`, which `action_head` already receives — concatenating it
> explicitly adds no new information, only redundancy. See `distill_hmd_joint_bug.md` memory.
>
> ⁶ Hypothesis: `action_head` only ever saw the env's self-obs slice (proprioception),
> discarding the task-obs slice — the same reference-vs-current delta feature the frozen
> teacher itself consumes to produce the DAgger regression target — so `action_head` was
> forced to reconstruct that signal blind from a compressed HMD embedding. Result: Stage E is
> statistically unchanged from v7 (0.539 vs 0.545 success) — the explicit signal didn't help
> behavior cloning. Stage R is *far* worse, collapsing from the first eval checkpoint instead
> of improving first like v7 did — explicit ground-truth delta conditioning appears to make
> the policy more brittle under PPO's exploration noise, not more stable. See
> `baseline_results.md` memory for full detail.

---

## Backbone Kinematic Quality (main_test, backbone only)

Measures backbone pose predictions against AMASS GT — independent of the physics sim.

| Model | Backbone MPJPE (cm) | Notes |
|---|---|---|
| AvatarPoser (pretrained) | 2.51 | baseline |
| frozen_bb | ~2.51 | backbone never modified |
| v7 / v7_stager (frozen backbone, fixed HMD pipeline) | ~10.3–10.7 | backbone never modified; gap vs. 2.51 baseline believed to be the reconstructed-input approximation ceiling (FK position reconstruction has a validated ~2-4cm/joint residual) — not chased further as of 2026-07-07 |

---

## Eval commands

```bash
# Automated (watches for checkpoint, runs both steps):
bash scripts/auto_eval.sh <exp_name> <step>

# Manual 

#Step 1: run student in sim (rlgpu env, from phc/)
/home/sia-irl/anaconda3/bin/conda run -n rlgpu --no-capture-output bash -c "
unset PYTHONPATH && export WANDB_MODE=offline
cd /home/sia-irl/AvatarPoser/phc && python phc/run_hydra.py \
  learning=distill exp_name=<EXP> test=True im_eval=True \
  env=env_im_getup_mcp robot=smpl_humanoid robot.freeze_hand=True robot.box_body=False \
  env.motion_file=sample_data/phc_test.pkl \
  'env.models=[output/HumanoidIm/phc_kp_pnn_iccv/Humanoid.pth]' \
  env.num_envs=16 env.obs_v=7 headless=True \
  learning.params.config.student_checkpoint=output/HumanoidIm/<EXP>/student/<CKPT>.pth"

# Step 2: compute metrics (avatarposer env, from repo root)
python scripts/evaluate_phc_metrics.py \
  --pos_dump phc/output/HumanoidIm/<EXP>/phc_positions_for_eval.pkl \
  --motion_file phc/sample_data/phc_test.pkl
```

---

## Quest2 PHC Physics Evaluation

> ## ⚠ 2026-08-11: root cause of the 0%-locomotion result found, and it is architectural
>
> Every `quest_distill` number below was produced by a student that **structurally cannot observe
> its own tracking error**, which is why its per-activity breakdown is 0.000 on all 461
> locomotion/dynamic clips. Two independent fixes landed; the reruns are in progress. Read this
> before quoting any distilled row.
>
> **1. The fatal one — the student had no closed-loop feedback.** `action_head`'s two inputs are
> `z` (the AvatarPoser latent, built from HMD features in **absolute world coordinates**,
> describing only the reference) and `state_enc(proprio)`, where `proprio =
> obs[:, :env._num_self_obs]` deliberately drops PHC's task-obs slice and
> `compute_humanoid_observations_quest_max` keeps only root **height** — world XY and heading are
> stripped out by construction. Neither input carries the humanoid's own world pose, so
> "how far am I from the reference and in which direction" is **not a function of the student's
> observations at all**. The frozen teacher it distills from *does* get exactly that signal (PHC's
> task_obs; cf. SimXR's `compute_imitation_observations`, `ref_body_pos - body_pos` rotated into
> the heading frame). An open-loop pose predictor keeps stationary error bounded but cannot servo
> to a moving reference — hence 0.60–0.80 on stationary activities and exactly 0.000 on
> locomotion. Fixed by `use_tracking_error` (default **True**, `learning/distill_quest.yaml`): a
> 27D reference-minus-simulated delta at the 3 HMD-tracked bodies, in the humanoid's heading
> frame, appended to `action_head`/`value_head`'s input. This is **deployable, not an oracle** — a
> real headset knows where the user's head and wrists are, and the simulator knows where its
> avatar's are; it never sees the reference's full-body pose. Unit-tested for zero-error,
> heading-invariance, world-translation-invariance and PHC's per-env `_global_offset` (which
> `resample_motions()` makes nonzero during training).
>
> **2. A real but minor one — velocity features at the wrong frame rate.** `_get_cur_prev_ref_states`
> queried the motion lib at `t - 1/60` for *both* pipelines, but the quest backbone is trained by
> `data/dataset_quest2.py` on **consecutive frames of 30fps pkls**, so 27 of its 54 input channels
> arrived at half the magnitude it was trained on. Fixed via `_HMD_VEL_DT_QUEST = 1/30` (the AMASS
> path keeps 1/60 by default and is unchanged). Measured cost before the fix: **+0.27° / +4.4mm**
> on locomotion, ~0 on stationary — real, worth fixing, but not the cause of the 0%.
>
> **3. The missing control, now run — and it is the best Quest2 result to date.** There was never
> a Quest2 counterpart to the SMPL section's naive baseline. Running it (frozen `phc_prim_quest`
> teacher tracking AvatarPoser's own predicted motion, **no distillation student at all**) gives
> **0.767 success / 96.4mm MPJPE / 13.82° MPJRE** against ground truth — including **walking 0.792
> and jogging 0.779**, versus 0.000/0.000 for every distilled checkpoint. It also beats the SimXR
> no-vision ablation (0.717 / 112.5mm / 29.76°), and does so on the *held-out test split* where
> SimXR's own per-activity dump is a train-split eval. See the naive-baseline subsection below.
>
> Taken together: the backbone's kinematic output is good enough for the physics teacher to
> locomote at ~79%, and the teacher tracks ground truth at 99.9% — so the 0% is neither the data,
> nor the backbone, nor the fps bug. It is the student's observation space.
>
> ### FINAL result of the fixed rerun (`quest_distill_trackerr`, Stage E, 500k steps, complete)
>
> **Best checkpoint `student_0380160.pth` (full-range scan, patience disabled): 0.7379 success.**
> Same backbone, same teacher, same recipe as `finetuned_backbone`; `use_tracking_error` is the
> only real variable (plus the 1/30 fps fix). Stage E only — `enable_stage_r: false`, verified
> 15,625 Stage E batches and 0 Stage R batches in the training log.
>
> | Metric | `finetuned_backbone` @440,160 | **`trackerr` @380,160** | Naive baseline | SimXR no-vision |
> |---|---|---|---|---|
> | Success (`phc_survival`) | 0.317 | **0.738** | 0.767 | 0.717 |
> | Success (`frame_head`) | 0.980 | **0.989** | 0.974 | 0.997 |
> | MPJPE (mm) | 185.60 | **87.43** | 96.38 | 112.5 |
> | MPJRE (°) | 25.65 | **15.23** | 13.82 | 29.76 |
> | MPJVE (cm/s) | 49.94 | **34.25** | 24.17 | 40.93 |
> | Jitter (m/s³) | 1117.46 | **524.78** | 221.94 | 704.29 |
> | Foot skate (cm/s) | 20.75 | **17.14** | 21.41 | 14.39 |
> | gpen_mean (mm) | 0.003 | 0.008 | — | 0.00 |
>
> **Every metric improves over the old run, most by ~2x** (success 2.3x, MPJPE 2.1x, jitter 2.1x).
> It also **beats the SimXR no-vision ablation on every metric except foot skate** — and SimXR is
> the like-for-like comparison (same frozen `phc_prim_quest` teacher, same `only_kin_loss`
> behavior cloning; see the correction further down). Against the no-student naive baseline it is
> *better* on MPJPE (87.4 vs 96.4mm) while still slightly behind on success (0.738 vs 0.767), so
> distillation is now roughly at parity with skipping distillation entirely — the first time that
> has been true in this project.
>
> Per-activity, best checkpoint vs the old run:
>
> | activity | old @440k | **new @380k** | | activity | old @440k | **new @380k** |
> |---|---|---|---|---|---|---|
> | walking | 0.000 | **0.812** | | standing | 0.689 | **0.985** |
> | jogging | 0.000 | **0.390** | | whiteboard | 0.647 | **0.968** |
> | interactions | 0.000 | **0.573** | | gestures | 0.314 | **0.893** |
> | dancing | 0.000 | **0.250** | | ROM | 0.600 | **0.800** |
>
> **Locomotion/dynamic aggregate: 236/461 = 0.512, from 0/461 = 0.000.** martialarts (0/9) and
> sports (5/45) remain near-zero and are the obvious next target.
>
> Training-dynamics notes for future runs: the run learns fast and then **plateaus at ~130k
> steps**, oscillating in a 0.58–0.74 band for the remaining 370k — the extra budget bought almost
> nothing, and the best checkpoint (380k) beat the final one (500k, 0.707). Adjacent checkpoints
> differ by up to ±0.10 success (0.700 @250k vs 0.598 @260k), which is *larger than the entire
> 2.3pp `finetuned_backbone`-vs-`Option2` gap* this project previously treated as a result.
> Single-checkpoint comparisons at that scale are not meaningful here.
>
> ### Stage R on top of the fixed Stage E: CATASTROPHIC, and now root-caused (2026-08-13)
>
> Seeded from the 0.73787 Stage E best (`student_0380160.pth`), 30k Stage R steps, trial 2's
> conservative hyperparameters (`stage_r_lr=5e-6`, `kl_threshold=0.03`, `entropy_coef=0.003`,
> `lambda_kinematic_stage_r=0.3`). It did not trade rotation for survival this time — **it simply
> destroyed the policy**, monotonically, from the first checkpoint:
>
> | step | +480 | +2k | +4k | +8k | +12k | +30k |
> |---|---|---|---|---|---|---|
> | success | 0.394 | 0.190 | 0.040 | 0.009 | 0.001 | 0.006 |
>
> Seed 0.738 → **0.394 after 480 steps** → ~0.005 and flat thereafter. The scan's "best" (0.39366)
> is just the least-damaged checkpoint; nothing ever exceeded the seed.
>
> **Root cause: PPO starts with an untrained critic.** `_train_on_batch` (Stage E) only ever calls
> `action_head` — `value_head` is never trained, so it enters Stage R at random initialization, and
> `_train_stage_r_on_batch` optimizes actor and critic *jointly from batch 1* with no warm-up.
> Observed critic loss starts at **~79** and only reaches ~6 by batch ~287. Until then GAE
> advantages are computed from a garbage value function, so the sign of the advantage is close to
> noise and the first PPO updates push the actor in essentially random directions — by the time the
> critic is usable, the actor is already wrecked.
>
> #### Critic warm-up experiment (`quest_stager_warmup`, 2026-08-13) — OUTCOME B
>
> Single-variable test of the above: identical Stage R settings (`stage_r_lr=5e-6`,
> `kl_threshold=0.03`, `entropy_coef=0.003`, `action_log_std_init=-3.6`,
> `lambda_kinematic_stage_r=0.3`), same seed, with `stage_r_critic_warmup=True` added. Warm-up
> trains `value_head` alone against **lambda=1 Monte-Carlo returns** (at lambda=1 the
> `_stage_r_gae` recursion telescopes, so the critic enters only via the tail bootstrap; the
> default 0.95 makes every target depend on the random critic), with backbone/state_enc/
> action_head/action_log_std frozen and asserted **bit-identical** afterwards. Ran 47 batches to
> an EV plateau (best **EV 0.809**, calibration `mean_pred 58.6` vs `mean_target 58.7`, from
> `2.9 vs 9.8` at random init), then 4,416 PPO steps.
>
> **Invariant (required): post-warm-up, pre-PPO eval reproduced the seed EXACTLY** — 0.73787
> success / 87.43mm / 15.23°, bit-for-bit. The actor was genuinely untouched.
>
> | steps after warm-up | success | MPJPE | MPJRE | un-warmed run |
> |---|---|---|---|---|
> | 0 (pre-PPO) | **0.7379** | 87.43 | 15.23 | 0.7379 (seed) |
> | +480 | **0.6185** | 101.38 | 18.83 | **0.394** |
> | +960 | 0.3601 | 129.58 | 23.37 | — |
> | +1440 | 0.1810 | 143.99 | 28.19 | — |
> | +2400 | 0.0858 | 147.85 | 31.17 | 0.190 (@+2000) |
> | +3360 | 0.0065 | 203.34 | 37.78 | — |
> | +4416 | 0.0215 | 210.28 | 45.49 | 0.040 (@+4000) |
>
> **The immediate cliff is roughly two-thirds smaller** (first-checkpoint drop -0.119 vs -0.344),
> so the random critic was a real contributor. **But the policy still collapses**, only more
> slowly. Warm-up alone is not sufficient.
>
> Two concrete suspects, both visible in the PPO diagnostics and neither changed in this run:
> - **The KL-adaptive controller undid the conservative LR.** Measured KL early was 0.006-0.015,
>   below `kl_threshold*0.5`, so `_adapt_lr` *raised* `stage_r_lr` 5e-6 -> 7.5e-6 -> 1.13e-5 ->
>   **1.69e-5** — it accelerates precisely because the policy is (correctly) barely moving. KL
>   then ran away late: 0.0034 -> 0.0419 -> **0.2539**, ~8x the threshold.
> - **Critic gradient floods the SHARED trunk.** `_action_feat` detaches `z` but **not**
>   `h = state_enc(proprio)`, so critic loss backprops into `state_enc` with `critic_coef=5`.
>   Critic loss fell 49 -> 35 while the policy was healthy, then exploded to **216** as the policy
>   drifted (distribution shift) — a ~1000-magnitude gradient into the shared trunk against an
>   actor loss of ~0.1-0.3, i.e. a plausible runaway loop.
>
> `reward_mean` decays monotonically (0.819 -> 0.711 -> 0.418 -> 0.373), so the degradation is
> real in the environment's own terms, not an eval artifact. Next steps, in order: pin the LR
> (disable `kl_adaptive`) to test suspect 1, then detach `h` for the critic path or drop
> `critic_coef` to test suspect 2. Warm-up itself should be kept — it is a strict improvement.
>
> This retroactively explains all three Stage R trials. Trials 1–2 were seeded from the *broken*
> 31.7% Stage E policy, which had little to lose, so the damage read as "noisy trade-off"
> (38.3% success / 39° MPJRE). Seeded from a genuinely good policy the same mechanism is
> unmistakable. **Stage R should not be used in its current form.** The fix to try before any
> further Stage R work is a critic warm-up phase — freeze `action_head`, train `value_head` alone
> until critic loss plateaus, then enable actor updates. Until that exists, the Stage E result
> stands as the Quest2 headline.
>
> Also note the live periodic eval **silently died at 130k** (Isaac Gym OOM on the concurrent sim,
> as training's own allocation grew 0.4GB → 7.2GB via the documented env-stepping leak) and kept
> reporting a stale best. On a 16GB card, trust the post-hoc scan, not live tracking, past ~130k.
>
> <details><summary>Earlier interim reading at ~110k steps (superseded by the above)</summary>
>
> Matched-step aggregate success, both on the 1072-clip held-out test:
>
> | step | `finetuned_backbone` (old) | `trackerr` (fixed) |
> |---|---|---|
> | 20k | 0.212 | 0.351 |
> | 40k | 0.300 | 0.493 |
> | 60k | 0.195 | 0.489 |
> | 80k | 0.249 | 0.571 |
> | 100k | 0.181 | **0.675** |
>
> The old run oscillated in 0.18–0.30 for its entire 500k budget and never beat 0.317 (@440k).
> The fixed run passes that at ~40k steps and is still climbing at 100k. Per-activity at ~110k:
> **walking 0.000 → 0.736, jogging 0.000 → 0.301, interactions 0.000 → 0.547, dancing 0.000 →
> 0.500**; stationary activities also improved across the board (standing 0.689 → 0.985,
> whiteboard 0.647 → 0.904, gestures 0.314 → 0.761). Locomotion/dynamic aggregate **0.000 →
> 0.462**.
>
> For scale: walking 0.736 and jogging 0.301 essentially match SimXR's own 0.755/0.298 — and
> ours is held-out test where theirs is a train-split dump.
>
> </details>

Same idea as the SMPL section above, but for the `quest_humanoid` rig (SimXR's 25-joint skeleton,
`has_upright_start: False`) and the real SimXR Quest2 synthetic dataset
(`data/quest2_synthetic/`, 1072-sequence held-out test split, `test_seg_motion`). Two
"success rate" metrics are reported: `phc_survival` (physics sim never terminates the episode —
the harder, headline number) and `frame_head` (a looser per-frame head-tracking criterion,
consistently ~0.97-0.99 across every distilled checkpoint below, i.e. not a differentiating
metric here).

> **Split caveat — the test split is subject-OVERLAPPING (verified 2026-08-04).** Train/test
> separation itself is correctly enforced everywhere: `data/dataset_quest2.py`'s
> `subdir = 'train_seg_motion' if phase=='train' else 'test_seg_motion'` handles the AvatarPoser
> side, and every PHC run passes `env.motion_file=.../train_seg_motion` for training vs
> `.../test_seg_motion` for eval — 4,097 train clips vs 1,072 test clips, **zero overlapping
> filenames**. However, the split is by (subject, activity) take, **not by subject**: 13 of the 23
> test subjects also appear in train (with disjoint activities — e.g. `SUBJECT008.drinks` is the
> only test take while its other 7 activities are all in train). That makes **326/1072 = 30.4% of
> the test set "seen subject, unseen activity"**, and only the remaining 69.6% a genuine
> unseen-person test. Consequence: absolute numbers here are mildly optimistic as an estimate of
> generalization to a brand-new headset user — state them as a clip-level/subject-dependent split
> if reporting externally. Two things this does NOT affect: (1) every internal comparison
> (Option2 vs `finetuned_backbone` vs `explicit_pose` vs Stage R…) is unaffected, since the same
> split is a constant across all of them; (2) the SimXR comparison stays apples-to-apples — the
> overlap is present in SimXR's own **raw released** `train/`+`test/` directories (verified: same
> 13 subjects, before any processing this project did), and their `split_data.py` only segments
> those pre-existing dirs into clips rather than creating the division. This is the dataset's own
> split convention, not something introduced here. **Confirmed by the CVPR 2024 paper itself**
> (Sec. 4, Datasets): *"We randomly split the data using an 8:2 ratio, resulting in 1758k frames
> for training and 458k frames for testing"* — an explicitly **random** split, so subject overlap
> is expected and the paper makes no cross-subject/subject-independent claim. Their Table 1 lists
> Quest2-Synthetic as `4097 / 1758k` train and `1072 / 458k` test, **exactly matching our local
> clip counts (4097/1072)** — confirming we hold their released split unmodified.
>
> **Metric-definition caveat (from the paper, Sec. 4 Metrics):** SimXR reports joint errors over
> **11 selected common joints** (head, L/R shoulders, elbows, wrists, knees, ankles) — *"rather
> than evaluating all joints as in the prior art"* — because they need one metric consistent
> across their SMPL (24-joint) and Quest (25-joint) humanoids. This repo's
> `scripts/evaluate_phc_metrics_quest.py` computes over **all 25 joints**. Note the upstream
> code's own joint-subsetting (`pred_pos_all = [i[:, _reset_bodies_id] ...]` in
> `im_amp_players.py`) is gated behind `if flags.real_traj:` — i.e. NOT active on the standard
> synthetic-test path — so the replication column here is also all-joints and remains directly
> comparable to the `quest_distill` columns. But **neither is computed under the paper's own
> stated 11-joint definition**, so treat any comparison against numbers quoted from the paper's
> tables as approximate. Also note the paper's success rate is defined per-sequence (*"at every
> point during imitation, the head joint is < 0.5m from the headset pose"*), i.e. a whole sequence
> fails if any single frame exceeds the threshold — closest to our `phc_survival`, not `frame_head`.

> **All five `quest_distill` columns below (Option3 / Option2 / `explicit_pose` / `finetuned_backbone` / both Stage R trials) are SUPERSEDED.** They were trained without the tracking-error observation and are retained as the before-state for that finding, not as current numbers — the comparable current result is `trackerr` @380,160 (0.738 success, 87.43mm), in the ⚠ banner at the top of this section. Do not quote a number from these columns without saying which defect it predates.

| Metric | Phase A (kinematic, no physics)¹ | Phase B (teacher)² | **SimXR reference ablation, no-vision (upstream repo)**⁶ | **Option3 `actionloss` @149.8k (best via periodic eval)**³ | **Option2 `scratch` @399.8k (best via periodic eval)**³ | **`explicit_pose` @490.08k (best via full-range scan)**⁸ | **`finetuned_backbone` @440k (best via full-range scan)**⁹ | **`finetuned_backbone_stager` @443.6k (Stage R trial 1, UNVERIFIED)**¹¹ | **`finetuned_backbone_stager_v2` @442.08k (Stage R trial 2, bug-fixed)**¹² |
|---|---|---|---|---|---|---|---|---|---|
| Success (`phc_survival`) | 1.000 | 0.999 | **0.717**⁶ | 0.241 | 0.294 | 0.293 | 0.317 | **0.383** | 0.326 |
| Success (`frame_head`) | — | 0.9999 | 0.997⁷ | 0.974 | 0.976 | 0.972 | 0.980 | 0.967 | 0.976 |
| MPJPE (mm)⁵ | 40.24 | 21.81 | **112.5**⁶ | 177.67 | 185.87 | 189.14 | 185.60 | 192.04 | 190.01 |
| MPJRE (°) | 10.91 | 4.52 | 29.76⁷ | 25.03 | 26.35 | 29.27 | 25.65 | **39.11** | 37.01 |
| MPJPE (cm)⁵ | 4.02 | 2.18 | **11.25**⁶ | 17.77 | 18.59 | 18.91 | 18.56 | 19.20 | 19.00 |
| MPJVE (cm/s) | 15.47 | 9.29 | 40.93⁷ | 50.33 | 52.91 | 64.29 | 49.94 | 57.66 | 56.13 |
| Jitter (m/s³) | 136.95 | 59.97 | 704.29⁷ | 1232.05 | 1193.38 | 1102.75 | 1117.46 | 834.67 | 969.71 |
| Foot Skate (cm/s) | 29.17 | 5.90 | 14.39⁷ | 20.57 | 21.75 | 22.83 | 20.75 | 20.14 | 18.73 |
| gpen_mean (mm)⁴ | 12.82 | 0.005 | 0.00⁷ | 0.003 | 0.015 | 0.006 | 0.003 | 0.006 | 0.000 |
| gpen_rate⁴ | 0.541 | 0.0001 | 0.000⁷ | 0.000 | 0.001 | 0.000 | 0.000 | 0.000 | 0.000 |

> ¹ Phase A = `AvatarPoser`/`quest_humanoid` backbone alone, no PHC physics simulation at all
> (`main_test_avatarposer_quest2.py`). `results/baselines/avatarposer_quest2_phase_a.json`. Its
> gpen is computed by a different script (kinematic-only, no simulator to reference), not
> affected by footnote 4 below.
>
> ² Phase B = frozen PHC teacher (`phc_prim_quest`) tracking ground-truth motion directly —
> the physics-simulated upper bound this pipeline distills from, not something the student is
> expected to match. `results/baselines/phc_prim_quest_full_test_1072.json` (re-evaluated
> 2026-07-24 for footnote 4's gpen fix; every other metric reproduced identically to the
> 2026-07-14 original, confirming the rollout itself is deterministic).
>
> ³ Best checkpoints found retroactively via full-range `quest_checkpoint_scan.sh` (patience
> disabled), not the final checkpoint — see "Early-stopping doesn't transfer..." in Results
> Discussion below for why that matters.
>
> ⁴ **gpen ground reference fixed 2026-07-24**: previously computed against each sequence's own
> empirical GT-foot-joint minimum, which — since these are joint centers (twist joints), not sole
> vertices — sits a few cm above the real floor, making even well-tracked rollouts read as
> "penetrating." Now computed against the real Isaac Gym ground plane (`Z=0`), matching how the
> SMPL pipeline's `evaluate_phc_metrics.py` checks true mesh vertices against the same real floor.
> **Caveat**: quest_humanoid has no mesh, so this is still a joint-position approximation — since
> Isaac Gym's solver already prevents the real collision geometry from crossing `Z=0`, gpen now
> reads ~0 for essentially any non-collapsed rollout. Treat it as a coarse "did the character
> collapse into the floor" signal, not a fine-grained quality metric; `success_rate`/`pos_error`/
> `rot_error` remain the metrics that actually differentiate checkpoints. `foot_skate_cm_s`
> intentionally still uses a per-sequence reference instead of `Z=0` — quest_humanoid's foot-twist
> joint sits several `H_THRESH` multiples above the true floor even in solid contact, so gating
> against the literal floor collapses this metric to ~0 instead.
>
> `MPJPE (mm)` stays roughly flat across the distilled columns (~178-189mm) despite the real
> success_rate spread — this metric is dominated by failed episodes' large near-constant error,
> not sensitive to successful-episode quality (same pattern as the SMPL pipeline).
>
> ⁵ `MPJPE (mm)` and `MPJPE (cm)` are the exact same underlying metric (mean full-body joint
> position error), reported in two units for continuity with older baseline JSONs; `MPJPE (cm)`
> is the one directly comparable to the primary AMASS table's `MPJPE (cm)` column.
>
> ⁶ ⁷ SimXR's own no-vision ablation, replicated in the separate/upstream SimXR repo
> (`env=env_simxr_quest`, `learning=im_simxr`, vision disabled via `+env.pretrain=True`, same
> `quest2_synthetic` split). Needed an adaptive/KL-based LR schedule (`lr_schedule=adaptive`,
> `kl_threshold=0.008`) to avoid a training collapse the default constant schedule hit — see
> `ablation_results.md` for the full investigation trail. **Values reported here are the peak
> checkpoint, epoch 1,500** (of a 6,104-epoch run; `frame: 24,576,000` per that run's own log) —
> the same run's final checkpoint (epoch 6,105) settles lower, to 51.4% success / 116.5mm MPJPE,
> still well above either Option below but not the number in this table. **Normalized to this
> repo's own step convention**: SimXR's `frame` count sums simulated steps across all `num_envs`
> (64) parallel actors, whereas this repo's own step counter increments once per `env_step()`
> call regardless of `num_envs` (already a per-environment count, since every env steps in
> lockstep) — dividing by `num_envs` recovers the same units: 24,576,000 / 64 = **384,000
> per-environment-equivalent steps**, comparable in scale to Option3's 149,760, Option2's
> 399,840, and `explicit_pose`'s 490,080 (not a dramatically smaller or larger training budget).
> See "SimXR's upstream ablation..." in Results Discussion below for what the results actually
> mean.
>
> ⁸ See "Explicit-pose conditioning vs. the opaque latent" in Results Discussion below.
>
> ⁹ Same recipe as Option2 (z-conditioned, `explicit_pose_action_input=False`), only
> `avatarposer_checkpoint` swapped to the AMASS-then-Quest2-finetuned backbone — see
> "AMASS-pretrained backbone fine-tuned on Quest2" in Results Discussion below for the
> single-variable verification and the result itself.
>
> ¹¹ **UNVERIFIED — do not treat as the new best without reproducibility confirmation.** First
> real trial of Quest2 Stage R (`enable_stage_r=True`), seeded from `finetuned_backbone`'s own
> best checkpoint (`student_0440160.pth`), capped at a 10k-step budget, scanned every 1,000 steps
> (full range, patience disabled). This is the single highest success-rate checkpoint found in
> that scan — but the full 10-point trajectory oscillated wildly (0.3% → 6% → 13% → **38.3%** →
> 33% → 10% → 6% → 11% → 31% → 24%), crashing near-zero twice within the same 10k-step window —
> unlike the regression recipe's established noisy-but-bounded ~0.23–0.29 plateau (see "Early-
> stopping doesn't transfer" below), this is a much more severe swing. See "Stage R trades
> rotation accuracy for survival" in Results Discussion below for what this checkpoint actually
> costs, and why it needs the same reproducibility check `v7_validate` got in the SMPL pipeline
> before being adopted as anything more than a single noisy data point.
>
> ¹² **Second Stage R attempt, with a real bug fixed along the way (not just retuned
> hyperparameters).** `_collect_stage_r_batch()`/`_train_stage_r_on_batch()` never called
> `.eval()`, so the backbone's transformer encoder (PyTorch default `dropout=0.1`, never
> overridden) computed `old_logp`/`logp` under two *different* random dropout masks of the same
> weights — contaminating the PPO ratio/KL with dropout noise indistinguishable from real policy
> drift (the SMPL pipeline's own Stage R already avoids this, a detail missed when porting to
> Quest2). Fixed (`self.student.eval()` in both methods — doesn't affect `backward()`/gradients),
> combined with retuned hyperparameters (`stage_r_lr` 2e-5→5e-6, `kl_threshold` 0.1→0.03,
> `entropy_coef` 0.01→0.003, `lambda_kinematic_stage_r` 0.1→0.3), same seed
> (`student_0440160.pth`), 5k-step budget. Full 13-point scan (250-step resolution, full range):
> best 0.326 @ step 442,080 (barely above the seed's 0.317) — but MPJRE is still 37.01° (vs. the
> seed's 25.65°), and the trajectory still crashed to 0.2% success later on (steps 443,890–
> 444,140) before partially recovering. **Conclusion: the bug fix made the LR-adaptation
> mechanism itself behave correctly (verified via smoke test), but did not eliminate the
> instability or the rotation-accuracy trade-off** — both persisted even with clean gradients,
> pointing at the reward function/training dynamics themselves rather than the earlier numerical
> bug as the primary cause. See "Stage R trades rotation accuracy for survival" below.

### Pre-sim (backbone-only) kinematic quality of each distilled checkpoint

The table above's gpen is entirely **post-sim** (footnote 4) — there is no pre-sim/qref
instrumentation in `DistillPlayerQuest` at all (unlike the SMPL pipeline's `DistillPlayer`/
`qref_gpen.log`). Computed retroactively (2026-07-30) instead, by extracting each checkpoint's
`backbone` sub-dict and running it through the existing Phase-A eval script
(`main_test_avatarposer_quest2.py`) — valid because HMD input features are always derived from the
ground-truth reference motion (never the simulator's own state, see `distill_agent_quest.py`'s
module docstring point 2), so this open-loop kinematic pass reproduces exactly what each
checkpoint's `q_base` would have been at every step of its real closed-loop rollout, without
needing to touch Isaac Gym at all. **Uses the per-sequence empirical ground reference (Phase A's
own convention), NOT the post-sim table's `GROUND_Z=0` — not directly comparable to that table's
gpen column, only to Phase A's own two rows below.**

| Model | rot_error_deg | pos_error_cm | gpen_mean_mm | gpen_rate |
|---|---|---|---|---|
| Phase A (from-scratch) | 10.91 | 4.02 | 12.82 | 0.541 |
| Phase A (AMASS-finetuned) | 10.65 | 3.79 | 11.61 | 0.489 |
| `Option2` | 14.34 | 5.34 | 13.55 | 0.427 |
| `explicit_pose` | 14.29 | 5.32 | 13.54 | 0.411 |
| `finetuned_backbone` | 14.27 | 5.28 | 13.65 | 0.420 |
| `Option3` (pure action-loss)¹⁰ | **90.90** | **47.03** | 15.32 | 0.022 |

Results: `results/baselines/avatarposer_quest2_presim_{option2,option3,explicit_pose,
finetuned_backbone}.json`.

> ¹⁰ Not a real improvement despite the lowest gpen_rate — `Option3` trains with
> `lambda_kinematic=0.0` (zero kinematic supervision at all, pure action-loss), and its backbone's
> pose reconstruction is consequently badly broken (90.9° rotation error, 47cm position error, vs.
> ~14.3°/~5.3cm for every kinematically-supervised variant). Its low gpen_rate reflects a mostly-
> incoherent pose that happens not to dip below the per-sequence floor much, not cleaner motion —
> see "Pre-sim quality vs. post-sim outcome" in Results Discussion below.

### Results Discussion

> **The findings below are from the SUPERSEDED (pre-`use_tracking_error`) runs.** Compressed
> 2026-08-13 from a chronological write-up into claims only; full prose and dates are in git
> history. Several are now explained by, or subsumed into, the root-cause finding in the ⚠ banner
> at the top of this section.

- **Locomotion was exactly 0.000.** `finetuned_backbone`'s per-activity split was binary:
  stationary 0.60–0.80, and **0/461 on every locomotion/dynamic clip** (walking 0/144, jogging
  0/136, interactions 0/75, dancing 0/20, martialarts+stretching+retail+workout 0/41). The 31.7%
  headline was close to a restatement of "what fraction of this test set is stationary", which is
  why averaged MPJPE/MPJRE barely moved across those columns. **Root-caused and fixed** — see the
  ⚠ banner; the same split is now 0.512 on those 461 clips.
- **Near-zero gpen was a symptom of that, not of safer motion** — the character was often standing
  still rather than attempting the reference at all (confirmed by direct rollout inspection at an
  early 11.85%-success checkpoint).
- **Explicit-pose conditioning (`q_base`) ≈ opaque latent `z`: no measurable difference** (29.3% vs
  Option2's 29.4%). With the SMPL pipeline's `v8` (concatenating `q_base` *alongside* `z`, also a
  no-op), these triangulate: **which pose representation feeds the action head is not the
  bottleneck.** That conclusion survives the root-cause finding — the bottleneck was the *missing*
  feedback signal, not the pose encoding.
- **Joint kinematic+action training degrades the backbone** (~10.9°→~14.3°, ~4.0cm→~5.3cm) for all
  three variants, despite `lambda_kinematic=1.0` dominating `lambda_action=0.1`. Action-loss
  gradients and/or the `QuestHumanoidBodyModel` FK path pull it away from per-frame accuracy
  regardless of loss weighting. Not revisited post-fix.
- **Early stopping does not transfer from the SMPL/Stage-R pipeline.** `success_rate` is noisy and
  non-monotonic across the *entire* range for this recipe (Option2 oscillates ~0.23–0.29 from 100k
  to 500k). Patience=2 would have stopped both Option2 and Option3 hundreds of thousands of steps
  before their real best. **Always use a full-range scan or live periodic eval.** Reconfirmed on
  the fixed run, which also plateaus and whose best (380k) beats its final (500k).
- **Stage R traded rotation accuracy for survival, twice.** Trial 1: 38.3% success but MPJRE
  38–48° vs the 25.65° seed. Trial 2, after fixing a real bug (`_collect_stage_r_batch`/
  `_train_stage_r_on_batch` never called `.eval()`, so PPO's ratio compared log-probs under
  *different* dropout masks — batch-1 KL dropped ~40-50x once fixed) and with conservative
  hyperparameters: only 32.6%, MPJRE still 37.0°, trajectory still crashing to ~0.2% before
  partially recovering. **The bug fix corrected the numerics but eliminated neither the
  instability nor the rotation trade-off** — it is a property of the reward setup, which never
  directly supervises rotation. Neither trial was adopted.
- **The `finetuned_backbone` 2.3pp "win" is probably noise.** A frozen-backbone probe (freeze each
  backbone, train fresh `state_enc`/`action_head` for a matched 50k budget) put `Option2` *ahead*
  by 2.1pp — same magnitude, opposite direction, both inside Option2's own oscillation band. Two
  cheaper checks (raw `z` cosine similarity; "frankenstein" head/backbone swaps, 0.000 success
  both ways) were uninformative because they test basis compatibility, not information content.
  The probe did establish one thing cleanly: **frozen backbones land at ~20–22% vs ~29–32%
  unfrozen**, so keeping the backbone trainable is worth ~8–10pp. Reproduce:
  `bash scripts/run_frozen_backbone_probe.sh [max_distill_step] [num_envs]`.
- **SimXR's replication handles locomotion** (walking 0.755, jogging 0.298, interactions 0.568,
  dancing 0.161) versus 0.000 for all of those here, while stationary gaps were far smaller — i.e.
  essentially the entire aggregate gap to SimXR was locomotion. *Its dump is a TRAIN-split eval,
  so those absolute rates are optimistically biased; the safe reading is only "SimXR's policy can
  locomote, ours could not".*

### Quest2 naive baseline (2026-08-11) — no distillation student at all

Naivew baseline: frozen teacher tracks
AvatarPoser's raw output fro Quest 2 motion. 

Location of script for experiment: `scripts/convert_avatarposer_quest2_to_phc.py` It requires no training — one conversion pass plus one eval rollout.

Recipe: `convert_avatarposer_quest2_to_phc.py` with the *same* fine-tuned backbone behind the
`finetuned_backbone` column (`AvatarPoseEstimationQuest2Finetune/models/15000_G.pth`) over all
1072 test sequences → frozen `phc_prim_quest` teacher tracks those predicted pkls → scored against
ground truth with `scripts/score_naive_baseline_vs_gt.py`. Result:
`results/baselines/quest2_naive_baseline_vs_gt.json`.

| Metric | Phase B (teacher on GT) | **Naive baseline (this)** | SimXR no-vision ablation | `finetuned_backbone` @440k |
|---|---|---|---|---|
| Success (`phc_survival`) | 0.999 | **0.767** | 0.717 | 0.317 |
| Success (`frame_head`) | 0.9999 | **0.974** | 0.997 | 0.980 |
| MPJPE (mm) | 21.81 | **96.4** | 112.5 | 185.60 |
| MPJRE (°) | 4.52 | **13.82** | 29.76 | 25.65 |
| MPJVE (cm/s) | 9.29 | **24.17** | 40.93 | 49.94 |
| Foot skate (cm/s) | 5.90 | 21.41 | 14.39 | 20.75 |

Per-activity `phc_survival`: walking **0.792**, jogging **0.779**, standing 0.985, whiteboard
0.974, gestures 0.830, ROM 0.762, interactions 0.533, dancing 0.500 — against **0.000 for walking,
jogging, interactions and dancing** in the distilled column. Locomotion is not hard for this
pipeline; it was unreachable for that student.

**This is the best Quest2 result in this table**, and it beats SimXR's own no-vision ablation on
success, MPJPE, MPJRE and MPJVE — on the held-out test split, where SimXR's per-activity dump is a
train-split eval. Same framing caveat as the SMPL naive baseline applies: it solves an easier
problem than the distilled student (it hands the teacher a full per-frame kinematic target instead
of compressing tracking into a 3-point-only network), so it belongs in the table as the control
that bounds what the physics stage can do, not as a headline accuracy claim over distillation.

> **Alignment matters here and is easy to get wrong.** The predicted pkls start `window_size+1`
> frames into each clip, and PHC's own dump lags by one further frame. Scoring at offset 0 reports
> **568mm MPJPE / 0.108 success** — pure temporal misalignment (a walker covers ~1.5m in 42
> frames), not tracking error. The offset of **42** is derived, not tuned: 41 from the prediction
> pipeline (`dataset_quest2`'s test branch drops 2 raw frames, and the converter predicts the
> window's `(W-1)`th element), plus 1 for PHC's dump lag — the latter established independently on
> the **Phase B** artifacts, where the teacher tracked ground truth with no windowing at all and
> still aligns best at offset 1 (23.6mm vs 37.1mm at offset 0, matching Phase B's published
> 21.81mm). Anything scoring a PHC dump against source pkls must apply this.

**SimXR's upstream ablation: much better success/position, but not rotation accuracy.**
Replicating SimXR's own no-vision ablation in the upstream repo scores far higher on success
(71.7% vs. 24–29%) and position error (112.5mm vs. 178–189mm) than every `quest_distill` variant
tried here — but `MPJRE` (rotation error) is *not* better (29.76° there vs. 25–29° here), if
anything slightly worse.

> **CORRECTION (2026-08-11): the "different training objective" explanation below was wrong, and
> the config side-by-side it asked for has now been done.** SimXR's no-vision quest run is *not*
> PPO+AMP+reward. `env_simxr_quest.yaml` sets `distill: True`, `only_kin_loss: True`, and
> `models: ['output/HumanoidIm/phc_prim_quest/Humanoid.pth']` — **it is a student behavior-cloned
> from the exact same frozen teacher checkpoint this repo distills from**, and `amp_agent.py:243`
> comments that branch "pure behavior cloning, kinemaitc loss" while stepping the sim with the
> student's own action (`env_step(res_dict['mus'])`). Same teacher, same objective, same
> student-driven rollout, same 0.5m termination threshold (`im_amp_players.py:43`, identical line
> in both repos). So 71.7% vs 31.7% is a **like-for-like** comparison, not an
> apples-to-oranges one.
>
> The structural difference is the observation space. SimXR's policy consumes
> `compute_imitation_observations_v6` over `trackBodies: ["b_head"]` — `ref_body_pos - body_pos`
> plus rotation/velocity deltas, rotated into the humanoid's heading frame, i.e. **the head's
> tracking error**. Ours consumed no equivalent signal at all (see the ⚠ banner at the top of this
> section). Note SimXR gets *less* reference information than we do — head only, versus our
> head + 2 wrists into the backbone — and still locomotes, which is further evidence the deficit
> was closed-loop feedback rather than input richness.
>
> The MPJRE split is still real and still unexplained by the above: SimXR never directly
> supervises per-joint rotation, whereas `distill_agent_quest.py` adds an explicit
> `kinematic_loss` on rotations, which plausibly explains why our rotation error is competitive
> while everything else is not. That part of the original reasoning survives; the "PPO + AMP +
> head reward" premise does not.

~~A pure bug in a shared code path would be expected to move every metric
in the same direction; this split (success/position much better, rotation not) is more consistent
with a genuine algorithmic difference: SimXR's `im_simxr` trains with full PPO + an AMP
discriminator + a head-tracking task reward, which directly rewards keeping the head on-target
and motion looking natural, but never directly supervises exact per-joint rotation against the
reference the way `distill_agent_quest.py`'s pure action-regression loss does. Not confirmed — a
direct side-by-side of the two training code paths would be needed to rule out a remaining port
bug — but the MPJRE split is real, non-obvious evidence pointing toward "different training
objective," not just "bug."~~

**AMASS-pretrained backbone fine-tuned on Quest2 — kinematic gate, PASSED (still valid).**
Warm-starting the Quest2 backbone from `model_zoo/avatarposer.pth` instead of training from
scratch (transferring everything except `joint_rotation_decoder`'s final layer, 126-dim for
SMPL vs 144 for quest_humanoid) improved **every** kinematic metric, none worse:

| Metric | From-scratch | Fine-tuned from AMASS |
|---|---|---|
| rot_error_deg | 10.91 | 10.65 |
| pos_error_cm | 4.02 | 3.79 |
| vel_error_cm_s | 15.47 | 14.50 |
| jitter_m_s3 | 136.95 | 136.46 |
| foot_skate_cm_s | 29.17 | 23.34 |
| gpen_mean_mm | 12.82 | 11.61 |
| gpen_max_mm | 41.88 | 38.53 |
| gpen_rate | 0.541 | 0.489 |
| eg_mpjpe_mm | 40.24 | 37.85 |

Gains are modest (2–20%) — the from-scratch fit was already near the AMASS/SMPL baseline
(4.05°/2.51cm), leaving little headroom — but every metric moving the same way is signal: the
AMASS encoder transfers useful "sparse head/wrist → full-body configuration" structure across
genuinely different skeletons. This is a backbone result and is **unaffected by the
distillation defect**; it is why `trackerr` uses this checkpoint.
`results/baselines/avatarposer_quest2_phase_a_finetuned.json`,
`results/AvatarPoseEstimationQuest2Finetune/models/15000_G.pth`.

> Downstream, this gate's improvement appeared to transfer through physics (31.7% vs Option2's
> 29.4%, every metric better) — but see the frozen-backbone-probe bullet above: that 2.3pp is
> most likely noise. Both are superseded by `trackerr` regardless.

### Per-clip metrics for the rendered clips

Everything in this section is averaged over all 1072 test clips. For the **individual clips that
were rendered to video** (one per motion category: planted / full-body-transition / locomotion),
with the same metric definitions plus a root-drift vs root-relative-pose decomposition, see
**`results_per_clip_renders.md`** (generated by `scripts/per_clip_metrics.py`). Videos live in
`output/renderings/`.

### Quest2 eval commands

```bash
# Single checkpoint, nearest-match by step (physics rollout + full metric set).
# num_envs (default 128) does NOT meaningfully reduce GPU memory use -- fixed overhead
# dominates -- so don't rely on it to dodge OOM from a concurrently-running training job:
bash scripts/auto_eval_quest.sh <exp_name> <target_step> [extra_hydra_overrides] [num_envs]

# Future training runs — always trains to max_distill_step, periodic real eval throughout,
# reports best checkpoint found (no early kill):
bash scripts/train_distill_quest_with_periodic_eval.sh <exp_name> <max_distill_step> \
     [num_envs] [eval_interval] [max_crash_retries] [gpu_free_mib_threshold] [extra_hydra_overrides]

# Retroactive analysis of an already-finished run (no live training process). patience should
# be set high (e.g. 999) to effectively disable early-stop -- see footnote 3, it's unreliable
# for this recipe. extra_hydra_overrides is REQUIRED for any run not trained with every
# learning.params.config default (e.g. explicit_pose_action_input=True, see footnote 8):
bash scripts/quest_checkpoint_scan.sh <exp_name> [eval_interval] [patience] [start_step] \
     [extra_hydra_overrides]

# Resuming an INTERRUPTED scan: quest_checkpoint_scan.sh does NOT know it was killed partway --
# re-running with the same start_step blindly redoes every point from the beginning (overwriting
# existing scan_eval_<step>.log files), it does not skip already-done ones. To actually resume,
# pass start_step = the step that was mid-eval when killed (redoes that one cleanly -- its log
# file is a giveaway, roughly half the size of a complete one -- then continues normally from
# there). Real example (2026-07-31, quest_distill_finetuned_backbone_stager_v2's Stage R scan,
# killed mid-eval at step 442140 out of a 440640-445184 range):
nohup bash scripts/quest_checkpoint_scan.sh quest_distill_finetuned_backbone_stager_v2 250 999 442140 \
  "learning.params.config.enable_stage_r=True" \
  > phc/output/HumanoidIm/quest_distill_finetuned_backbone_stager_v2/scan_resume.log 2>&1 &
disown
# CAVEAT: the resumed invocation's own "best so far" tracking restarts at -1/0 -- its final
# "[scan] DONE. Best checkpoint..." line only reflects the RESUMED range, not the full one.
# Reconcile the true best manually across all scan_eval_<step>.log files from both the original
# and resumed invocations, don't trust the resumed run's own summary line in isolation.

# Manual (what auto_eval_quest.sh does internally):
#   Step 1: physics rollout (rlgpu env, from phc/)
cd phc && conda run -n rlgpu --no-capture-output python phc/run_hydra.py \
  learning=distill_quest exp_name=<EXP> env=env_quest_im_basic robot=quest_humanoid \
  env.motion_file=data/quest2_synthetic/test_seg_motion \
  env.num_envs=128 env.cycle_motion=False env.min_length=5 \
  learning.params.config.student_checkpoint=output/HumanoidIm/<EXP>/student/<CKPT>.pth \
  test=True im_eval=True headless=True disable_multiprocessing=False

#   Step 2: compute metrics (avatarposer env, from repo root)
conda run -n avatarposer --no-capture-output python scripts/evaluate_phc_metrics_quest.py \
  --pos_dump phc/output/HumanoidIm/<EXP>/phc_positions_for_eval.pkl
```
