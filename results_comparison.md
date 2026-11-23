# Results Comparison

Two pipelines: **AMASS/SMPL** (original AvatarPoser dataset, `smpl_humanoid` rig) and **Quest2**
(SimXR synthetic dataset, `quest_humanoid` rig). Each distills AvatarPoser's kinematic backbone
into a small student policy that drives a frozen PHC physics teacher from sparse HMD tracking
alone.

## Glossary — what each variant changes, architecturally

| Name | Pipeline | Architecture |
|---|---|---|
| **Naive baseline** | Both | No distillation student at all — the frozen PHC teacher tracks the backbone's raw kinematic output directly. Isolates "how good is physics-correcting the kinematic prediction" from "how good is the compressed policy." |
| `frozen_bb` | AMASS | Backbone frozen; only a lightweight action head trained on top. |
| `v7` | AMASS | Backbone frozen; `state_enc`+`action_head` trained via DAgger-style behavior cloning (Stage 1) then on-policy fine-tuning (Stage E) against the frozen teacher, on the validated HMD input pipeline (6 bugs fixed — `distill_hmd_joint_bug.md`). |
| Stage R (`_stager`/`_validate`) | AMASS + Quest2 | A Stage E checkpoint further fine-tuned with PPO against a reward, instead of pure behavior cloning. |
| `v8` | AMASS | `v7` + `action_head`/`value_head` also given explicit `q_base` (decoded pose) alongside `z`. Negative — redundant with `z`. |
| `goal-delta-input` | AMASS | `v7` + `action_head` given the env's full obs (self-obs + task-obs) instead of self-obs only. Negative for BC, much worse under PPO. |
| Phase A | Quest2 | `AvatarPoser`/`quest_humanoid` kinematic backbone alone — no PHC physics. |
| Phase B | Quest2 | Frozen PHC teacher (`phc_prim_quest`) tracking ground truth directly — the physics upper bound. |
| SimXR no-vision ablation | Quest2 | SimXR's own reference student (separate/upstream repo), behavior-cloned from the same frozen-teacher family, vision branch disabled. Same objective as `Option2`/`trackerr`, different codebase. |
| `Option2`/`scratch` | Quest2 | Backbone trained from scratch on Quest2; `action_head` conditioned on the opaque AvatarPoser latent `z`. Pre-fix — no tracking-error feedback. |
| `Option3`/`actionloss` | Quest2 | Same as `Option2` but `lambda_kinematic=0.0` — zero kinematic supervision, pure action-loss. Backbone pose reconstruction ends up badly broken. |
| `explicit_pose` | Quest2 | `Option2` but `action_head` conditioned on the explicit decoded pose `q_base` instead of `z`. No measurable difference from `Option2`. |
| `finetuned_backbone` | Quest2 | `Option2` recipe, backbone warm-started from the AMASS-pretrained checkpoint instead of trained from scratch. Pre-fix — still no tracking-error feedback. |
| **`trackerr` (current best)** | Quest2 | `finetuned_backbone` + `use_tracking_error`: appends a 27D reference-minus-simulated position/rotation delta at the 3 HMD-tracked bodies (heading-frame, deployable — not an oracle) to `action_head`/`value_head`'s input. Closes the closed-loop feedback gap that caused 0% locomotion success in every earlier Quest2 variant. Also fixes a velocity-feature fps bug (1/60→1/30). |

---

## AMASS/SMPL Pipeline

### Primary results

Full student (backbone + action head) in IsaacGym physics simulation on PHC test set (536 sequences).
Results saved to `phc/output/HumanoidIm/<EXP>/phc_eval.log`.

| Model | Checkpoint | Success | MPJPE (cm) | qref_MPJPE (cm) | MPJVE (cm/s) | MPJRE (°) | Jitter (m/s³) | Foot Skate (cm/s) | gpen_mean (mm) | gpen_rate |
|---|---|---|---|---|---|---|---|---|---|---|
| `frozen_bb` — frozen backbone | 489.5k | 0.584 | 34.83 | — | 89.39 | 38.73 | 1188 | 2.66 | — | — |
| `v2`–`v6` — pre-v7 refinement-branch line (7 rows, collapsed)† | 431k–1.95M | 0.440–0.498 | 41.14–50.71 | 21.57–61.40 | 99.59–113.59 | 40.86–45.02 | 1345–1612 | 2.28–2.81 | 0.04–0.74 | 0.003–0.167 |
| **`v7` — action distillation (Stage E), post HMD-pipeline fix**¹ | 250k | 0.545 | 38.18 | 10.42 | 96.69 | 40.31 | 1337 | 2.41 | 2.46 | 0.745 |
| `v7_stager` — Stage R, 10k steps, coarse manual milestone | 260k | 0.644 | 30.84 | 10.72 | 82.88 | 37.35 | 1028 | 2.13 | 1.24 | 0.388 |
| `v7_stager` — Stage R, 30k steps (regressing) | 280k | 0.506 | 39.71 | 10.50 | 97.40 | 44.75 | 1275 | 1.64 | 1.30 | 0.375 |
| `v7_stager` — Stage R, 50k steps (COLLAPSED — do not use) | 300k | 0.062 | 53.95 | 10.12 | 161.63 | 59.68 | 2433 | 1.16 | 1.04 | 0.333 |
| **`v7_validate` — Stage R, 6k steps, collapse-detection script — BEST distilled result⁴** | 256k | **0.756** | **24.59** | 10.76 | 68.99 | 34.83 | 773 | 1.66 | 1.46 | 0.388 |
| `v8` — explicit `q_base` input (negative result⁵) | 250k | 0.521 | 40.79 | 10.69 | 100.09 | 40.93 | 1370 | 2.36 | 1.23 | 0.382 |
| `v8_stager` — Stage R, 10k steps (v8's peak, still worse than v7_validate) | 260k | 0.634 | 33.61 | 10.81 | 88.51 | 39.67 | 1210 | 1.91 | 1.31 | 0.380 |
| `goal-delta-input` — full obs instead of self-obs only (negative result⁶) | 250k | 0.539 | 38.85 | 10.67 | 94.02 | 40.45 | 1235 | 2.41 | 1.23 | 0.378 |
| `goal-delta-input` — Stage R, 10k steps (peak, collapses fast — do not use) | 260k | 0.332 | 51.38 | 10.30 | 126.06 | 49.87 | 1665 | 1.85 | 1.05 | 0.355 |
| **Naive baseline — frozen PHC teacher tracks raw AvatarPoser output, no student²** | phc_kp_mcp_iccv | **0.912** | **14.06** | — | 43.39 | 18.93 | 318 | 0.88 | (23.79)³ | (0.654)³ |

> `qref_MPJPE` = kinematic accuracy before physics sim; `MPJPE` = physics sim trajectory accuracy — different outputs.
> `gpen_mean`/`gpen_rate` are **pre-sim** (q_ref/q_base), for rows above the naive baseline; rows with no
> reconstruction step (`frozen_bb`, `v2`, `v3`) have no pre-sim reference, hence `—`.
> AvatarPoser kinematic baseline: MPJPE 2.51cm, Jitter 1357 m/s³, Foot Skate 33.67 cm/s, gpen_mean 23.79mm, gpen_rate 0.654.
>
> † **Pre-v7 rows (2026-08-13):** all superseded by `v7`, and their `qref_MPJPE` was computed on a
> broken HMD pipeline (¹) — not comparable to anything below. In one line each: v4 was the best of
> the line (750k); v5's position-space `L_ref` improved qref_MPJPE 61→22cm but spiked gpen to 16.7%
> (conflicted with encoder training); v6's gradient isolation protected physics but left the
> refinement head producing near-zero corrections. Full per-version numbers in git history.
>
> ¹ Six bugs in HMD feature construction fixed 2026-07-06/07 (joint indices, rotation frame,
> feature layout, 6D rotation convention, velocity frame-rate, FK position reconstruction) — took
> `qref_MPJPE` from ~56–61cm to ~10.3–10.7cm. Full writeup: `distill_hmd_joint_bug.md`. Don't
> compare pre-v7 `qref_MPJPE` against v7's.
>
> ² Evaluated on the full 536-sequence test set 2026-07-07. **Beats every distillation/Stage R
> result above on every metric** (2x MPJPE, 41% success, 3x jitter). *(Audit note 2026-11-23:
> those ratios predate `v7_validate`. Against the best student, `v7_validate`, the gap is 1.75x
> MPJPE (14.06 vs 24.59 cm), +0.156 absolute success (0.912 vs 0.756) and 2.4x jitter (318 vs
> 773).)* Not because distillation is
> broken — it solves a harder problem (compressing full-body tracking into a network that only
> ever sees a sparse 3-point HMD signal) — but don't present distillation as an accuracy win over
> this baseline without closing the gap; reframe around latency/compute instead.
>
> > **⚠ Open question (2026-08-13):** this "harder problem" framing is unconfirmed for this
> > pipeline. Quest2 had the identical-shaped defect — the student couldn't observe its own
> > tracking error — and fixing it closed nearly the whole gap to *its* naive baseline (0.317→0.738
> > success vs. naive's 0.767; see Quest2 section). `DistillAgent` here feeds `action_head` the same
> > `[z, state_enc(self_obs)]` with task-obs dropped, so the defect is *structurally present*. The
> > one counter-evidence (`goal-delta-input`, footnote ⁶) fed a delta and didn't help — but it used
> > ground-truth task-obs, not a deployable HMD-vs-current delta like Quest2's fix, so it doesn't
> > settle this. Treat "harder problem" as a hypothesis, not a conclusion, until someone reruns it.
>
> ³ Naive baseline has no q_ref/backbone step — pre-sim gpen is identical to the AvatarPoser
> kinematic baseline (23.79mm/0.654), not a new number. Its **post-sim** gpen (computed directly on
> PHC's output, since no `qref_gpen.log` exists for a non-distillation run) is 0.00mm/0.000 —
> physics eliminates penetration entirely here; don't read it as a pre-sim number.
>
> ⁴ Found via `scripts/train_stage_r_with_early_stop.sh`'s collapse-detection sweep from the same
> `v7`@250k checkpoint. That sweep scores checkpoints with `scripts/auto_eval.sh` on
> `phc_test.pkl`, the same 536-sequence test set reported here, so this row is test-set-selected. Confirmed deterministic across 3 reruns, not luck. **Best distilled
> checkpoint overall** — still well behind the naive baseline on every metric.
>
> ⁵ Hypothesis: explicit `q_base` access (vs. only implicitly via `z`) would help. Result: v8 is
> *worse* than v7 at every checkpoint — `q_base` is a deterministic function of `z`, which
> `action_head` already receives, so this only adds redundancy.
>
> ⁶ Hypothesis: dropping the env's task-obs slice (the reference-vs-current delta the teacher
> itself uses) starved `action_head` of that signal. Result: Stage E unchanged from v7 (0.539 vs
> 0.545), Stage R far worse (collapses from the first checkpoint instead of improving first) —
> explicit ground-truth delta conditioning made the policy more brittle under PPO, not less.

### Backbone kinematic quality (`main_test`, backbone only)

Backbone pose predictions vs. AMASS GT — independent of the physics sim.

| Model | Backbone MPJPE (cm) | Notes |
|---|---|---|
| AvatarPoser (pretrained) | 2.51 | baseline |
| `frozen_bb` | ~2.51 | backbone never modified |
| `v7`/`v7_stager` | ~10.3–10.7 | backbone never modified; gap vs. 2.51 believed to be the reconstructed-input approximation ceiling (FK reconstruction has a validated ~2–4cm/joint residual) — not chased further |

### Eval commands

```bash
# Automated (watches for checkpoint, runs both steps):
bash scripts/auto_eval.sh <exp_name> <step>

# Manual — Step 1: run student in sim (rlgpu env, from phc/). REPO = this repository's root.
conda run -n rlgpu --no-capture-output bash -c "
unset PYTHONPATH && export WANDB_MODE=offline
cd \"$REPO/phc\" && python phc/run_hydra.py \
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

## Quest2 Pipeline

Same idea, for the `quest_humanoid` rig (SimXR's 25-joint skeleton, `has_upright_start: False`)
and the real SimXR Quest2 synthetic dataset (1072-sequence held-out test split,
`test_seg_motion`). Two success metrics: `phc_survival` (episode never terminates — the harder,
headline number) and `frame_head` (looser per-frame head-tracking criterion, ~0.97–0.99 across
every checkpoint, not differentiating).

### Root cause of 0% locomotion (found 2026-08-11), and the fix

Every Quest2 checkpoint before `trackerr` had a structural bug: `action_head`'s inputs (`z`, the
AvatarPoser latent describing only the *reference*, and `state_enc(proprio)`, which strips world
XY/heading by construction) could not express "how far am I from the reference and which way" —
the student's own tracking error was not a function of its observations at all. The frozen
teacher it distills from *does* get that signal (PHC's task-obs). An open-loop predictor keeps
stationary error bounded but cannot servo to a moving reference — hence 0.60–0.80 on stationary
activities and **exactly 0.000 on all 461 locomotion/dynamic clips**, across every pre-fix
variant.

**Fix**: `use_tracking_error` (default `True`) appends a 27D reference-minus-simulated delta at
the 3 HMD-tracked bodies, in the heading frame, to `action_head`/`value_head`'s input —
deployable (a real headset knows head/wrist position; it never sees the reference's full-body
pose), unit-tested for zero-error/heading/translation invariance. A secondary fps bug (HMD
velocity features queried at 1/60 when the backbone was trained on 30fps deltas) cost an
additional +0.27°/+4.4mm on locomotion — fixed via `_HMD_VEL_DT_QUEST = 1/30`.

Also run for the first time alongside this fix: a proper **naive baseline** (frozen teacher
tracking the backbone's raw output directly, no student) — previously missing from this pipeline.

### Current results

`trackerr` = `finetuned_backbone` (same backbone, same teacher, same recipe) + `use_tracking_error`
+ the fps fix, as the only real variables. Stage E only (`enable_stage_r: false`; Stage R is
currently broken — see below). Best checkpoint `student_0380160.pth`, found via a full-range
scan (patience disabled — early stopping doesn't work for this recipe, see Lessons below).

> **Reporting status (audit 2026-11-23). Read before citing any number in this table.**
>
> - **Test-set checkpoint selection.** `student_0380160.pth` was chosen by
>   `scripts/quest_checkpoint_scan.sh`, which ranks checkpoints by `phc_survival` on
>   `test_seg_motion`, the same 1,072 clips reported here. No validation split was used. The
>   SimXR column's epoch-1,500 checkpoint was also chosen as the test-set peak of 12 checkpoints.
>   The Naive PHC baseline involves no checkpoint selection. The two trained columns are
>   therefore optimistically biased relative to Naive PHC. Final checkpoints, for comparison:
>   `trackerr`@500k = 0.707, SimXR epoch 6,105 = 0.514 (`scripts/simxr_novision/README.md`).
> - **Verified against tracked files:** Naive PHC (`results/baselines/quest2_naive_baseline_vs_gt.json`,
>   including per-activity rates), Phase A (`results/baselines/avatarposer_quest2_phase_a_finetuned.json`),
>   and all Mesh GPen rows (`scripts/dataset_mesh_gpen_aggregate.json`).
> - **Recorded in contemporaneous notes only:** the Finetuned Backbone (`trackerr`) post-sim metrics
>   and per-activity rates, and the SimXR column. Their raw outputs
>   (`output/HumanoidIm/quest_distill_trackerr/quest_eval_0380160.json`, rollout dumps and the
>   SimXR checkout) were never committed and were **lost with the original workstation**, so they
>   cannot be re-checked. `QUEST2_TRAINING_REPRODUCTION.md` quotes the `trackerr` eval JSON to full
>   precision.
> - **SimXR column mixes two evaluation runs.** Success 0.717 and MPJPE 112.5 mm are from SimXR's own
>   evaluator (`eval_peak_1500.log`, `num_envs=512`). MPJRE, MPJVE, Jitter and Foot Skate come
>   from this repo's `evaluate_phc_metrics_quest.py`, run on a `num_envs=64` re-run dump (which
>   gave 0.7155 / 112.36 mm on the same checkpoint; see `simxr_ref/README.md`).
> - **Unresolved: `trackerr` backbone initialisation.** This table describes `trackerr` as using
>   the AMASS-finetuned backbone (`results/AvatarPoseEstimationQuest2Finetune/`). But according to
>   `QUEST2_TRAINING_REPRODUCTION.md`, the committed `distill_quest.yaml` default is
>   `results/AvatarPoseEstimationQuest2/models/15000_G.pth` (from scratch), and the reconstructed
>   launch command passes no override.
>   - The recovered PHC fork's history shows this from-scratch path as the default at **every**
>     committed revision.
>   - The agent's own docstring says the earlier `finetuned_backbone` round swapped in the
>     AMASS-finetuned checkpoint through a command-line override.
>   - No record shows whether `trackerr` received the same override. The checkpoints that could
>     settle it were lost.
>
>   "Finetuned Backbone" is therefore the **intended, not verified,** initialisation of
>   `trackerr`.

| Metric | Stage | Ground Truth *(control)* | Phase A / Kinematics Only | Naive PHC Baseline | SimXR Baseline | **Finetuned Backbone** |
|---|---|---|---|---|---|---|
| Success (`phc_survival`) | post | — | N/A¹ | **0.767** | 0.717 | 0.738 |
| Success (`frame_head`) | post | — | — | 0.974 | **0.997** | 0.989 |
| MPJPE (mm) | ² | — | *37.85* | 96.38 | 112.5 | **87.43** |
| MPJRE (°) | ² | — | *10.65* | **13.82** | 29.76 | 15.23 |
| MPJVE (cm/s) | ² | — | *14.50* | **24.17** | 40.93 | 34.25 |
| Jitter (m/s³) | ² | — | *136.46* | **221.94** | 704.29 | 524.78 |
| Foot Skate (cm/s) | ² | — | *23.34* | 21.41 | **14.39** | 17.14 |
| **Mesh GPen Mean (mm)** | **pre-sim** | *1.72* | 3.27 | 3.27¹⁸ | N/A⁴ | 4.48 |
| **Mesh GPen Rate** | **pre-sim** | *0.017* | 0.018 | 0.018¹⁸ | N/A⁴ | 0.024 |
| **Mesh GPen Mean (mm)** | **post-sim** | — | N/A⁵ | **2.56** | 3.30 | 6.03¹⁹ |
| **Mesh GPen Rate** | **post-sim** | — | N/A⁵ | **0.020** | 0.031 | 0.041¹⁹ |

> **Mesh GPen = ground penetration of the RECONSTRUCTED SMPL-H avatar, not of PHC's collision
> bodies.** Definition: `utils.physical_metrics.compute_ground_penetration` — Mean is the mean depth
> over penetrating (vertex, frame) pairs; Rate is the fraction of *frames* with ≥1 vertex below the
> floor; tolerance 1e-4 m; floor Z = 0; SMPL-H neutral, betas = 0, 6890 vertices. All values are the
> **full 1072-clip test set**, equal-weight per clip, every stream through the identical
> Quest→SMPL-H IK. Isaac Gym's own rigid-body penetration is ≈0.000 mm for every method here and
> measures a different quantity; it is deliberately not used.
>
> ¹ Phase A is the kinematic backbone alone — no physics runs, so `phc_survival` is undefined.
>
> ² **Stage differs by column.** Phase A's italicised values are **pre-simulation** (kinematic
> output vs GT); Naive PHC, SimXR and Finetuned Backbone are **post-simulation**. Phase A's 37.85 mm
> MPJPE is therefore *not* comparable to the others' 87–112 mm — it has not been through physics.
>
> ¹⁸ Naive PHC's pre-sim reference **is** Phase A's output (it tracks that backbone), so this is the
> same measured stream, not an independent measurement.
>
> ⁴ SimXR maps sensors directly to actions with no intermediate pose ("no intermediate
> representations" is the paper's central claim), so pre-sim GPen is **undefined**, not unmeasured.
>
> ⁵ Phase A is never simulated, so it has no post-sim stage.
>
> ¹⁹ **Do not read 4.48 → 6.03 as "physics makes penetration worse."** The completed diagnosis
> (`scripts/diag_fb_gpen/DIAGNOSIS.md`) found a three-part mechanism: (a) the Quest rig has **no toe
> joint**, so Quest→SMPL-H retargeting leaves a shared **~34–40 mm joint-to-sole offset** present in
> *every* stream including pre-sim; (b) simulation lowers the body, cutting Finetuned Backbone's
> mesh-sole clearance from 73 mm to 47 mm; (c) its faster, less-settled foot motion (feet 1.31×
> faster, 19% harder landings) drives larger excursions that exceed the reduced margin.
> **30.8% of its penetrating frames occur while the foot is airborne** — where real penetration is
> impossible — so roughly a third of the measured excess is reconstruction-induced, not simulator
> contact penetration. Naive PHC, through the identical pipeline, produces 0.0% airborne frames.

Against the pre-fix `finetuned_backbone`@440k, every metric improves, but by 1.2–2.3x, not
uniformly 2–2.3x: success 2.33x (0.317→0.738), MPJPE 2.12x, Jitter 2.13x, MPJRE 1.68x, MPJVE
1.46x, Foot Skate 1.21x. *(Corrected 2026-11-23; the earlier text said "every metric improves
2–2.3x".)*

Against the SimXR no-vision ablation, `trackerr` is better on `phc_survival`, MPJPE, MPJRE, MPJVE
and Jitter. It is worse on Foot Skate, `frame_head` success (0.989 vs 0.997) and post-sim Mesh
GPen (6.03 vs 3.30 mm). *(Corrected; the earlier text said "every metric except foot skate".)*

Against the Naive PHC baseline, `trackerr` is better on MPJPE (87.4 vs 96.4 mm), Foot Skate and
`frame_head` success. It is worse on `phc_survival` (0.738 vs 0.767), MPJRE, MPJVE, Jitter and
Mesh GPen. Distillation is roughly at parity with skipping distillation, with a mixed metric
profile rather than a dominant one. Both comparisons are subject to the test-set selection caveat
above.

Per-activity `phc_survival`:

| Activity | `finetuned_backbone` (old) | **`trackerr` (new)** | Naive baseline |
|---|---|---|---|
| walking | 0.000 | **0.812** | 0.792 |
| jogging | 0.000 | **0.390** | 0.779 |
| interactions | 0.000 | **0.573** | 0.533 |
| dancing | 0.000 | **0.250** | 0.500 |
| standing | 0.689 | **0.985** | 0.985 |
| whiteboard | 0.647 | **0.968** | 0.974 |
| gestures | 0.314 | **0.893** | 0.830 |
| ROM | 0.600 | **0.800** | 0.762 |

Locomotion/dynamic aggregate: **236/461 = 0.512, from 0/461 = 0.000**. `martialarts` (0/9) and
`sports` (5/45) remain near-zero and are the obvious next target. Naive baseline is not shown for
these two — not separately broken out in that run.

> **Provenance.** Phase A uses the **AMASS-finetuned** standalone AvatarPoser
> (`results/baselines/avatarposer_quest2_phase_a_finetuned.json`), because every other column
> descends from that backbone and the Naive PHC baseline tracks its raw output. The from-scratch
> variant (MPJPE 40.24, MPJRE 10.91) appears in the Phase A comparison table further down.
> *(Audit note: the Phase A metrics JSON records `models/15000_G.pth`, but
> `scripts/mesh_gpen_dataset.py` computes the Phase A Mesh GPen stream from `models/15001_G.pth`.
> Both checkpoints were lost with the original workstation, so whether they hold the same weights
> cannot be verified.)*
> `Finetuned Backbone` is `trackerr` @380k (`student_0380160.pth`); the superseded pre-fix
> `finetuned_backbone` @440k run is retained only in the historical table below.
> Phase B (frozen teacher on ground truth) was previously a column here; it is removed from the
> comparison tables, and its post-sim penetration is kept in the physics-refinement section as
> reference evidence rather than as a compared method.

**Training dynamics**: learns fast then plateaus at ~130k steps, oscillating 0.58–0.74 for the
remaining 370k (best checkpoint, 380k, beats the final one, 500k @ 0.707). Adjacent checkpoints
can differ by up to ±0.10 success — larger than several gaps this project previously treated as
findings. Single-checkpoint comparisons at this scale aren't meaningful without a full-range scan.

### Stage R on `trackerr`: catastrophic collapse, root cause found, not yet fixed

Seeded from the 0.738 Stage E checkpoint, 30k Stage R (PPO) steps: monotonic collapse from the
first checkpoint (+480 steps → 0.394, +2k → 0.190, +4k → 0.040, +12k → 0.001, +30k → 0.006 —
"best" is just the least-damaged point; nothing ever beat the seed).

**Root cause**: Stage E's `_train_on_batch` never trains `value_head`, so Stage R's PPO starts
with a random critic (loss ~79 → ~6 only by batch ~287). Early GAE advantages are computed from a
near-random value function, so the first PPO updates push the actor in ~random directions before
the critic becomes usable.

**Critic warm-up experiment** (`quest_stager_warmup`): pre-trained `value_head` alone (λ=1
Monte-Carlo returns, actor/backbone frozen and verified bit-identical afterward; reached EV 0.809
from 2.9 pre-warmup) before enabling PPO. Post-warmup, pre-PPO eval reproduced the seed exactly
(0.7379/87.43mm/15.23°). Result: the immediate collapse is ~2/3 smaller (−0.119 vs. −0.344 at the
first checkpoint) but **still collapses** (0.619 → 0.360 → 0.181 → 0.086 → 0.007 → 0.022 by step
4416) — warm-up helps but isn't sufficient alone. Two unaddressed suspects: (1) the KL-adaptive LR
controller *raises* `stage_r_lr` when KL is low (5e-6 → 1.69e-5 observed) — exactly while the
policy is correctly barely moving — then KL runs away later (0.003 → 0.25, ~8x threshold); (2)
critic loss backprops into the shared `state_enc` trunk (only `z` is detached, not `h`) at
`critic_coef=5`, and critic loss itself explodes 35 → 216 as the policy drifts. `reward_mean`
decays monotonically alongside, confirming real degradation, not an eval artifact.

**Conclusion: don't use Stage R on Quest2 yet.** `trackerr`'s Stage E checkpoint (@380k) stands as
the headline result. Next steps if revisited: pin the LR (disable `kl_adaptive`) to isolate
suspect 1, then detach `h` for the critic path or lower `critic_coef` for suspect 2.

Operational note: live periodic eval died silently at 130k in this run (Isaac Gym OOM from the
documented env-stepping leak, training's own allocation growing 0.4→7.2GB) and kept reporting a
stale best — on a 16GB card, trust the post-hoc full-range scan past ~130k, not live tracking.

### SimXR ablation: like-for-like, and what the remaining MPJRE gap means

SimXR's no-vision quest run (`env_simxr_quest.yaml`: `distill: True`, `only_kin_loss: True`,
`models: ['.../phc_prim_quest/Humanoid.pth']`) is a student behavior-cloned from the **exact same
frozen teacher checkpoint** this repo distills from, stepping the sim with its own action — same
teacher, same objective, same student-driven rollout, same 0.5m termination threshold. So its
71.7% vs. this pipeline's numbers is a like-for-like comparison, not apples-to-oranges.

The structural difference is observation space: SimXR's policy consumes
`compute_imitation_observations_v6` over `trackBodies: ["b_head"]` — i.e. exactly the head
tracking-error signal `use_tracking_error` adds here. This *confirms* the root-cause diagnosis
above rather than being a separate mystery — and SimXR gets *less* reference information than we
do (head only vs. head + 2 wrists) and still locomotes, further evidence the deficit was
closed-loop feedback, not input richness.

The MPJRE gap is still real: SimXR (29.76°) is not better than `trackerr` (15.23°) despite SimXR's
higher success/lower MPJPE on the old pre-fix checkpoints it was originally compared against.
Plausible explanation: SimXR never directly supervises per-joint rotation, whereas
`distill_agent_quest.py` adds an explicit `kinematic_loss` on rotations — consistent with our
rotation error being comparatively good while (pre-fix) everything else wasn't.

### AMASS-pretrained backbone, fine-tuned on Quest2

Warm-starting the Quest2 backbone from `model_zoo/avatarposer.pth` (transferring everything except
`joint_rotation_decoder`'s final layer, 126-dim SMPL vs. 144 for quest_humanoid) instead of
training from scratch improved every kinematic metric:

| Metric | From-scratch | Fine-tuned from AMASS |
|---|---|---|
| rot_error_deg | 10.91 | 10.65 |
| pos_error_cm | 4.02 | 3.79 |
| vel_error_cm_s | 15.47 | 14.50 |
| jitter_m_s3 | 136.95 | 136.46 |
| foot_skate_cm_s | 29.17 | 23.34 |
| Pre-sim gpen mean (mm) | 12.82 | 11.61 |
| Pre-sim gpen rate | 0.541 | 0.489 |
| eg_mpjpe_mm | 40.24 | 37.85 |

Gains are modest (2–20%, little headroom left above the AMASS/SMPL baseline of 4.05°/2.51cm), but
every metric moving the same way is signal that the AMASS encoder transfers useful structure
across skeletons. This is why `trackerr` uses this checkpoint. (Its downstream physics effect
pre-`trackerr`, 31.7% vs. `Option2`'s 29.4%, is most likely noise — see Lessons below.)

`results/baselines/avatarposer_quest2_phase_a_finetuned.json`,
`results/AvatarPoseEstimationQuest2Finetune/models/15000_G.pth`.

### Naive baseline — alignment gotcha

Recipe: `convert_avatarposer_quest2_to_phc.py` runs the fine-tuned backbone over all 1072 test
sequences → frozen `phc_prim_quest` teacher tracks those predicted pkls → scored against ground
truth (`scripts/score_naive_baseline_vs_gt.py`, `results/baselines/quest2_naive_baseline_vs_gt.json`).

> **Alignment matters and is easy to get wrong.** Predicted pkls start `window_size+1` frames into
> each clip, and PHC's own dump lags one further frame. Scoring at offset 0 reports **568mm
> MPJPE / 0.108 success** — pure temporal misalignment (a walker covers ~1.5m in 42 frames), not
> tracking error. The correct offset, **42**, is derived, not tuned: 41 from the prediction
> pipeline, +1 for PHC's dump lag (established independently on Phase B, which has no windowing at
> all and still aligns best at offset 1). Anything scoring a PHC dump against source pkls needs
> this.

### Dataset caveats (apply to every Quest2 number above)

> **Test split is subject-overlapping, not subject-independent (verified 2026-08-04).**
> Train/test file separation is correctly enforced (4,097 train / 1,072 test clips, zero
> overlapping filenames), but the split is by (subject, activity) take, not by subject — 13 of 23
> test subjects also appear in train with disjoint activities. **326/1072 = 30.4%** of the test set
> is "seen subject, unseen activity"; only 69.6% is a genuine unseen-person test. This is SimXR's
> own dataset convention (confirmed in their CVPR'24 paper, Sec. 4: an explicitly *random* 8:2
> split — table counts match ours exactly, 4097/1072), not something introduced here. Doesn't
> affect internal comparisons (same split throughout) or the SimXR comparison (same overlap exists
> in their own raw released dirs) — but state numbers as clip-level/subject-dependent if reporting
> externally.
>
> **Metric definitions differ slightly from the paper.** SimXR reports over **11 selected common
> joints**; this repo's `evaluate_phc_metrics_quest.py` uses **all 25**. Their success is
> per-sequence (any single frame over threshold fails the whole sequence — closest to our
> `phc_survival`, not `frame_head`). Treat comparisons against numbers quoted from their tables as
> approximate.

### Lessons (still-relevant methodology, independent of the root-cause fix)

- **Early stopping doesn't transfer from the SMPL/Stage-R pipeline.** `success_rate` is noisy and
  non-monotonic across the *entire* range here (old `Option2` oscillated ~0.23–0.29 from 100k to
  500k; `trackerr` plateaus similarly). Patience-based early stopping would have stopped both runs
  hundreds of thousands of steps early. Always use a full-range scan or live periodic eval.
- **Frozen vs. unfrozen backbone matters (~8–10pp), but small deltas between unfrozen runs may
  not.** A frozen-backbone probe (matched 50k budget) put `Option2` *ahead* of `finetuned_backbone`
  by 2.1pp — opposite direction from the 2.3pp gap this project had briefly treated as a finding,
  same magnitude, both within `Option2`'s own oscillation band. The probe's one clean result:
  frozen backbones land at ~20–22% vs. ~29–32% unfrozen. Reproduce:
  `bash scripts/run_frozen_backbone_probe.sh [max_distill_step] [num_envs]`.
- **Which pose representation feeds the action head is not the bottleneck.** `explicit_pose`
  (`q_base`) vs. `Option2` (`z`): no measurable difference (29.3% vs. 29.4%), echoing the SMPL
  pipeline's `v8` finding. The real bottleneck was the *missing* feedback signal, not the encoding.

<details>
<summary><b>Historical: pre-fix superseded runs (Option2/3, explicit_pose, finetuned_backbone, Stage R trials 1–2)</b> — click to expand</summary>

**All columns below were trained without `use_tracking_error` and are superseded by `trackerr`.**
Kept as the full historical record and for the per-checkpoint detail (Stage R trial diagnostics,
step-normalization vs. SimXR, etc.) that doesn't fit the current-results table above.

| Metric | Phase A | SimXR ablation⁶ | `Option3`@149.8k³ | `Option2`@399.8k³ | `explicit_pose`@490.08k⁸ | `finetuned_backbone`@440k⁹ | `_stager` trial 1@443.6k¹¹ | `_stager_v2` trial 2@442.08k¹² |
|---|---|---|---|---|---|---|---|---|
| Success (`phc_survival`) | 1.000 | 0.717 | 0.241 | 0.294 | 0.293 | 0.317 | **0.383** | 0.326 |
| Success (`frame_head`) | — | 0.997 | 0.974 | 0.976 | 0.972 | 0.980 | 0.967 | 0.976 |
| MPJPE (mm) | 40.24 | 112.5 | 177.67 | 185.87 | 189.14 | 185.60 | 192.04 | 190.01 |
| MPJRE (°) | 10.91 | 29.76 | 25.03 | 26.35 | 29.27 | 25.65 | **39.11** | 37.01 |
| MPJVE (cm/s) | 15.47 | 40.93 | 50.33 | 52.91 | 64.29 | 49.94 | 57.66 | 56.13 |
| Jitter (m/s³) | 136.95 | 704.29 | 1232.05 | 1193.38 | 1102.75 | 1117.46 | 834.67 | 969.71 |
| Foot Skate (cm/s) | 29.17 | 14.39 | 20.57 | 21.75 | 22.83 | 20.75 | 20.14 | 18.73 |
| Pre-sim gpen mean (mm) | 12.82 | N/A¹⁴ | 15.32 | 13.55 | 13.54 | 13.65 | —¹⁵ | —¹⁵ |
| Pre-sim gpen rate | 0.541 | N/A¹⁴ | 0.022 | 0.427 | 0.411 | 0.420 | —¹⁵ | —¹⁵ |

> **Pre-sim gpen is measured on the kinematic/reference motion before physics simulation.**
> Post-sim penetration is analysed separately because Isaac Gym contact generally drives it
> close to zero and it is not discriminative in the main comparison.
>
> Previously this row mixed definitions: Phase A's cell was pre-sim (12.82 mm) while every other
> cell was post-sim (~0.00x mm). Values now come from the pre-sim table below, one convention
> throughout.

> ¹⁴ **SimXR** has no intermediate pose representation by design, so pre-sim gpen is undefined.
>
> ¹⁵ **Stage R trials** run PPO on top of `finetuned_backbone`, which updates the backbone, so their
> `q_base` differs from it; pre-sim gpen is measurable but was never computed. Reported as — rather
> than copying `finetuned_backbone`'s value.
>
> ³ Best checkpoints found via full-range `quest_checkpoint_scan.sh` (patience disabled), not the
> final checkpoint — early stopping is unreliable for this recipe (see Lessons above).
>
> ⁶ SimXR's own no-vision ablation, replicated in the separate/upstream SimXR repo
> (`env=env_simxr_quest`, `learning=im_simxr`, vision disabled via `+env.pretrain=True`, same
> dataset split). Needed an adaptive/KL-based LR schedule to avoid a collapse the default constant
> schedule hit (`ablation_results.md` has the full trail). Peak checkpoint, epoch 1,500 of 6,104;
> final checkpoint settles to 51.4%/116.5mm. Normalized step count (SimXR's `frame` sums across
> all 64 parallel envs, this repo's counter is already per-env): 24,576,000/64 = 384,000
> per-env-equivalent steps — comparable in scale to `Option3`'s 149,760, `Option2`'s 399,840,
> `explicit_pose`'s 490,080, not a dramatically different budget. See "SimXR ablation" discussion
> above for what the numbers mean.
>
> ⁸ `explicit_pose`: same as `Option2` but `action_head` conditioned on `q_base` instead of `z` —
> no measurable difference (see Lessons above).
>
> ⁹ `finetuned_backbone`: same recipe as `Option2`, only `avatarposer_checkpoint` swapped to the
> AMASS-then-Quest2-finetuned backbone (see the fine-tuning subsection above).
>
> ¹¹ **UNVERIFIED — not reproducibility-checked.** First Stage R trial, seeded from
> `finetuned_backbone`'s best checkpoint, 10k-step budget, scanned every 1,000 steps. Highest
> single point in an otherwise wildly oscillating trajectory (0.3%→6%→13%→**38.3%**→33%→10%→6%→
> 11%→31%→24%, crashing near-zero twice within the window) — traded rotation accuracy for survival
> (MPJRE 38–48° vs. the 25.65° seed).
>
> ¹² **Second Stage R trial, with a real bug fixed.** `_collect_stage_r_batch`/
> `_train_stage_r_on_batch` never called `.eval()`, so the backbone's transformer computed
> `old_logp`/`logp` under two *different* random dropout masks — contaminating the PPO ratio/KL
> with dropout noise indistinguishable from real policy drift (the SMPL pipeline's Stage R already
> avoided this; missed when porting to Quest2). Fixed, plus retuned hyperparameters
> (`stage_r_lr` 2e-5→5e-6, `kl_threshold` 0.1→0.03, `entropy_coef` 0.01→0.003,
> `lambda_kinematic_stage_r` 0.1→0.3). Result: best 0.326 @442,080 (barely above the 0.317 seed),
> MPJRE still 37.01°, trajectory still crashed to 0.2% later before partially recovering.
> **Conclusion: the fix corrected the numerics but eliminated neither the instability nor the
> rotation trade-off** — pointing at the reward/training dynamics themselves, not the numerical
> bug, as the primary cause. (Directly informed the `trackerr` Stage R investigation above, which
> found the actual root cause: an untrained critic.)

**Pre-sim (backbone-only) kinematic quality**, computed retroactively by extracting each
checkpoint's `backbone` sub-dict and running it through the Phase-A eval script — valid because
HMD input features are always derived from the ground-truth reference, never the simulator's own
state. Uses the per-sequence empirical ground reference (Phase A's convention), not `GROUND_Z=0`.
This is the source of the **Pre-sim gpen** rows in both tables above; all pre-sim gpen figures in
this document share this one convention.

| Model | rot_error_deg | pos_error_cm | Pre-sim gpen mean (mm) | Pre-sim gpen rate |
|---|---|---|---|---|
| Phase A (from-scratch) | 10.91 | 4.02 | 12.82 | 0.541 |
| Phase A (AMASS-finetuned) | 10.65 | 3.79 | 11.61 | 0.489 |
| `Option2` | 14.34 | 5.34 | 13.55 | 0.427 |
| `explicit_pose` | 14.29 | 5.32 | 13.54 | 0.411 |
| `finetuned_backbone` | 14.27 | 5.28 | 13.65 | 0.420 |
| `Option3` (pure action-loss)¹⁰ | **90.90** | **47.03** | 15.32 | 0.022 |

> ¹⁰ Not a real improvement despite the lowest pre-sim gpen rate — `Option3` trains with
> `lambda_kinematic=0.0`, so its pose reconstruction is badly broken (90.9° rotation error vs.
> ~14.3° for every kinematically-supervised variant); its low pre-sim gpen rate reflects an incoherent
> pose that happens not to dip below the floor much, not cleaner motion.

Superseded findings, compressed: locomotion was exactly 0.000 for `finetuned_backbone` (walking
0/144, jogging 0/136, interactions 0/75, dancing 0/20, martialarts+stretching+retail+workout
0/41) — now explained and fixed, see the root-cause section above. Near-zero gpen on these old
checkpoints was a symptom of standing still rather than attempting the reference, not safer
motion (that remark refers to **post-sim** penetration; see the physics-refinement note below). Joint kinematic+action training degraded the backbone (~10.9°→~14.3°, ~4.0cm→~5.3cm) for
all three pre-fix variants despite `lambda_kinematic=1.0` dominating — not revisited post-fix.
SimXR's original replication (pre-fix comparison) handled locomotion (walking 0.755, jogging
0.298) where every local variant scored 0.000 — this is what pointed at the root cause, though
SimXR's dump is a train-split eval so its absolute numbers are optimistically biased.

</details>

### Post-sim ground penetration (physics refinement) — not in the main comparison

Post-sim penetration is deliberately kept out of the tables above. Isaac Gym contact drives it
essentially to zero for **every** Quest2 method, so it does not discriminate between them and,
placed in the same column as pre-sim figures, it produced a genuinely mixed metric (Phase A's cell
was pre-sim, every other cell post-sim).

Qualitatively, on the 1072-clip test set every Quest2 row lands in **0.000–0.015 mm mean with rate
≤ 0.001**. Recomputing Phase B directly from its retained dump gives **0.00533 mm, with any
penetration at all in 1 of 1072 sequences**. Preserved for the record (all joint-position,
`GROUND_Z = 0`, i.e. a *different* convention from the pre-sim rows and not comparable to them):

| Method | Post-sim gpen mean (mm) | Post-sim gpen rate |
|---|---|---|
| Phase B (teacher) | 0.005 | 0.0001 |
| `Option3` | 0.003 | 0.000 |
| `Option2` | 0.015 | 0.001 |
| `explicit_pose` | 0.006 | 0.000 |
| `finetuned_backbone` @440k | 0.003 | 0.000 |
| `trackerr` @380k | 0.008 | — |
| `_stager` trial 1 / trial 2 | 0.006 / 0.000 | 0.000 / 0.000 |
| SimXR no-vision ablation | 0.00 | 0.000 |
| Naive baseline | not measured¹⁶ | not measured¹⁶ |

> ¹⁶ Post-sim gpen is reference-free (it needs only the simulated trajectory and a floor), so it is
> computable for the naive baseline in principle. The rollout dump it would be computed from
> (`output/HumanoidIm/phc_prim_quest/phc_positions_for_eval.pkl`) was overwritten on 2026-08-20 by a
> single-clip run reusing the same `exp_name`; only the Phase B copy in
> `results/baselines/phase_b_artifacts_backup/` survives. Recomputing needs a fresh 1072-sequence
> rollout, and given the range above would near-certainly return 0.00x mm.
>
> **Unresolved inconsistency (audit 2026-11-23):** `scripts/run_mesh_gpen_study.sh` reads the
> Naive stream from this same path, yet `scripts/dataset_mesh_gpen_aggregate.json` (committed
> 2026-09-10) reports a 1,072-clip Naive post-sim Mesh GPen (2.56 mm / 0.020). Either the dump was
> regenerated after 2026-08-20 and this footnote is stale, or the Mesh GPen value came from
> elsewhere. The dump itself was lost with the original workstation, so this **cannot now be
> resolved**. Both statements are kept as recorded.

A genuine pre-vs-post penetration analysis — where the metric *is* informative, because it is
measured against a motion with real artefacts rather than after physics has removed them — lives in
the SMPL-side physics-refinement work: `scripts/quest_smpl_step6e_report.md` (per-clip pre/post) and
`scripts/quest_smpl_charstudy_report.md` (10-clip characterization, e.g. sports 24.5 → 0.0 mm).

### Per-clip metrics for the rendered clips

Everything above is averaged over all 1072 test clips. For the individual clips that were
rendered to video (one per motion category: planted / full-body-transition / locomotion), with a
root-drift vs. root-relative-pose decomposition, see **`results_per_clip_renders.md`**
(`scripts/per_clip_metrics.py`). Videos in `output/renderings/`.

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
# be set high (e.g. 999) to effectively disable early-stop -- see Lessons, it's unreliable for
# this recipe. extra_hydra_overrides is REQUIRED for any run not trained with every
# learning.params.config default (e.g. explicit_pose_action_input=True):
bash scripts/quest_checkpoint_scan.sh <exp_name> [eval_interval] [patience] [start_step] \
     [extra_hydra_overrides]

# Resuming an INTERRUPTED scan: quest_checkpoint_scan.sh does NOT know it was killed partway --
# re-running with the same start_step blindly redoes every point from the beginning. To resume,
# pass start_step = the step that was mid-eval when killed. Example (2026-07-31, killed mid-eval
# at step 442140 out of a 440640-445184 range):
nohup bash scripts/quest_checkpoint_scan.sh quest_distill_finetuned_backbone_stager_v2 250 999 442140 \
  "learning.params.config.enable_stage_r=True" \
  > phc/output/HumanoidIm/quest_distill_finetuned_backbone_stager_v2/scan_resume.log 2>&1 &
disown
# CAVEAT: the resumed invocation's own "best so far" tracking restarts at -1/0 -- its final
# summary line only reflects the RESUMED range. Reconcile the true best manually across all
# scan_eval_<step>.log files from both invocations.

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
