# Results Comparison

## PHC Physics Evaluation — Primary Results

Full student (backbone + action head) in IsaacGym physics simulation on PHC test set (536 sequences).
Results saved to `phc/output/HumanoidIm/<EXP>/phc_eval.log`.

| Model | Checkpoint | Success | MPJPE (cm) | qref_MPJPE (cm) | MPJVE (cm/s) | MPJRE (°) | Jitter (m/s³) | Foot Skate (cm/s) | gpen_mean (mm) | gpen_rate |
|---|---|---|---|---|---|---|---|---|---|---|
| `frozen_bb` — frozen backbone | 489.5k | 0.584 | 34.83 | — | 89.39 | 38.73 | 1188 | 2.66 | — | — |
| `v2` — trainable heads, z.detach() | 1.95M | 0.455 | 50.71 | — | 113.59 | 45.02 | 1612 | 2.28 | — | — |
| `v3` — refinement head, no staged training | 431k | 0.440 | 47.18 | — | 105.24 | 43.07 | 1418 | 2.63 | — | — |
| `v4` — staged training 1→2→3→E | 750k | 0.498 | 41.14 | 61.40 | 99.59 | 40.86 | 1345 | 2.51 | 0.04 | 0.003 |
| `v5` — position-space L_ref (no isolation) | 500k | 0.465 | 46.12 | 21.57 | 107.89 | 43.04 | 1513 | 2.61 | 0.74 | 0.167 |
| `v5` — position-space L_ref (no isolation) | 750k | 0.461 | 47.06 | 22.02 | 109.56 | 43.40 | 1544 | 2.81 | 0.74 | 0.157 |
| `v6` — full gradient isolation + angle-bounded Δq | 500k | 0.476 | 43.60 | 61.36 | 104.47 | 42.44 | 1438 | 2.48 | 0.05 | 0.003 |
| `v6` — full gradient isolation + angle-bounded Δq | 750k | 0.470 | 44.56 | 61.23 | 104.38 | 42.30 | 1462 | 2.65 | 0.09 | 0.003 |
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
> v4 was the best of the old (pre-v7) refinement-branch line, peaking at 750k.
> v5: qref_MPJPE improved dramatically (61→22 cm) but physics degraded and gpen spiked (16.7%) — L_ref conflicted with encoder training.
> v6: full isolation protected physics (gpen clean, success closer to v4) but refinement head produces near-zero corrections — qref_MPJPE unchanged from v4.
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

Same idea as the SMPL section above, but for the `quest_humanoid` rig (SimXR's 25-joint skeleton,
`has_upright_start: False`) and the real SimXR Quest2 synthetic dataset
(`data/quest2_synthetic/`, 1072-sequence held-out test split, `test_seg_motion`). Two
"success rate" metrics are reported: `phc_survival` (physics sim never terminates the episode —
the harder, headline number) and `frame_head` (a looser per-frame head-tracking criterion,
consistently ~0.97-0.99 across every distilled checkpoint below, i.e. not a differentiating
metric here).

| Metric | Phase A (kinematic, no physics)¹ | Phase B (teacher)² | **SimXR reference ablation, no-vision (upstream repo)**⁶ | **Option3 `actionloss` @149.8k (best via periodic eval)**³ | **Option2 `scratch` @399.8k (best via periodic eval)**³ | **`explicit_pose` @490.08k (best via full-range scan)**⁸ |
|---|---|---|---|---|---|---|
| Success (`phc_survival`) | 1.000 | 0.999 | **0.717**⁶ | 0.241 | 0.294 | 0.293 |
| Success (`frame_head`) | — | 0.9999 | 0.997⁷ | 0.974 | 0.976 | 0.972 |
| MPJPE (mm)⁵ | 40.24 | 21.81 | **112.5**⁶ | 177.67 | 185.87 | 189.14 |
| MPJRE (°) | 10.91 | 4.52 | 29.76⁷ | 25.03 | 26.35 | 29.27 |
| MPJPE (cm)⁵ | 4.02 | 2.18 | **11.25**⁶ | 17.77 | 18.59 | 18.91 |
| MPJVE (cm/s) | 15.47 | 9.29 | 40.93⁷ | 50.33 | 52.91 | 64.29 |
| Jitter (m/s³) | 136.95 | 59.97 | 704.29⁷ | 1232.05 | 1193.38 | 1102.75 |
| Foot Skate (cm/s) | 29.17 | 5.90 | 14.39⁷ | 20.57 | 21.75 | 22.83 |
| gpen_mean (mm)⁴ | 12.82 | 0.005 | 0.00⁷ | 0.003 | 0.015 | 0.006 |
| gpen_rate⁴ | 0.541 | 0.0001 | 0.000⁷ | 0.000 | 0.001 | 0.000 |

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
> `ablation_results.md` for the full investigation trail. See "SimXR's upstream ablation..." in
> Results Discussion below for what the results actually mean.
>
> ⁸ See "Explicit-pose conditioning vs. the opaque latent" in Results Discussion below.

### Results Discussion

**Early-stopping doesn't transfer from the SMPL/Stage-R pipeline.** The SMPL pipeline's Stage R
collapse-detection heuristic (patience=2, assumes an early-peak-then-monotonic-collapse
trajectory) does not apply to `quest_distill`'s recipe. Retroactively scanning `option3_actionloss`/
`option2_scratch` with that same patience would have stopped both runs at a temporary early dip
(hundreds of thousands of steps before their real best checkpoints), because `success_rate` is
genuinely noisy/non-monotonic across the *entire* training range for this recipe, not just early
on (`option2_scratch` oscillates ~0.23–0.29 from step 100k all the way to 500k with no discernible
trend). Conclusion: any `quest_distill` variant needs either a full-range retroactive scan
(patience disabled) or live periodic eval throughout training — never an early-stopping heuristic
carried over from the SMPL/Stage-R pipeline.

**Explicit-pose conditioning vs. the opaque latent: no measurable difference.** Conditioning
`action_head`/`value_head` on the backbone's explicitly decoded pose (`q_base`) instead of the
opaque transformer latent `z` (`explicit_pose_action_input=True`) produced a best success rate
(29.3%) statistically indistinguishable from the z-conditioned `Option2` recipe's own best
(29.4%) — not a win, not a loss. This is informative precisely because it isn't dramatic: combined
with the SMPL pipeline's earlier `v8` result (concatenating `q_base` *alongside* `z` was also a
no-op, since `q_base = decode(z)` is a deterministic function `action_head` already had implicit
access to), these two experiments together triangulate to a real conclusion — **using the decoder
layer's explicit output essentially does not differ from using the raw latent representation `z`
for control purposes.** Whatever capacity `z` has beyond what's needed to reconstruct the explicit
pose isn't being exploited for control in either direction: adding it on top of `q_base` changed
nothing (`v8`), and removing it in favor of the explicit decode doesn't help either (this result).
The implication is that the bottleneck keeping this whole distillation family well behind Phase
B's physics-tracking-ground-truth ceiling is *not* which pose representation feeds the action
head — it's more likely the fundamental difficulty of compressing full-body control into a small
network that only ever sees a sparse 3-point HMD signal, echoing the SMPL pipeline's own
"naive baseline beats every distilled variant" finding.

**gpen stays near-zero regardless of success rate — likely a static-sequence bias, not safer
motion.** Every `quest_distill` checkpoint evaluated post-fix (footnote 4) shows near-zero gpen,
including `explicit_pose`'s *best* checkpoint (0.006mm) despite its 29.3% success rate. This is
the same signature seen at this same run's very first (11.85%-success) checkpoint, where direct
inspection of the recorded rollout confirmed the character was essentially standing still rather
than attempting the reference motion at all. Since success rate climbed to 29.3% without gpen ever
becoming meaningfully nonzero, the open (unconfirmed) question is whether these checkpoints
succeed mainly on this test set's more static sequences (standing/gestures/yoga-type activities)
while continuing to fail dynamic ones (walking/jogging/dancing) without literally falling, rather
than genuinely learning safer or more careful dynamic control. Resolving this needs a
per-activity-type success breakdown or watching actual rollouts — the aggregate success/gpen
numbers alone can't distinguish the two explanations.

**SimXR's upstream ablation: much better success/position, but not rotation accuracy.**
Replicating SimXR's own no-vision ablation in the upstream repo scores far higher on success
(71.7% vs. 24–29%) and position error (112.5mm vs. 178–189mm) than every `quest_distill` variant
tried here — but `MPJRE` (rotation error) is *not* better (29.76° there vs. 25–29° here), if
anything slightly worse. A pure bug in a shared code path would be expected to move every metric
in the same direction; this split (success/position much better, rotation not) is more consistent
with a genuine algorithmic difference: SimXR's `im_simxr` trains with full PPO + an AMP
discriminator + a head-tracking task reward, which directly rewards keeping the head on-target
and motion looking natural, but never directly supervises exact per-joint rotation against the
reference the way `distill_agent_quest.py`'s pure action-regression loss does. Not confirmed — a
direct side-by-side of the two training code paths would be needed to rule out a remaining port
bug — but the MPJRE split is real, non-obvious evidence pointing toward "different training
objective," not just "bug."

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
