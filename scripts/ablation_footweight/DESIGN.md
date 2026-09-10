# Foot-weighted kinematic loss — experimental design

> **Amended by [PROTOCOL.md](PROTOCOL.md), which is frozen and supersedes this document wherever they differ.** In particular: the mean-normalised weighting scales non-foot rows DOWN (7.4% at w=2.0, 19.4% at w=4.0) rather than leaving them unchanged; the thresholds below are practical noise guardrails, not significance tests; and the checkpoint-selection rule is fixed there.

---

## Step 1 — Current kinematic loss, as it actually is

**File:** `phc/phc/learning/distill_agent_quest.py`
**Function:** `DistillAgentQuest._train_stage_e_on_batch`, lines **843–849**
(an identical block exists in `_train_stage_r_on_batch`, lines 1043–1046; Stage R is unused —
`trackerr` is a Stage E checkpoint and Stage R collapses.)

```python
z = self.student.backbone(flat["hmd"][idx])                       # 838
q_base_root, q_base_joints = self.student.backbone.decode(z)      # 839
...
pred_pos = _quest_fk_module(q_base_root, q_base_joints, self._quest_body_model)   # 841

root_loss      = nn.functional.l1_loss(q_base_root,   flat["gt_root_6d"][idx])    # 843
joint_rot_loss = nn.functional.l1_loss(q_base_joints, flat["gt_joints_6d"][idx])  # 844
joint_pos_loss = nn.functional.l1_loss(pred_pos,      flat["gt_pos"][idx])        # 845
kinematic_loss = self._kinematic_root_weight * root_loss + joint_rot_loss + joint_pos_loss
action_loss    = nn.functional.mse_loss(control, flat["teacher_action"][idx])
total_loss     = self._lambda_kinematic * kinematic_loss + self._lambda_action * action_loss
```

with `kinematic_root_weight = 0.02`, `lambda_kinematic = 1.0`, `lambda_action = 0.1`
(`distill_quest.yaml:100–104`). Formula verified against the training log:
`0.02·0.049008 + 0.035077 + 0.014127 = 0.050184` — exactly the logged `kinematic 0.050184`.

**Which joints each term covers**

| term | tensor | shape | covers | foot indices |
|---|---|---|---|---|
| `joint_pos_loss` | `pred_pos` = `SkeletonState.global_translation` | `[B, 25, 3]` | **all 25 bodies**, root-relative | **21** = `b_l_foot_twist`, **24** = `b_r_foot_twist` |
| `joint_rot_loss` | `q_base_joints` | `[B, 144]` = 24×6D | **24 non-root joints** (dof order) | dof **20 / 23** → 6D slices `[120:126]`, `[138:144]` |
| `root_loss` | `q_base_root` | `[B, 6]` | root orientation only | — |

Indices verified numerically from `SkeletonTree.from_mjcf(quest_humanoid.xml).node_names`.
The quest rig has **no toe joint** — the ankle (`*_foot_twist`) is the leaf of the leg chain.

**Current joint weighting: uniform.** `l1_loss` reduces with `mean` over every element, so each of
the 25 bodies × 3 coords carries 1/75 of the position term and each of the 144 rotation dims carries
1/144. There is **no per-joint weighting of any kind**.

**Root-specific weighting already exists** — but only on root *orientation*
(`_kinematic_root_weight = 0.02`), not on any position. Note that body 0's position is identically
zero on both sides (`gt_pos` is root-relative; FK runs with `trans=None`), so 3 of the 75 position
elements are structurally zero and merely dilute the mean.

**Stage: pre-simulation.** All three terms compare the kinematic backbone's predicted pose
`q_base` against the recorded reference pose. `pred_pos` is forward kinematics on predicted
rotations, not simulator output. The sim was already stepped in Pass 1; this `control` "never
touches the sim" (module docstring, line 92).

**Does the kinematic loss reach the actions at all?** Yes — first-order. With
`explicit_pose_action_input: False` and `freeze_backbone: False` (both the `trackerr` settings),
line 832 is `z_for_action = z.detach() if self._freeze_backbone else z` — so **`z` is not
detached**, and `feat = cat(z, h, track_err)` feeds `action_head`. `kinematic_loss` back-propagates
through `decode(z)` into the same `z` the action head consumes. The intervention is therefore not
structurally incapable of changing behaviour, which was the main thing worth checking before
proposing it.

---

## Step 2 — Smallest valid modification

One tensor multiply on the position term only. Rotation, root, action, architecture, observations,
simulator, teacher, reward, retargeting, termination and metrics are all untouched.

```python
# --- add near the other loss weights in __init__ (~line 459) -------------------
self._foot_pos_weight = float(cfg.get("foot_pos_weight", 1.0))
# body indices in SkeletonTree.node_names order == pred_pos/gt_pos index order
self._foot_body_ids = [21, 24]          # b_l_foot_twist, b_r_foot_twist (quest has no toe)

# --- replace line 845 ----------------------------------------------------------
if self._foot_pos_weight == 1.0:
    joint_pos_loss = nn.functional.l1_loss(pred_pos, flat["gt_pos"][idx])
else:
    W = torch.ones(pred_pos.shape[1], device=pred_pos.device, dtype=pred_pos.dtype)
    W[self._foot_body_ids] = self._foot_pos_weight
    W = W * (pred_pos.shape[1] / W.sum())     # mean-normalise: total loss scale unchanged
    joint_pos_loss = (W[None, :, None] * (pred_pos - flat["gt_pos"][idx]).abs()).mean()
```

**Why mean-normalised.** Without it, raising the foot weight also raises the magnitude of
`joint_pos_loss`, which changes the effective learning rate on the kinematic term — the ablation
would then confound "where the loss looks" with "how hard it pulls". With normalisation
`sum(W) = 25` always, so the arms differ only in the *distribution* of emphasis.

**Verified:** at `foot_pos_weight = 1.0` the weighted expression returns a value **bitwise
identical** to `nn.functional.l1_loss` (delta exactly 0.00e+00 on a random batch), so the baseline
arm is reproduced exactly and the branch is a no-op by default.

### Proposed weights — and why not 1.5

The relevant question is how much of the objective the feet actually command. At convergence
(step ≈380k, from the training log):

| term | value | share of total objective |
|---|---|---|
| `joint_rot_loss` | 0.00649 | **71.1%** |
| `joint_pos_loss` | 0.00234 | **25.6%** |
| `0.02 · root_loss` | 0.000158 | 1.7% |
| `0.1 · action_loss` | 0.000145 | 1.6% |

The feet are 2 of 25 bodies, so at baseline they command ≈8% of a term that is itself 25.6% of the
objective — **about 2% of the total**. That makes the sweep the user proposed weaker than it looks:

| foot weight | share of `joint_pos_loss` | share of total objective | change vs baseline |
|---|---|---|---|
| 1.0 (baseline) | 8.0% | 2.0% | — |
| 1.5 | 11.5% | 3.0% | **+1.0 pp** |
| 2.0 | 14.8% | 3.8% | +1.8 pp |
| 4.0 | 25.8% | 6.6% | +4.6 pp |

**Recommendation: sweep 1.0 / 2.0 / 4.0, and drop 1.5.** A +1.0 pp shift in the objective is very
unlikely to produce an effect distinguishable from the checkpoint-to-checkpoint noise measured
below, so a 1.5 arm would most likely return "no significant difference" and cost a full training
run to say nothing. 4.0 is not an arbitrary large weight: even there the feet remain a **minority**
of the position term (25.8%) and 6.6% of the objective — it is the smallest weight that produces a
change of the same order as the noise floor. If 4.0 shows movement and 2.0 does not, that itself
locates the threshold.

`lambda_action` stays at 0.1 and every other setting is identical across arms.

---

## Step 3 — Success criteria, fixed before training

### The baseline must not be checkpoint 380160

Across the **26 late checkpoints (≥250k) of the single existing `trackerr` run** — all on the
plateau, all evaluated on the full test set — the metrics vary this much:

| metric | plateau mean | std | ckpt 380160 (the reported number) | z |
|---|---|---|---|---|
| success (`phc_survival`) | 0.6946 | 0.027 | 0.7379 | +1.60 |
| MPJPE_g (mm) | 90.31 | 1.18 | **87.43** | **−2.45** |
| foot skate (cm/s) | 17.59 | 0.675 | 17.14 | −0.66 |
| jitter (m/s³) | 561.1 | 36.0 | 524.78 | −1.01 |

**The headline FB numbers in `results_comparison.md` come from a checkpoint that is a 2.45σ
favourable outlier on MPJPE within its own run's plateau.** It was chosen by scanning success rate
on the test set (`checkpoint_scan.log`). Comparing a new arm's checkpoints against 87.43 mm would
therefore be scoring it against a cherry-pick and would bias the ablation toward "no improvement".

**Every arm is scored as the mean over its 8 late checkpoints (330k–400k, 10k spacing)**, never a
single best checkpoint. Standard error of an 8-checkpoint mean:

| metric | ±1 s.e. at k=8 |
|---|---|
| success | ±0.010 |
| MPJPE_g | ±0.42 mm |
| foot skate | ±0.24 cm/s |
| jitter | ±12.7 m/s³ |

### Thresholds

**Primary target — post-simulation foot-joint tracking error** (mean ‖pred−ref‖ over bodies 21, 24).
Current FB 11.86 cm vs SimXR 9.35 cm on the shared 38 clips; the gap to close is 2.51 cm.
→ **Success: ≥0.5 cm reduction** vs the baseline arm's 8-checkpoint mean, with non-overlapping
±2 s.e. intervals.

**Secondary — foot skate.** → **Success: ≥0.5 cm/s reduction** (≈2 s.e., and ≈1/5 of the 2.75 cm/s
FB→SimXR gap).

**Guardrails** (any breach = the arm is rejected regardless of foot gains):

| guardrail | tolerance | rationale |
|---|---|---|
| MPJPE_g | must not worsen by >1.2 mm (≈1σ of plateau noise) | preserves most of FB's 25 mm advantage over SimXR |
| MPJRE | must not worsen by >1.0° | FB's 15.23° vs SimXR's 29.76° is the clearest FB win |
| jitter | must not increase by >36 m/s³ (1σ) | SimXR's foot gain came with 2.9× jitter; do not buy the same trade |
| success | must not drop by >0.027 (1σ) | survival comparable |

**Also tracked, not gated:** ankle |Δa| p95 and ankle/knee |Δa| ratio (to confirm the intervention
does *not* work by ankle smoothing — which the previous diagnostic argued against).

**Mesh GPen Mean/Rate: only for the single winning arm**, after selection — it is the most expensive
metric in the project (full-set IK) and is not needed to rank arms.

---

## Step 4 — Smallest informative experiment

### Validation subset: 90 clips, disjoint from the reporting set

Built by `build_valsubset.py` from the 791 test clips the current FB survives:

| category | high-foot-error | clean control | random | mean foot err, high / clean |
|---|---|---|---|---|
| upper-body | 12 | 8 | 5 | 24.7 / 1.8 cm |
| lower-body | 12 | 8 | 0 | 25.2 / 10.8 cm |
| full-body/dynamic | 12 | 8 | 2 | 36.8 / 2.1 cm |
| standing | 12 | 8 | 3 | 18.9 / 3.3 cm |

Subset FB foot error 16.92 cm (range 1.33–66.29) vs 13.37 cm full-set — deliberately enriched for
the failure mode, with clean controls included so a regression on easy clips cannot hide.

**The remaining 701 clips are reserved for the final number and are not looked at during
selection.** This is a departure from previous practice worth stating plainly: every Quest2
checkpoint so far was picked by scanning the full test set, so its reported figure carries a
selection bonus. Keeping the split disjoint here means the winner's final number is honest, but it
also means it will likely look slightly worse than 87.43 mm — that is the bias being removed, not a
regression.

Arms are evaluated under identical conditions: same subset, same checkpoints steps, same eval
config, all metrics from `evaluate_phc_metrics_quest.py`. Selection is by the pre-registered rule
above, applied to all three arms at once — no post-hoc choice of which metric to favour.

---

## Step 5 — Decision rule

**Continue** only if an arm shows: foot-joint error ≥0.5 cm lower **and** no guardrail breached,
ideally with foot skate ≥0.5 cm/s lower.

**Stop and conclude "wrong stage"** if `joint_pos_loss` at the feet falls clearly (the pre-sim
target is being optimised) but post-sim foot error does not improve. Both are logged, so this is
directly checkable rather than inferred. Given the loss is pre-simulation and the coupling to
actions runs only through the shared `z`, this is a live possibility and the most likely null
result.

**Stop** if the only arm that helps is 4.0 *and* it breaches a guardrail — that would say the effect
requires an emphasis shift large enough to cost whole-body accuracy, i.e. the same trade SimXR makes,
which we already have without retraining.

---

## Estimated cost

Measured from the existing run: **≈20k steps/hour** (start 08-11 11:14 → 380160 at 08-12 06:14).

| item | cost |
|---|---|
| Train arm `w=2.0` to 400k | ≈20 h |
| Train arm `w=4.0` to 400k | ≈20 h |
| Baseline `w=1.0` | **0 h — reuse the existing `trackerr` run** (the branch is bitwise identical at w=1.0) |
| Evaluate 8 checkpoints × 3 arms on 90 clips | ≈8 min each → **≈3.2 h** |
| Final full-set eval of the winner (701 held-out clips) | ≈1 h |
| **Total** | **≈44 h**, dominated by two training runs |

Two practical notes:

- **Reusing the baseline controls checkpoint noise but not seed noise.** The existing run gives 26
  plateau checkpoints, so its mean is well estimated, but all share one initialisation. If the
  result lands close to threshold, a fresh `w=1.0` arm (+20 h) is needed before believing it. I would
  run the two new arms first and only add the fresh baseline if the outcome is marginal.
- **GPU contention.** Training peaked at 7.7 GB allocated / 9.9 GB reserved, and an unrelated
  3D-Gaussian-Splatting job is currently holding 8.6 GB of the 16 GB card. The two arms should run
  sequentially, and probably after that job finishes.

---

## What this design does not do

- No architecture, observation, simulator, PHC teacher, action-loss, reward, retargeting,
  termination-criteria or evaluation-metric change.
- No change to `joint_rot_loss` — the ankle *rotation* dofs (20, 23) keep weight 1.0. Foot
  *position* depends on the whole leg chain, so weighting the position term at the foot body
  propagates gradient through hip/knee/ankle together; weighting the ankle rotation instead would
  target foot orientation, which is not the measured deficit.
- No change to Stage R's copy of the loss (lines 1043–1046) — Stage R is not used.
