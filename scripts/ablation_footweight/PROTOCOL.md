# Frozen protocol — foot-weighted kinematic loss ablation

Amends `DESIGN.md` per review. **Written and frozen before any training run started.**
Supersedes `DESIGN.md` wherever the two differ.

---

## 1. What the weighting actually does (corrected)

The weight vector is mean-normalised, `sum(W) == 25`, so `joint_pos_loss` keeps approximately the
same overall scale across arms and the arms differ in *where* the term looks rather than *how hard*
it pulls. That normalisation has a consequence the earlier write-up stated too loosely:

> **Raising the foot weight scales the non-foot rows DOWN.** It redistributes relative position-loss
> weight toward the feet; it does not leave every other joint's effective weight unchanged.

| `foot_pos_weight` | effective weight, feet | effective weight, other 23 bodies | change to non-foot |
|---|---|---|---|
| 1.0 | 1.0000 | 1.0000 | — |
| 2.0 | 1.8519 | 0.9259 | **−7.4%** |
| 4.0 | 3.2258 | 0.8065 | **−19.4%** |

So the w=4.0 arm is not "feet 4× louder, everything else as before" — it is "feet 3.23× and every
other body 0.81×". Any whole-body regression in that arm is an expected consequence of the design,
not an anomaly, and the guardrails are what decide whether it is acceptable.

### Confirmed scope (regression-tested, `test_foot_weight.py`, 12/12 pass)

- `foot_pos_weight = 1.0` is **bitwise identical** to the previous `nn.functional.l1_loss` — verified
  at three batch sizes, exact float equality, and the code takes a literal `if w == 1.0` no-op branch.
- **Only `joint_pos_loss` is changed.** `joint_rot_loss`, `root_loss` (and its 0.02 weight),
  `action_loss` (MSE, `lambda_action = 0.1`), `lambda_kinematic = 1.0`, the architecture,
  observations, simulator, PHC teacher, rewards, retargeting, termination criteria and evaluation
  metrics are all untouched. `git diff` confirms no `-` line on any of those expressions.
- Indices 21/24 verified to be `b_l_foot_twist` / `b_r_foot_twist` in the tensor's own index order;
  the rig has no toe joint.
- Stage R's copy of the loss (lines ~1043–1046) is deliberately left unmodified; Stage R is unused.

### One addition beyond the loss: a pre-simulation diagnostic

Distinguishing Outcome A from Outcome B requires the **pre-sim** foot error, which was not logged.
Added as a detached quantity that never enters the gradient:

```python
jpos_foot = _pos_absdiff[:, self._foot_body_ids, :].mean().detach()
```

logged as `jpos_foot` in the existing `[distill] step …` line. This changes no gradient and no loss
value; without it, Outcome B would not be checkable.

---

## 2. Model-selection protocol (frozen)

**The 90-clip validation subset (`val_clips.json`) is the only data used for any selection.**
The 701 held-out clips are not evaluated, inspected, or summarised until a single final candidate
exists and the user has approved the held-out run.

**Predefined checkpoints — identical for all three arms:**
`330000, 340000, 350000, 360000, 370000, 380000, 390000, 400000` (nearest saved checkpoint to each).

**Per arm, report mean ± SD across those 8 checkpoints** for:
post-sim foot-joint error · Foot Skate · MPJPE · MPJRE · Jitter · success ·
ankle |Δa| p95 · ankle/knee |Δa| ratio.
Reporting the SD alongside the mean is what shows whether an effect is robust across checkpoints
rather than one lucky checkpoint.

**Choosing the winning arm** — compare 8-checkpoint means against the w=1.0 arm's 8-checkpoint mean
(never against checkpoint 380160, which is a 2.45σ favourable outlier within its own run).

**Choosing the single final checkpoint from the winning arm** (the user's rule, adopted as stated):

1. discard any checkpoint breaching a guardrail (MPJPE, MPJRE, jitter, success) on validation;
2. among survivors, take the **lowest validation post-sim foot-joint error**;
3. tie-break on **lowest validation Foot Skate**;
4. if still tied, take the **earliest** checkpoint step (deterministic, and prefers the less-trained
   model).

I am adopting this rather than proposing an alternative — it is deterministic, validation-only, and
ordered so the primary target decides. One property to record now rather than discover later: step 2
picks a minimum over 8 noisy values, so **the selected checkpoint's validation foot error is
optimistically biased** by roughly the checkpoint spread. That bias is exactly what the untouched
701-clip held-out set exists to remove, and it is why the validation number must not be quoted as
the result.

---

## 3. Status of the thresholds

These are **practical guardrails derived from observed within-run checkpoint variability**, not
formal statistical significance tests. They come from 26 plateau checkpoints (≥250k) of the single
existing `trackerr` run — one run, one seed — so they describe checkpoint-to-checkpoint scatter, not
seed-to-seed scatter, which is unmeasured and probably larger.

| quantity | threshold | source |
|---|---|---|
| foot-joint error improvement | ≥ 0.5 cm | target; gap to SimXR is 2.51 cm |
| Foot Skate improvement | ≥ 0.5 cm/s | target; gap to SimXR is 2.75 cm/s |
| MPJPE guardrail | ≤ +1.2 mm | ≈1 SD (1.18) |
| MPJRE guardrail | ≤ +1.0° | judgement; MPJRE was not logged per-checkpoint |
| Jitter guardrail | ≤ +36 m/s³ | ≈1 SD (36.0) |
| success guardrail | ≥ −0.027 | ≈1 SD (0.027) |

**If the winner is marginal relative to this scatter, no improvement will be claimed.** The
recommendation in that case is a fresh-seed replication before any conclusion — the baseline arm
reuses the existing run, so a marginal result is confounded with seed variance that this design
cannot separate.

---

## 4. Execution

- Wait until the unrelated 3D-Gaussian-Splatting job releases GPU memory (training peaks at
  7.7 GB allocated / 9.9 GB reserved on a 16 GB card).
- Train `foot_pos_weight=2.0`, then `foot_pos_weight=4.0`, **sequentially**.
- Same seed, config, and training duration as the baseline run wherever reproducible; the only
  changed value is `foot_pos_weight`.
- Baseline `w=1.0` reuses the existing `trackerr` run (bitwise-identical code path), re-scored on
  the 90 validation clips.
- Evaluate only on the 90-clip validation set during development.

---

## 5. Decision rule

| outcome | signature | action |
|---|---|---|
| **A** | pre-sim foot error ↓ **and** post-sim foot error/skate ↓ | foot weighting propagates through the controller — proceed to held-out evaluation (with approval) |
| **B** | pre-sim foot error ↓ but post-sim unchanged | kinematic weighting targets the wrong stage — **stop; do not raise the weight further** |
| **C** | foot behaviour ↑ but MPJPE/MPJRE materially worsen | explicit foot-vs-whole-body trade — report as a trade, do not present as a win |
| **D** | no meaningful effect anywhere | reject the intervention |

`jpos_foot` in the training log gives the pre-sim half directly, so A and B are separated by
measurement rather than inference.

**After selecting the single winner and checkpoint from validation, STOP and report validation
results. The 701-clip held-out evaluation will not be run without explicit approval.**

---

## Addendum — measured validation-set precision (recorded before any arm result exists)

The baseline arm has now been scored on the 90 validation clips (8 checkpoints, 330k–400k). Its
scatter differs from the full-test scatter the thresholds in §3 were derived from, so the
thresholds' detection power is recorded here **before** either new arm finishes, not afterwards.

| metric | baseline mean | SD | SE of an 8-ckpt mean | threshold | threshold / SE |
|---|---|---|---|---|---|
| foot-joint error (cm) | 16.86 | 0.392 | 0.138 | 0.5 target | **3.6** |
| foot skate (cm/s) | 14.43 | 0.594 | 0.210 | 0.5 target | 2.4 |
| MPJPE (mm) | 85.42 | 2.209 | 0.781 | 1.2 guardrail | **1.5 — tightest** |
| MPJRE (deg) | 15.54 | 0.407 | 0.144 | 1.0 guardrail | 6.9 |
| jitter | 317.7 | 29.35 | 10.38 | 36 guardrail | 3.5 |
| success | 0.7750 | 0.032 | 0.012 | 0.027 guardrail | 2.4 |

**The experiment is adequately powered for its primary question**: a 0.5 cm foot-error improvement
is 3.6 SE of the arm mean.

**The MPJPE guardrail is the weak point.** MPJPE scatters more on 90 clips (SD 2.21) than on the
full test set (SD 1.18), so +1.2 mm is only ~1.5 SE and will sometimes trip on noise alone. It is
kept unchanged — the protocol is frozen — but a breach of the MPJPE guardrail *alone*, by a margin
under ~1.6 mm (2 SE), will be reported as inconclusive on that guardrail rather than as a
established regression, and would be a reason to check MPJPE on the held-out set before concluding.

Note also that the validation subset was built only from clips the baseline **survives**, so its
absolute success (0.775) sits above the full-test figure (0.738) by construction. Success on this
subset can fall but has little room to rise; it is usable as a regression guardrail, not as a
headline number, and is not comparable to any full-test success figure.
