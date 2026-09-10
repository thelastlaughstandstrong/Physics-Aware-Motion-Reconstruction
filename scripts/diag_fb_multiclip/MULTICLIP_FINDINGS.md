# Multi-clip validation: is the ankle action-delta tail inherited from the PHC teacher?

Diagnostic only — nothing retrained, no loss/reward/simulator/metric/retargeting change.
**49 clips** from the Quest2 test set where BOTH the Finetuned Backbone (`trackerr`@380k) and the Naive PHC baseline survive to the end of the clip, plus the single-clip anchor from the earlier diagnostic (reported separately, never pooled).
*6 selected clip(s) terminated early in this batched run despite surviving the full-test run and were excluded from the aggregate (`SUBJECT088.interactions.pkl_1329_1771`, `SUBJECT061.stretching.pkl_3947_4384`, `SUBJECT078.ROM.pkl_3233_3636`, `SUBJECT080.jogging.pkl_430_859`, `SUBJECT074.standing.pkl_433_865`, `SUBJECT075.standing.pkl_1645_2055`); PHC appends post-reset frames to a terminated episode, which would inject a spurious |Δa| jump.*

**Three action streams, kept separate throughout:**

| stream | policy | motions | states visited |
|---|---|---|---|
| `student` | FB `trackerr`@380k | ground-truth Quest | its own |
| `teacher*` (counterfactual) | frozen PHC teacher | ground-truth Quest | **the student's** — mu queried on the same obs |
| `naive` (own trajectory) | frozen PHC teacher, same weights | AvatarPoser predictions | **its own**, 42 frames shorter |

`teacher*` answers *does the student imitate the teacher's tail*. `naive` answers *what does this policy do when it drives itself*. They are different states and are never pooled.

## Aggregate by motion category

Median across clips (IQR in brackets). `|Δa| p95`, per joint group.

### hip

| category | n | student p95 | teacher* p95 | student/teacher | naive p95 |
|---|---|---|---|---|---|
| upper-body | 15 | 0.052 [0.031–0.116] | 0.055 [0.031–0.138] | **0.91** [0.81–0.97] | 0.049 [0.032–0.067] |
| lower-body | 14 | 0.117 [0.091–0.152] | 0.138 [0.119–0.168] | **0.83** [0.78–0.89] | 0.086 [0.074–0.107] |
| full-body/dynamic | 12 | 0.090 [0.032–0.122] | 0.115 [0.047–0.169] | **0.74** [0.65–0.82] | 0.070 [0.039–0.085] |
| other | 8 | 0.034 [0.021–0.041] | 0.040 [0.030–0.051] | **0.85** [0.68–0.98] | 0.024 [0.021–0.037] |
| **all** | 49 | 0.085 | 0.109 | **0.82** | |

### knee

| category | n | student p95 | teacher* p95 | student/teacher | naive p95 |
|---|---|---|---|---|---|
| upper-body | 15 | 0.057 [0.028–0.118] | 0.060 [0.031–0.143] | **0.87** [0.82–1.02] | 0.054 [0.032–0.071] |
| lower-body | 14 | 0.132 [0.107–0.151] | 0.155 [0.133–0.179] | **0.83** [0.79–0.89] | 0.109 [0.093–0.121] |
| full-body/dynamic | 12 | 0.093 [0.035–0.135] | 0.119 [0.048–0.189] | **0.76** [0.68–0.86] | 0.085 [0.029–0.106] |
| other | 8 | 0.040 [0.023–0.058] | 0.038 [0.029–0.047] | **0.95** [0.74–1.11] | 0.030 [0.019–0.034] |
| **all** | 49 | 0.097 | 0.115 | **0.84** | |

### ankle

| category | n | student p95 | teacher* p95 | student/teacher | naive p95 |
|---|---|---|---|---|---|
| upper-body | 15 | 0.124 [0.062–0.286] | 0.143 [0.078–0.272] | **0.94** [0.82–1.02] | 0.129 [0.079–0.186] |
| lower-body | 14 | 0.284 [0.225–0.337] | 0.299 [0.242–0.343] | **0.94** [0.91–1.02] | 0.221 [0.201–0.288] |
| full-body/dynamic | 12 | 0.213 [0.069–0.298] | 0.199 [0.091–0.335] | **0.83** [0.74–0.95] | 0.162 [0.064–0.261] |
| other | 8 | 0.067 [0.048–0.130] | 0.084 [0.064–0.107] | **0.88** [0.79–0.98] | 0.065 [0.041–0.076] |
| **all** | 49 | 0.208 | 0.220 | **0.91** | |

## Q1 — does the teacher itself have a heavier ankle tail than knee/hip?

Teacher* ankle/knee and ankle/hip |Δa| p95 ratio, per clip.

| category | n | teacher ankle/knee | >1 in | student ankle/knee | naive ankle/knee |
|---|---|---|---|---|---|
| upper-body | 15 | **2.19** [2.03–2.42] | 100% | 2.18 | 2.51 |
| lower-body | 14 | **1.87** [1.68–2.04] | 100% | 2.07 | 2.21 |
| full-body/dynamic | 12 | **1.84** [1.67–2.01] | 92% | 2.04 | 2.19 |
| other | 8 | **2.18** [1.97–2.27] | 100% | 2.36 | 2.52 |
| ALL | 49 | **2.00** [1.76–2.23] | 98% | 2.14 | 2.39 |

Teacher ankle |Δa| p95 exceeds its own knee on **48/49 clips** (exact two-sided sign test p = 1.78e-13).

## The two teacher streams, side by side (kept separate on purpose)

Same frozen weights. `teacher*` is evaluated in the **student's** states on ground-truth motions; `naive` drives **itself** on AvatarPoser predictions. If the ankle-heavy tail were an artefact of the states FB visits, `naive` would not show it.

| category | n | teacher* ankle/knee | naive ankle/knee | teacher* ankle p95 | naive ankle p95 |
|---|---|---|---|---|---|
| upper-body | 15 | 2.19 | **2.51** | 0.143 | 0.129 |
| lower-body | 14 | 1.87 | **2.21** | 0.299 | 0.221 |
| full-body/dynamic | 11 | 1.86 | **2.19** | 0.260 | 0.162 |
| other | 8 | 2.18 | **2.52** | 0.084 | 0.065 |
| ALL | 48 | 2.02 | **2.39** | 0.229 | 0.176 |

Naive's own ankle |Δa| p95 exceeds its own knee on **48/48 clips** (sign test p = 7.11e-15).

## Q2 — does the student reproduce the teacher's ankle tail?

student/teacher* |Δa| ratio at each percentile (median across clips, all categories).

| group | p50 | p90 | p95 | p99 |
|---|---|---|---|---|
| hip | 0.78 | 0.79 | 0.82 | 0.82 |
| knee | 0.69 | 0.81 | 0.84 | 0.91 |
| ankle | 0.80 | 0.90 | 0.91 | 0.90 |

Student ankle p95 within ±15% of the teacher on **31/49 clips** (63%).

## Q5 — is the student's extra ankle imitation error large enough to matter?

| category | n | ankle err | knee err | ankle/knee err | ankle/knee after magnitude-normalising |
|---|---|---|---|---|---|
| upper-body | 15 | 0.1194 | 0.0593 | 1.93 | 0.87 |
| lower-body | 14 | 0.1414 | 0.0732 | 1.80 | 1.19 |
| full-body/dynamic | 12 | 0.1399 | 0.0906 | 1.52 | 0.90 |
| other | 8 | 0.1148 | 0.0593 | 1.96 | 0.91 |
| ALL | 49 | 0.1335 | 0.0705 | 1.82 | 1.03 |

## Q4 — do larger ankle tails go with higher realised ankle velocity, jitter, foot skate?

Spearman rank correlation across clips, with a 2000-sample permutation p-value.

| x | y | rho | p | n |
|---|---|---|---|---|
| student ankle |Δa| p95 | realised ankle angvel p95 | +0.94 ** | 0.000 | 49 |
| student ankle |Δa| p95 | jitter | +0.90 ** | 0.000 | 49 |
| student ankle |Δa| p95 | foot skate | +0.88 ** | 0.000 | 49 |
| student ankle |Δa| p95 | contact transitions/s | +0.84 ** | 0.000 | 49 |
| teacher* ankle |Δa| p95 | realised ankle angvel p95 | +0.92 ** | 0.000 | 49 |
| teacher* ankle |Δa| p95 | jitter | +0.88 ** | 0.000 | 49 |
| teacher* ankle |Δa| p95 | foot skate | +0.88 ** | 0.000 | 49 |
| teacher* ankle |Δa| p95 | contact transitions/s | +0.83 ** | 0.000 | 49 |
| student/teacher ankle ratio | realised ankle angvel p95 | +0.54 ** | 0.000 | 49 |
| student/teacher ankle ratio | jitter | +0.51 ** | 0.000 | 49 |
| student/teacher ankle ratio | foot skate | +0.43 ** | 0.001 | 49 |
| student/teacher ankle ratio | contact transitions/s | +0.49 ** | 0.001 | 49 |
| ankle imitation error | realised ankle angvel p95 | +0.73 ** | 0.000 | 49 |
| ankle imitation error | jitter | +0.70 ** | 0.000 | 49 |
| ankle imitation error | foot skate | +0.74 ** | 0.000 | 49 |
| ankle imitation error | contact transitions/s | +0.63 ** | 0.000 | 49 |

`**` = p < 0.05 (uncorrected; 16 tests, so treat isolated marks with caution).

The raw rows above are **confounded by motion intensity** — a vigorous clip has both a bigger action tail and faster realised motion, for the student and the teacher alike (note the teacher’s rows are nearly as strong as the student’s). The question that actually bears on a student-side intervention is whether the student’s tail predicts realised instability *beyond what the teacher’s own tail already predicts*:

| x (controlling for teacher* ankle p95) | y | partial rho | p | n |
|---|---|---|---|---|
| student ankle |Δa| p95 | realised ankle angvel p95 | +0.51 ** | 0.000 | 49 |
| student ankle |Δa| p95 | jitter | +0.40 ** | 0.005 | 49 |
| student ankle |Δa| p95 | foot skate | +0.20 | 0.194 | 49 |
| student ankle |Δa| p95 | contact transitions/s | +0.32 ** | 0.026 | 49 |

## Distribution across clips (not just the mean)

student/teacher* ankle |Δa| p95 ratio:

| min | p10 | p25 | median | p75 | p90 | max |
|---|---|---|---|---|---|---|
| 0.47 | 0.74 | 0.82 | 0.91 | 1.01 | 1.08 | 1.63 |

teacher* ankle/knee |Δa| p95 ratio:

| min | p10 | p25 | median | p75 | p90 | max |
|---|---|---|---|---|---|---|
| 0.83 | 1.65 | 1.76 | 2.00 | 2.23 | 2.49 | 3.03 |

## Representative clips

| | clip | category | teacher ankle/knee | student ankle/knee | s/t ankle p95 | realised ankle angvel p95 | skate |
|---|---|---|---|---|---|---|---|
| lowest teacher ankle/knee | `SUBJECT034.workout.pkl_2030_2435` | full-body/dynamic | 0.83 | 0.93 | 0.94 | 433°/s | 1.3 |
|  | `SUBJECT034.workout.pkl_3651_4055` | full-body/dynamic | 1.56 | 2.50 | 1.32 | 281°/s | 5.3 |
| median | `SUBJECT081.whiteboard.pkl_0_416` | upper-body | 2.00 | 2.44 | 1.03 | 323°/s | 12.5 |
|  | `SUBJECT088.jogging.pkl_1341_1787` | lower-body | 2.03 | 2.42 | 0.98 | 479°/s | 10.4 |
| highest teacher ankle/knee | `SUBJECT079.gestures.pkl_1676_2094` | upper-body | 2.83 | 2.13 | 0.83 | 64°/s | 3.0 |
|  | `SUBJECT008.drinks.pkl_2384_2780` | upper-body | 3.03 | 2.05 | 0.76 | 94°/s | 5.0 |

### Anchor clip (earlier single-clip diagnostic, reported separately)

`SUBJECT086.walking.pkl_2889_3300` — teacher ankle/knee **2.04**, student/teacher ankle p95 **0.97**. Published single-clip values were 2.05 and 1.03.

> Note: FB **terminates early** on this clip in the full test run, so it was not a representative choice — it is retained only as the pipeline's reproducibility check.


---

# Verdict: **A — inherited teacher ankle behaviour generalises strongly**

with a small, measurable student-side residual that does **not** rise to option D.

## Answers to the five questions

**1. Does the PHC teacher consistently have a heavier ankle Δ-tail than knee/hip?**
**Yes, essentially without exception.** Teacher* ankle |Δa| p95 exceeds its own knee on **48/49
clips** (median ratio **2.00**, IQR 1.76–2.23; sign test p = 1.8e-13). The naive stream — the same
weights driving *themselves* on different motions — shows it on **48/48 clips** at an even higher
**2.39** (p = 7.1e-15). All 6 clips excluded for early termination also exceed 1, so the exclusion
does not manufacture the effect. The single-clip value of 2.05 was representative.

**2. Does the Finetuned Backbone reproduce it?**
**Yes, slightly attenuated.** Median student/teacher* ankle |Δa| p95 = **0.91**, within ±15% on
31/49 clips. The important comparison is against the other groups: the student attenuates the hip
to **0.82** and the knee to **0.84**, but the ankle only to **0.91** — the ankle is where the
student smooths *least*. Its own ankle/knee asymmetry (**2.14**) is therefore marginally *sharper*
than the teacher's (2.00), not because it adds ankle motion but because it removes less of it there.

**3. Lower-body only, or everywhere?**
**Everywhere — and it is not a lower-body phenomenon at all.** The teacher's ankle/knee ratio is
*highest* in upper-body (2.19) and standing (2.18), lowest in full-body/dynamic (1.84). What varies
by category is the absolute magnitude (lower-body ankle p95 0.284 vs standing 0.067), not the
asymmetry. Even in near-static standing clips the policy commands ~2.2× more ankle action churn
than knee. This is a property of the control policy, not of locomotion.

**4. Do larger ankle tails go with higher realised ankle velocity, jitter, foot skate?**
**Yes raw, but mostly as a motion-intensity confound.** Raw Spearman is very strong (student ankle
p95 vs realised ankle angular velocity **+0.94**), but the *teacher's* tail correlates nearly as
strongly (+0.92) — both largely restate "vigorous clips are vigorous". Controlling for the
teacher's own tail, the student's residual still predicts realised ankle angular velocity
(**+0.51**, p<0.001), jitter (**+0.40**, p=0.005) and contact transitions (+0.32, p=0.026), but
**not foot skate** (+0.20, p=0.194, n.s.). So there is a real but secondary student-specific
component, and it does not reach the metric (foot skate) that most directly reflects contact quality.

**5. Is the student's extra ankle imitation error large enough to matter?**
**No — and this corrects the single-clip diagnostic.** Raw ankle/knee imitation-error ratio is 1.82,
but the ankle also carries proportionally larger actions; after magnitude normalisation the ratio is
**1.03** across 49 clips — no ankle-specific imitation penalty. The earlier report's "~1.5× worse
relative imitation at the ankle" was a single-clip artefact. Only lower-body shows any residual
(1.19); upper-body (0.87), dynamic (0.90) and standing (0.91) are at or below parity.

## Why not the other options

| | |
|---|---|
| **B** (does not generalise) | Rejected — 48/49 and 48/48 clips, p < 1e-12 in both streams. |
| **C** (category-dependent) | Rejected — present in all four categories; *strongest* in the least dynamic ones. Only the magnitude is category-dependent, not the effect. |
| **D** (student adds substantial instability) | Rejected as the *dominant* account. The student's ankle tail is **smaller** than the teacher's (0.208 vs 0.220), its normalised ankle imitation error is 1.03, and it is smoother than the teacher at every joint group. The partial correlations show a genuine residual, but it is secondary to an effect the teacher exhibits on ~100% of clips. |

## Fairness note

The two teacher streams were never pooled. `teacher*` is the frozen teacher's mu on the *student's*
observations (counterfactual, ground-truth motions); `naive` is the same weights driving themselves
on AvatarPoser predictions (own trajectory, 42 frames shorter, its own failure set). They agree on
the direction and significance of the ankle asymmetry while differing in magnitude (2.02 vs 2.39),
which is itself informative: the asymmetry survives a complete change of state distribution.

---

# Is this enough to justify testing an ankle-smoothness intervention?

**Not as a defect fix — the evidence argues against that framing.**

The measurements remove the premise such an intervention would rest on. There is **no excess student
ankle tail to remove**: across 49 clips the student's ankle |Δa| p95 is already **0.91×** the
teacher's, and it is smoother than the teacher at every joint group. An action-delta or
ankle-smoothness penalty would not be correcting a distillation artefact; it would push the student
*further away* from a teacher it currently under-shoots, trading imitation fidelity for smoothness
against a policy that is ankle-jumpy by design on ~100% of clips including near-static standing.

Two further points argue the same way:

- The student's residual (partial rho +0.51 ankle angular velocity, +0.40 jitter) is real but does
  **not** reach foot skate (p = 0.194) — the metric most tied to contact quality, and the one
  motivating the GPen work that started this thread.
- FB's higher realised jitter versus naive (524.78 vs 221.94 m/s³ in `results_comparison.md`) is
  substantially an evaluation-design difference — FB tracks ground truth, naive tracks AvatarPoser's
  smoother predictions — established in the earlier control audit, not a policy defect.

**What the evidence would support instead**, if smoother ankles are the goal:

1. Treat it as a deliberate **fidelity-vs-smoothness trade**, stated as such, not a bug fix — and
   measure the cost in tracking accuracy, not only the benefit in smoothness.
2. Target the **teacher**, or the reference/state distribution, since that is where the ankle
   asymmetry demonstrably originates and where ~100% of the effect lives.
3. If a student-side penalty is still wanted, scope it to the measured residual (an ankle-weighted
   action-delta term) and pre-register the expected effect size — the residual is modest, so an
   underpowered A/B would not resolve it.

**Recommendation: do not run the intervention as a fix.** If you want it as an experiment, run it as
an explicit trade-off study with tracking accuracy as a co-primary metric. Awaiting your decision —
nothing has been changed.

## Caveats

- 49 clips of 1072; stratified but not exhaustive. Standing is n=8 after exclusions.
- 6 selected clips terminated early in this batched run despite surviving the full-test run; FB's
  per-clip survival is **not deterministic across runs** (the anchor clip shows the reverse — it
  fails the full test but survived here). Per-clip ratios carry run-to-run noise of roughly ±0.06
  (anchor: 1.03 published, 0.97 here).
- Realised kinematics come from the existing full-test FB dump, not from this batched run, so they
  pair with a rollout of the same policy on the same clip but not the same execution.
- 16 uncorrected correlation tests; the partial correlations are the ones to trust, and foot skate
  is not significant among them.
