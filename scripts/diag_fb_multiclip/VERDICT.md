
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
