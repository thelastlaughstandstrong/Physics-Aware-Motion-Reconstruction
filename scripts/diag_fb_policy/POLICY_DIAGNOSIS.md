# Why does the Finetuned Backbone have a heavy ankle-motion tail?

Diagnosis only — nothing retrained, no loss/reward/simulator/metric change.
Measurements: `SUBJECT086.walking_2889_3300` (412 frames, a high-GPen lower-body clip), student
control and teacher `mu` logged on **identical observations** via a runtime patch that modifies no
production file (`log_actions.py`).

## Step 2 — the distillation objective (code, unambiguous)

`distill_agent_quest.py:848` — `action_loss = mse_loss(control, teacher_action)`,
`total_loss = 1.0 * kinematic_loss + 0.1 * action_loss`.

| question | finding |
|---|---|
| action imitation loss | plain MSE over the full 72-D vector |
| all 72 dims weighted equally? | **yes** — no per-DOF weighting |
| ankle weighted like knee/hip? | **yes**, identical |
| temporal smoothness / action-delta penalty | **none anywhere in the codebase** |
| contact-state or lower-body weighting | **none** |
| direct or indirect supervision | direct, against teacher actions |
| teacher actions clipped before supervision? | **no** — `_query_teacher` returns raw `mus`, while the teacher *acts* through `clamp(mu,−1,1)` |
| relative weight of action imitation | **0.1** vs kinematic 1.0 |

## Step 3 — saturation: hypothesis refuted by measurement

| group | student >0.8 | >0.9 | >0.95 | teacher >0.8 | teacher >1.0 | teacher max |
|---|---|---|---|---|---|---|
| hip | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.58 |
| knee | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.64 |
| ankle | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.81 |

Whole vector: student max |a| = **0.784**; teacher |mu|>1 on **0.01%** of dims, max 1.14.

Neither network operates near its bound. The raw-vs-clamped supervision mismatch and the tanh
vanishing-gradient concern are therefore **real in code but negligible in practice** — they touch
0.01% of dimensions. *(Flagged because it was a plausible mechanism; measurement rules it out.)*

## Step 1 — where the tail originates

Per-group statistics, student output **before simulation**:

| group | \|a\| mean | p99 | \|Δa\| p95 | \|Δa\| p99 | student−teacher err | err p95 |
|---|---|---|---|---|---|---|
| hip | 0.059 | 0.222 | 0.098 | 0.196 | 0.085 | 0.220 |
| knee | 0.064 | 0.334 | 0.084 | 0.165 | 0.062 | 0.183 |
| **ankle** | **0.087** | **0.454** | **0.215** | **0.460** | **0.126** | **0.390** |

The heavy ankle tail **already exists in the network output**, ~2.6× the knee's Δ-tail at p95.
So it originates at **(A) network output**, not in post-processing, PD application, or contact.

### But it is inherited, not created

| group | student \|Δa\| p95 | teacher \|Δa\| p95 | ratio | student p99 | teacher p99 | ratio |
|---|---|---|---|---|---|---|
| hip | 0.098 | 0.114 | **0.86** | 0.196 | 0.194 | 1.01 |
| knee | 0.084 | 0.102 | **0.82** | 0.165 | 0.189 | **0.87** |
| ankle | 0.215 | 0.209 | **1.03** | 0.460 | 0.410 | 1.12 |

**The PHC teacher's own ankle Δ-tail is 2.05× its own knee's** (0.209 vs 0.102). The student
reproduces it within **3%** at p95, and is *smoother* than the teacher at hip (0.86×) and knee
(0.82×). The ankle-heavy tail is a property of the teacher policy that distillation faithfully
transmits — the student is not manufacturing it.

### Residual imitation gap, ankle-specific

Student−teacher error is 0.126 at the ankle vs 0.062 at the knee (**2.03×** absolute). Normalising
by each group's action magnitude (ankle 0.087 vs knee 0.064, 1.36×) leaves **~1.5× worse relative
imitation at the ankle**. Error is **not** concentrated at high-activity frames: top-10% |Δa| frames
carry 11.7% of the error mass (uniform = 10%), ratio 1.16×.

## Step 4 — classification: **D — Mixed**, with contributions quantified

| candidate | evidence | contribution |
|---|---|---|
| **A. Distillation mismatch** | ankle imitation ~1.5× worse than knee after magnitude normalisation — but Δ-tail reproduced within 3%, so no tail inflation | **secondary** |
| **B. Loss design** | no temporal-smoothness or action-delta term anywhere; action loss weighted 0.1 vs kinematic 1.0 — nothing corrects the residual ankle gap | **enabling, not causal** |
| **C. Representation limitation** | error only 1.16× higher on high-activity frames; 11.7% vs 10% error mass — sparse-input ambiguity is not where the error lives | **weak** |
| **Inherited teacher behaviour** *(not in A–C)* | teacher's own ankle Δ-tail is 2.05× its knee; student matches it at 1.03× | **dominant** |

**The dominant factor is not on the list A–C: the ankle-heavy tail is intrinsic to the PHC teacher
policy and is faithfully inherited.** Distillation adds a modest (~1.5× relative) ankle imitation
gap on top, which the loss design does nothing to correct because it contains no smoothness term
and weights action imitation at 0.1.

### Why FB's *realised* ankle motion still exceeds Naive's

Given the same state, student and teacher command comparably jumpy ankle actions (1.03×). The
1.40× gap in realised ankle angular velocity p95 therefore comes largely from the two systems
visiting different states: **FB tracks ground truth, Naive tracks AvatarPoser's predictions**, which
float +13.3 mm higher and are smoother. That is an evaluation-design difference established in the
control audit, not a policy defect.

## Caveats

Single clip (412 frames, one walking sequence). The teacher `mu` is a **counterfactual** — what the
teacher would command in the states the student created — not the teacher's own rollout; that is the
correct comparison for "does the student imitate well", but it is not the teacher's own trajectory.
A multi-clip repeat would firm up the Δ-tail ratios.

## Implication for any future fix (not applied)

The natural intervention — an action-delta / temporal-smoothness penalty on ankle DOFs — would make
the student **deviate from the teacher**, since the teacher is itself ankle-jumpy. It would trade
imitation fidelity for smoothness. That is a design decision, not a bug fix, and should be taken
deliberately.
