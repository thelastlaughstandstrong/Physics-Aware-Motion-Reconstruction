# Diagnosis — why Finetuned Backbone has higher mesh GPen than Naive PHC

Evidence in this directory. The full-dataset results were **not** modified.

## Step 1 — pipeline validity: no inconsistency found

| check | result |
|---|---|
| Coordinate frame / Z-up / units | identical (native Z-up, metres) |
| Floor | Z = 0 for every stream |
| SMPL-H model / betas | neutral, betas = 0, 6890 verts, hardcoded identically |
| IK procedure | same `batched_ik`, same objective/optimizer/settings |
| GPen tolerance / definition | 1e-4 m, unchanged import |
| Frame alignment | irrelevant — GPen is per-frame and reference-free |
| Vertical convention | dump **reference** min-Z over first 30 frames = **87.10 mm**, exactly the loader's 0.0871 m target; GT self-calibration applies the same rule to the same signal |

The one structural asymmetry (pre-sim self-calibrates, post-sim inherits the loader's calibration)
is **not** a confound: both apply the identical rule and the measurement confirms the reference
lands on target. **Decisively**, Naive PHC traverses the *same dump path* as Finetuned Backbone and
scores 0.00 mm on clips where FB scores 26–59 mm, so the FB/Naive gap cannot be a path artifact.

## Step 2 — clip selection (measured, not cherry-picked)

280 lower-body clips: **177 worsen**, 12 improve, mean delta **+6.58 mm** (q_base → post).
Top-15 worsening and 6 controls selected purely by measured delta (`selected_clips.json`).

## Step 3 — vertical geometry (17 clips, medians)

| stream | pelvis Z | foot Z | foot p5 | root→ankle |
|---|---|---|---|---|
| Ground Truth | 0.933 | 0.103 | 0.087 | — |
| fb_pre (q_base) | 0.945 | 0.109 | 0.083 | — |
| **fb_post** | **0.915** | 0.082 | **0.070** | 0.828 |
| naive | 0.922 | 0.085 | **0.070** | 0.834 |

Physics lowers FB by ~30 mm from its own kinematic input — but **fb_post and naive reach the
identical foot p5 of 0.070 m**. "FB puts its feet lower" is false at the rigid-body level.

## Step 4/5 — the decisive frame-level measurement

`SUBJECT086.walking_0_412` (FB 5.77 mm, Naive 0.00 mm), 6 penetrating frames of 412:

| quantity | penetrating | clean | ratio |
|---|---|---|---|
| rigid-body foot Z | 0.071 | 0.078 | 0.91 |
| mesh sole Z | **−0.010** | 0.056 | — |
| **joint → sole offset** | **81.4 mm** | **22.3 mm** | **3.64×** |
| max foot pitch | 42.7° | 33.3° | 1.28× |
| **foot vertical speed** | **0.128 m/s** | 0.036 | **3.53×** |
| IK residual | 34.65 mm | 34.25 mm | **1.01** |

Penetrating frames form **2 isolated runs** (4 and 2 frames), **100% during contact on this clip**.
Median joint→sole offset is the SAME for both methods on this clip (FB 22.7 mm, Naive 23.2 mm);
only the tail differs (p95: FB 73.4 mm vs Naive 60.9 mm). Median foot pitch is the same (24.9° vs 23.9°).

> **CORRECTION — these two figures do NOT generalise.** A 17-clip trace (`trace_finetuned_backbone.json`)
> gives a median joint→sole offset of **40.2 mm** (p95 **103.7 mm**), not ~23 mm, and only **63.6%** of
> penetrating frames occur during contact, not 100%. The ~23 mm / 100%-contact figures are specific to
> `walking_0_412`. The single-clip numbers below remain valid for that clip; the scale-level numbers
> above supersede them for any general claim.

### The correction strengthens the mechanism

Across the 17 clips, **a median 30.8% of FB's penetrating frames occur while the foot is airborne**
(rigid-body foot above the 87.1 mm contact height; per-clip range 0–100%). Penetration rate in
flight is 0.0446 vs 0.0585 in contact — only 1.31× apart.

**A foot that is physically off the ground cannot penetrate the floor.** On roughly a third of
penetrating frames the penetration is therefore produced *entirely by the mesh reconstruction*, with
no possible contribution from the simulated motion. That is direct, unambiguous evidence for the
unconstrained ankle→toe DOF as the amplifier — stronger than the contact-timing argument it
replaces, and it means the effect is not confined to stance.

**Root cause of the amplification:** the Quest rig's leg chain ends at `b_*_foot_twist` (→ SMPL
Ankle). There is **no toe target**, so the SMPL ankle→toe segment is unconstrained by the IK and its
pitch is set only by the regulariser. During rapid foot transitions that free segment swings
downward, driving the sole through the floor while the rigid body sits at a normal height. The
unchanged IK residual proves this is a faithful fit, not a failed one.

## Step 6 — dataset-scale confirmation (280 lower-body clips, no IK)

| driver (median) | FB | Naive | ratio |
|---|---|---|---|
| median foot speed | 0.097 m/s | 0.074 | 1.31× |
| p95 foot speed | 0.585 | 0.482 | 1.21× |
| frac frames > 0.10 m/s | 0.489 | 0.412 | 1.19× |
| contact fraction | 0.690 | 0.740 | 0.93× |
| pelvis height | 0.917 m | 0.924 | −7 mm |

GPen correlates with these for both methods: foot speed r ≈ +0.40, pelvis height r ≈ −0.48,
contact fraction r ≈ −0.37. FB is worse on every driver.

## Hypothesis table

| Hypothesis | Evidence | Verdict |
|---|---|---|
| **A. Measurement/conversion bug** | Same dump path as Naive, which scores 0.00 on the same clips; reference lands exactly on the 87.10 mm loader target; IK residual on penetrating frames 34.65 vs 34.25 mm (1.01×) | **unlikely** |
| **B. Root-height bias** | FB pelvis 7 mm lower and 30 mm below its own q_base; correlates with GPen (r = −0.48) — but foot p5 is identical (0.070) for both | **contributing, not dominant** |
| **C. Foot-pose difference** | Median foot pitch identical (24.9° vs 23.9°); but on penetrating frames pitch rises to 42.7° and the joint→sole offset to 81 mm | **contributing, tail-only** |
| **D. Collision-mesh mismatch** | Median joint→sole offset identical across methods (22.7 vs 23.2 mm) — a real ~23 mm constant, but the same for both | **real but not the differentiator** |
| **E. Contact/controller behaviour** | FB feet 1.31× faster, 19% more fast frames, 5% less contact time, lands 19% harder; penetrating frames have 3.53× foot speed. But only 63.6% occur in contact and the in-flight rate is 1.31× lower, not absent | **dominant driver of *how often*, but not stance-specific** |
| **G. Unconstrained retargeting DOF (added)** | Quest rig has no toe joint, so SMPL ankle→toe pitch is set only by the regulariser; joint→sole offset 40.2 mm median / 103.7 mm p95; **30.8% of penetrating frames occur while the foot is airborne**, where real penetration is impossible | **dominant amplifier** |

## Outcome: **C — mixed**

The pipeline is correct (no bug), and FB's motion is **genuinely more dynamic** — measurably faster
feet, shorter stance, harder landings, lower pelvis. That part is real controller behaviour.

But the mesh-GPen metric **amplifies it ~3.6×** through a degree of freedom the Quest rig cannot
constrain. The direction of the FB > Naive result is trustworthy; the magnitude is inflated by the
retargeting, not purely a property of the controller.

### Mechanism in plain terms

PHC guarantees its *collision capsules* never sink into the floor, and they don't — FB's foot joints
sit at the same height as Naive's. But we render a full SMPL-H body whose foot surface hangs about
23 mm below that joint, and whose toe angle nothing in the Quest data pins down. When FB's foot
whips through a fast transition, that free-floating foot tips downward and the toe of the
*reconstructed* mesh briefly dips under the ground — for a handful of frames at a time. The
simulator never sees it, because the simulator isn't simulating that foot.

### Recommendation on the headline number

Keep **6.03 mm / 0.041**, but caption it as measuring the *reconstructed mesh*, not the simulated
body, and note that lower-body figures are inflated by the unconstrained ankle→toe DOF. The safest
presentation statement is the ordering plus its driver ("FB produces faster, less-settled foot
motion"), not the absolute millimetres. Adding a toe target — or constraining foot pitch during
retargeting — would tighten the metric, but that is a change to the retargeting, not a bug fix.


---

## FINAL — FB vs Naive on the same 17 lower-body clips (both traces complete)

| quantity | Finetuned Backbone | Naive PHC | diff |
|---|---|---|---|
| joint→sole offset, median | 40.2 mm | 33.9 mm | +6.3 |
| joint→sole offset, **p95** | **103.7 mm** | **76.2 mm** | **+27.5** |
| pelvis Z | 0.915 | 0.922 | −7 mm |
| rigid-body foot Z | 0.082 | 0.085 | −2 mm |
| mesh sole Z | 0.047 | 0.059 | −12 mm |
| pen rate | **0.053** | **0.003** | **17× higher** |
| pen rate in contact | 0.059 | 0.006 | +0.052 |
| pen rate in flight | **0.045** | **0.000** | +0.045 |
| share of penetrating frames AIRBORNE | **30.8%** | **0.0%** | — |

### Resolution

The **amplifier is shared**: both methods carry a similar median joint→sole offset (40.2 vs
33.9 mm, ~19% apart), because both are reconstructed through the same toe-less Quest rig. That is a
property of the retargeting, not of either controller.

What is **FB-specific** is how hard it excites that amplifier:
* tail offset 103.7 vs 76.2 mm (+27.5 mm),
* penetration rate 17× higher,
* and, categorically, **FB penetrates on airborne frames (30.8%) while Naive never does (0.0%)**.

The airborne result is the cleanest evidence in the whole diagnosis. A foot in the air cannot
penetrate the floor, so those frames are 100% reconstruction artifact — and Naive, run through the
identical pipeline, produces exactly zero of them. The difference therefore cannot be blamed on the
pipeline; it is FB's motion driving a shared retargeting weakness into a regime Naive never reaches.

**Outcome C confirmed, with the split quantified:** genuine controller behaviour determines *how
often* the artifact fires (and it fires 17× more often); the shared unconstrained-foot DOF
determines *how large* each event reads. Roughly a third of FB's excess is provably not physics.
