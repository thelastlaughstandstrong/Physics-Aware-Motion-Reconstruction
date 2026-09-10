# FB vs Naive PHC — physics-control pipeline audit

Diagnosis only. No retraining, retuning, reward/retargeting/metric changes, no fixes applied.
Naive PHC (frozen `phc_prim_quest` teacher) is treated as the known-working control path.

## Step 1 — static code-path audit

Both runs use `env=env_quest_im_basic` + `robot=quest_humanoid` (verified from each run's saved
`.hydra/overrides.yaml`), so they instantiate the **same env task class** and the same control
config. Neither `learning=distill_quest` nor `learning=im_quest` overrides any env/control key.

| Item | Finetuned Backbone (student) | Naive PHC (teacher) | Verdict |
|---|---|---|---|
| action dimensionality | 72 (`action_head` out-dim) | 72 (`a2c_network.mu` out-dim) | **identical** |
| joint ordering / lower-body indices | env DOF order; **no remap in the action path** | same | **identical** |
| action → PD targets | `_action_to_pd_targets` = `offset + scale·a` | same function | **identical** |
| PD application | `set_dof_position_target_tensor` | same | **identical** |
| control mode | `isaac_pd` (`_pd_control=True`) | same | **identical** |
| PD stiffness / damping | from the same MJCF + robot cfg | same | **identical** |
| sim dt / decimation | `controlFrequencyInv: 2` | same | **identical** |
| residual-vs-absolute semantics | absolute PD target via shared offset/scale | same | **identical** |
| action bounds | `nn.Tanh()` → [−1,1] **by construction** | `clamp(−1,1)` + `rescale` (`clip_actions` defaults True) | **different mechanism, identical effect — inert** |
| eval loop / obs timing | same player loop; **only `get_action` is overridden** | same loop | **identical** |
| root treatment / reset | shared env | shared env | **identical** |
| observation content | AvatarPoser `z` + proprio + 27-D tracking error | PHC's own 972-D obs | **intentionally different** |
| tracked reference | GT (`test_seg_motion`) | AvatarPoser predictions (`test_seg_avatarposer_pred`) | **intentionally different — see caveat** |

### The one code asymmetry found, and why it is inert

`distill_agent_quest.py:1171-1177` returns `control` **unclamped**, whereas rl_games'
`PpoPlayerContinuous.get_action` (`players.py:64`) applies
`rescale_actions(low, high, clamp(a, −1, 1))` with `clip_actions` defaulting to `True`.
The action space is `Box(−1, 1)`, so that rescale is the identity and the teacher's path reduces to
a hard clamp.

**This cannot matter:** the student's `action_head` terminates in `nn.Tanh()`
(`distill_agent.py:355`) and `forward()` returns that tensor directly, so its output is already
confined to [−1,1]. The clamp the student skips would be a no-op. Both policies can therefore
command exactly the same PD-target range. *(Reported for completeness — worth aligning for clarity,
but it is not a behavioural difference.)*

## Step 2 — joint-level numerical evidence (17 lower-body clips, medians)

| quantity | FB | Naive | ratio |
|---|---|---|---|
| hip angular velocity (deg/s) | 153.3 | 133.5 | 1.15 |
| **knee angular velocity** | **196.6** | **194.4** | **1.01** |
| ankle angular velocity | 293.0 | 267.0 | 1.10 |
| **ankle angular velocity, p95** | **739.8** | **527.3** | **1.40** |
| ankle linear velocity (median) | 1.88 | 2.12 | 0.89 |
| ankle linear velocity, p95 | 4.59 | 4.06 | 1.13 |

**The excess is ankle-specific and tail-concentrated, not global.** A wrong action scale, gain, or
decimation would inflate every joint roughly uniformly; the knee — the largest lower-body joint —
matches to **1%**, and FB's *median* ankle linear velocity is actually **lower** than Naive's.
Only the ankle tail (p95) separates, at 1.40×. That is the signature of a learned policy, not a
mis-scaled control path.

## Step 4 — reset / settling audit (17 clips, medians)

| quantity | FB | Naive | diff |
|---|---|---|---|
| **reference** pelvis Z @ frame 0 | 0.9421 | 0.9481 | **−5.9 mm** |
| simulated pelvis Z @ frame 0 | 0.9358 | 0.9404 | −4.6 mm |
| pelvis Z, frames 60+ | 0.9136 | 0.9216 | −8.0 mm |
| **settling drop (f30-60 − f0)** | **−14.5 mm** | **−13.2 mm** | **−1.2 mm** |
| late-minus-early pelvis drift | −3.3 mm | −2.7 mm | −0.6 mm |

Settling and drift are nearly identical (1.2 mm and 0.6 mm apart). **The height gap is inherited,
not generated:** FB's reference already starts 5.9 mm lower, which is ~74% of the final 8.0 mm gap.
The control loop is not sinking the body relative to what it is asked to track.

### Evaluation caveat (by design, not a bug)

FB tracks **ground truth**; Naive PHC tracks **AvatarPoser's predictions**, which float higher than
GT (measured earlier: the backbone's output sits +13.3 mm above GT). So Naive is handed a
higher reference, ends higher, and retains more mesh clearance. This is inherent to how the naive
baseline is defined — it is not a defect in FB's control path, but it does mean part of the
FB-vs-Naive GPen gap reflects different tracking targets rather than controller quality.

## Step 3 — timing

Structural rather than numerical: both are driven by the **same player loop**, which calls
`get_action(obs)` then `env_step(action)`; only `get_action` is overridden. Observation, action and
physics step therefore occupy the same relative positions for both, and no ±1-frame offset can be
introduced by the student path without also affecting the teacher. No numerical ±1-frame sweep was
run — the shared call site makes a divergence structurally impossible.

## Conclusion — **Outcome B**

**No implementation mismatch in the physics-control pipeline.** Action dimensionality, joint
ordering, PD conversion, gains, control mode, decimation, root handling, reset and loop timing are
identical by construction, and the single code asymmetry found (missing clamp) is provably inert
because the student's head is `Tanh`-bounded.

The difference is **genuinely learned controller behaviour**, and it first appears at the **network
output / commanded joint motion** (A in the task's A–D scheme), localised to the **ankle tail**:
knee 1.01×, ankle p95 1.40×. It is amplified downstream by contact dynamics but does not originate
there.

Two effects that are *not* controller aggression and should be quantified separately:
1. **~74% of the pelvis-height gap is inherited from the different tracked references** (5.9 of
   8.0 mm), an evaluation-design property of the naive baseline.
2. **~30.8% of FB's mesh penetration occurs while airborne** (prior diagnostic), i.e. produced by
   the toe-less retargeting rather than by the simulator.

## Reportability

**6.03 mm mesh GPen, 17.14 cm/s foot skate and 524.78 m/s³ jitter remain valid and reportable.**
No bug was found that would invalidate them; all three are measured through the same code path as
the baselines. The existing caveats stand and are sufficient: GPen is a property of the
*reconstructed mesh* (a third of it non-physical), and part of the FB-vs-Naive gap reflects
different tracked references. Foot skate and jitter are measured directly on the simulated
rigid bodies and carry no such reconstruction caveat — they are clean.
