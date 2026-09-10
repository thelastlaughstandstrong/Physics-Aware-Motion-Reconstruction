
---

# Step 1 — SimXR's teacher/student setup, from the repository

| question | answer | evidence |
|---|---|---|
| Which pretrained PHC provides teacher actions | `output/HumanoidIm/phc_prim_quest/Humanoid.pth` — **md5-identical** (`4c5f57a2…`) to the checkpoint the Finetuned Backbone distils from | `env_simxr_quest.yaml:38`, `md5sum` both repos |
| Teacher frozen? | **Yes.** Loaded into the env, never added to an optimizer; `kin_optimizer` takes only `self.model.a2c_network.parameters()` | `humanoid_ego_img.py:85`, `amp_agent.py:67` |
| Teacher architecture | plain AMP actor: `actor_mlp` (3 layers) → `mu`. **No PNN, no MoE/composer, no VAE** — so `load_mcp_mlp` reconstructs the *complete* actor, i.e. both systems extract the same teacher function | checkpoint keys; `network_loader.py:11` |
| Exact action-imitation loss | `kin_loss = torch.norm(pred_action − gt_action, dim=−1).mean()` — **RMSE (L2 norm per sample), single term** | `amp_agent.py:843` |
| Is there a kinematic term too? | **No.** Despite the name, `only_kin_loss: True` means "only the behaviour-cloning term"; it routes past the entire PPO/AMP branch | `amp_agent.py:637` ("pure behavior cloning") |
| Action-smoothness / action-delta loss | **None.** The only temporal term in the file is an AR1 latent prior, reachable solely when `z_type == "vae"`; this env leaves `z_type` unset → `None` | `amp_agent.py:797`, `humanoid.py:339` |
| Ankle/foot-specific loss or weighting | **None** — the norm is taken over the full 72-D action vector | `amp_agent.py:843` |
| Contact-specific loss or reward | **None.** With `only_kin_loss` no reward is used at all | `amp_agent.py:637` |
| Teacher actions raw / sampled / clipped | **Raw `mu`**, unsampled and unclipped | `gt_action = self.encoder(full_obs)`, `humanoid_ego_img.py:466` |
| Temporal filtering before the simulator | **None.** `pre_physics_step` → `_action_to_pd_targets` directly | `humanoid.py:1349` |

**The premise needs correcting in one direction.** SimXR does not imitate teacher actions more weakly than
the Finetuned Backbone — it imitates them **more strongly**:

| | Finetuned Backbone | SimXR |
|---|---|---|
| objective | `1.0 · kinematic + 0.1 · action_MSE` | `‖pred − gt_action‖₂` **only** |
| relative weight on action imitation | 0.1 | **1.0, sole term** |
| smoothness term | none | none |

Nothing in SimXR's objective could suppress an ankle tail. Two *structural* differences are the real
candidates, and both are config-level:

| | FB (`env_quest_im_basic`) | SimXR (`env_simxr_quest`) |
|---|---|---|
| `reset_bodies` | 22 bodies, **including both legs** | `["b_head"]` only |
| `numTrajSamples` | 3 | **10** |
| `has_upright_start` | False | True (rest-pose convention only — world frame is Z-up in both, verified: root z ≈ 0.93, foot z ≈ 0.08 in both dumps) |

---

# Verdict: **B — SimXR does not smooth its commands; it realises them better**

SimXR **retains and in fact exceeds** PHC's ankle-heavy command pattern, while producing lower foot
skate. It is not option A, and not C.

## The four measurements that decide it

**1. SimXR's ankle asymmetry is the largest of any policy measured.**
Ankle ÷ knee |Δa| p95: SimXR **2.57**, naive PHC 2.42, FB 2.16, teacher\* 2.10. SimXR exceeds its own
knee on **38/38 clips** (sign test p = 7.3×10⁻¹²), minimum across clips 1.62 — it never approaches
parity. Its absolute ankle |Δa| p95 (0.180) is *higher* than FB's (0.154) and the teacher's (0.149).

**2. The two policies command the same amount of motion; they differ in its temporal structure.**

| group | mean \|a\| FB | mean \|a\| SimXR | lag-1 autocorrelation of Δa, FB | SimXR |
|---|---|---|---|---|
| hip | 0.0534 | 0.0537 | −0.181 | **−0.495** |
| knee | 0.0396 | 0.0388 | −0.407 | **−0.509** |
| ankle | 0.0822 | 0.0800 | −0.360 | **−0.502** |

Action magnitudes match to within 3% at every group. What differs is that SimXR's deltas are far more
*anti-correlated* — consecutive deltas reverse sign. Its larger |Δa| is **oscillatory chatter that
cancels**, not net movement, and the PD controller plus limb inertia low-pass most of it.

**3. That is why more command churn yields less realised motion.** SimXR's realised ankle angular
velocity is **0.93×** FB's despite 1.17× the command Δ-tail. Saturation is not involved: |a| reaches
1.0 on 1.1×10⁻⁶ of dimensions, p99.9 = 0.569.

**4. The lower skate is real, not a metric artifact.** Foot skate is contact-weighted, so a policy
whose feet hover would score falsely low. Contact fraction is **identical** — FB 0.956, SimXR 0.957.
The feet spend the same time in the contact band; SimXR's simply slide less.

## Realised behaviour, same clips, same definitions

| metric | FB | SimXR | ratio | SimXR lower in |
|---|---|---|---|---|
| foot skate (cm/s) | 8.22 | **5.62** | 0.68 | 82% of clips |
| ankle angular velocity p95 (°/s) | 198.9 | **184.9** | 0.93 | 61% |
| contact transitions / s | 0.8 | **0.7** | 0.86 | 61% |
| jitter (m/s³) | 97.9 | 279.8 | **2.86** | 8% |
| foot-joint tracking error (cm) | 11.86 | **9.35** | 0.79 | — |
| whole-body tracking error (cm) | **6.50** | 9.44 | 1.45 | — |

Directions agree with the full 1072-clip table (skate 0.84×, jitter 1.34× there); absolute magnitudes
differ because these are per-clip medians on 38 clips rather than full-test means.

**SimXR is not globally smoother — it is smoother only at the foot.** Its jitter is 1.8–2.9× FB's, and
it is jerkier on 92% of clips. The trade is visible in the tracking split: FB spends its accuracy on
whole-body pose (6.50 vs 9.44 cm), SimXR on the extremities (9.35 vs 11.86 cm at the feet). Since
foot skate is a foot metric, and the reference motion itself barely skates (**1.55 cm/s**), tracking
the feet more closely mechanically produces less skate. Neither policy inherits skate from the
reference — both add it (FB 5.3×, SimXR 3.6× the reference value).

---

# What SimXR is doing differently, in plain terms

Picture two drivers steering the same car toward the same route.

Both press the pedals by the same average amount — the *size* of their control inputs is nearly
identical. The difference is the wiggling. SimXR jiggles the wheel back and forth much faster, and
those jiggles mostly cancel out before they reach the wheels, because the steering linkage (here: the
PD controller and the limb's own mass) can't respond that quickly. So SimXR's twitchier-looking
commands actually produce slightly *less* foot rotation, not more.

The two also disagree about what "driving well" means. The Finetuned Backbone is graded on the whole
body: it is terminated if any of 22 body parts — including both legs — drifts more than 25 cm from
the reference. So it keeps hauling its legs toward the reference pose, and its feet pay for it,
skidding along the floor as it corrects. SimXR is graded on the head alone. It is free to let the rest
of the body be somewhat wrong, and what it gets in return is feet that land closer to where the
reference put them and stay there.

So the headline is not "SimXR is smoother". It is **"SimXR is jerkier overall but plants its feet
better"** — it buys foot quality with whole-body accuracy. That is the same trade already visible in
the results table, where SimXR's MPJPE (112.5 mm) and MPJRE (29.76°) are clearly worse than FB's
(87.43 mm, 15.23°) while its foot skate is better.

---

# Could any of this transfer to the Finetuned Backbone?

The measured deficit is specific: **FB's feet track the reference ~27% worse than SimXR's** (11.86 vs
9.35 cm) against a reference that is nearly skate-free. Anything that closes that gap should reduce
skate. Ranked by smallest architectural/training change first — none of these is causally tested, and
that is stated per row.

| # | Transfer | Change required | Evidence status |
|---|---|---|---|
| 1 | **Foot-weighted kinematic loss.** FB's `joint_rot_loss` and `joint_pos_loss` are uniform L1 over all joints (only the root has its own weight, `_kinematic_root_weight`). Adding a per-joint weight vector that upweights `b_l/r_foot_twist` is a single tensor multiply. | Loss only — no architecture, no observation, no dimension change. Retrain required. | Targets the one measured deficit directly. **But FB's kinematic loss is pre-simulation and the measured foot error is post-simulation** — the link is plausible, not demonstrated. |
| 2 | **Longer preview: `numTrajSamples` 3 → 10.** SimXR sees far more of the upcoming head trajectory. | One config value, but it changes the observation width → full retrain. | Config difference is verified; its causal role is **untested**. Plausible for foot-placement timing, unproven. |
| 3 | **Relax `reset_bodies` toward head-only.** | Config only. | Would raise success and change what "success" means, breaking comparability with every existing Quest2 number. **Not recommended** — it changes the metric, not the policy. |
| 4 | **Raise the action-imitation weight (0.1 → 1.0), matching SimXR's objective.** | One hyperparameter. | **Argues against itself.** SimXR's near-pure action imitation is precisely what gives it the *largest* ankle Δ-tail (2.57) and 2.9× the jitter. This would likely make FB jerkier. |

**Recommendation: only #1 is worth testing**, and it should be run as a controlled A/B with foot-joint
tracking error as the primary metric and whole-body MPJPE/MPJRE as a guardrail — the SimXR comparison
shows this trade is real, so foot gains may cost body accuracy. #2 is a reasonable second experiment
if #1 underdelivers. Nothing has been implemented.

---

## Caveats

- **The clip set is biased toward FB.** It was inherited from the 49-clip diagnostic, which selected
  clips where FB *and* naive PHC survive the full test. SimXR failed 13/56 here versus FB's 6/56;
  on the full test set they are much closer (0.717 vs 0.738). Command and realised comparisons use
  only the 38 clips where both survive, so the bias affects the survival counts, not the tables.
- **No counterfactual teacher stream in SimXR's states.** SimXR's env computes its BC target only
  when `not flags.test` (`humanoid_ego_img.py:421`); in an eval rollout the `gt_action` buffer stays
  at its zero initialisation. I logged it, verified it was **all zeros**, and discarded it rather
  than reporting it. So "what would the teacher command in SimXR's states" is unmeasured, and the
  teacher column here is the counterfactual in *FB's* states.
- **Different state distributions.** FB and SimXR share reference motions and an md5-identical robot,
  but not reset criteria or observation layout. Their command statistics are comparable; their
  visited states are not identical, and that difference is part of what is being measured.
- Mechanism findings 1–4 are measurements. The attributions to `reset_bodies` and `numTrajSamples`
  are hypotheses supported by verified config differences, **not** by intervention.
