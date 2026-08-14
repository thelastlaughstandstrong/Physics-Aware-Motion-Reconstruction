# Quest -> SMPL Joint Correspondence (Step 2)

- Quest: `phc/phc/data/assets/mjcf/quest_humanoid.xml` — **25 joints**
- SMPL: `support_data/body_models/smplh/neutral/model.npz` (SMPL-H, 52 rows) — **24-joint** body convention
- Quest rest stature (head→foot) **1.4848 m**; SMPL rest stature (Head→Ankle) **1.4443 m** (ratio 1.0280 — no global rescale needed)

Counts reconcile exactly:

- Quest 25 = 17 direct + 4 spine (merged) + 4 discarded
- SMPL 24 = 17 direct + 3 spine (merged) + 4 synthesized

## 1. Direct correspondence (1:1)

| Quest idx | Quest joint | SMPL idx | SMPL joint | Quest chain (m) | SMPL chain (m) | Q/stature | S/stature | ratio |
|---|---|---|---|---|---|---|---|---|
| 0 | `b_root` | 0 | `Pelvis` | 0.0000 | 0.0000 | 0.0000 | 0.0000 | nan |
| 19 | `b_l_upleg` | 1 | `L_Hip` | 0.0840 | 0.1150 | 0.0566 | 0.0797 | 0.711 |
| 22 | `b_r_upleg` | 2 | `R_Hip` | 0.0840 | 0.1131 | 0.0566 | 0.0783 | 0.723 |
| 20 | `b_l_leg` | 4 | `L_Knee` | 0.4197 | 0.3768 | 0.2827 | 0.2609 | 1.084 |
| 23 | `b_r_leg` | 5 | `R_Knee` | 0.4197 | 0.3846 | 0.2827 | 0.2663 | 1.062 |
| 21 | `b_l_foot_twist` | 7 | `L_Ankle` | 0.4205 | 0.4006 | 0.2832 | 0.2774 | 1.021 |
| 24 | `b_r_foot_twist` | 8 | `R_Ankle` | 0.4205 | 0.4010 | 0.2832 | 0.2776 | 1.020 |
| 5 | `b_neck0` | 12 | `Neck` | 0.5780 | 0.5244 | 0.3893 | 0.3631 | 1.072 |
| 7 | `b_l_shoulder` | 13 | `L_Thorax` | 0.5654 | 0.4552 | 0.3808 | 0.3152 | 1.208 |
| 13 | `b_r_shoulder` | 14 | `R_Thorax` | 0.5654 | 0.4556 | 0.3808 | 0.3154 | 1.207 |
| 6 | `b_head` | 15 | `Head` | 0.0707 | 0.0830 | 0.0476 | 0.0574 | 0.829 |
| 9 | `b_l_arm` | 16 | `L_Shoulder` | 0.2019 | 0.0964 | 0.1360 | 0.0667 | 2.039 |
| 15 | `b_r_arm` | 17 | `R_Shoulder` | 0.2019 | 0.1018 | 0.1360 | 0.0705 | 1.930 |
| 10 | `b_l_forearm` | 18 | `L_Elbow` | 0.2568 | 0.2614 | 0.1730 | 0.1810 | 0.956 |
| 16 | `b_r_forearm` | 19 | `R_Elbow` | 0.2568 | 0.2550 | 0.1730 | 0.1766 | 0.980 |
| 12 | `b_l_wrist` | 20 | `L_Wrist` | 0.2618 | 0.2494 | 0.1763 | 0.1727 | 1.021 |
| 18 | `b_r_wrist` | 21 | `R_Wrist` | 0.2618 | 0.2555 | 0.1763 | 0.1769 | 0.997 |

**Chain** = cumulative distance to the nearest *mapped* ancestor, not the raw local offset. This is the quantity that must agree: discarded joints sit between mapped ones (Quest splits elbow→wrist across `b_*_wrist_twist` + `b_*_wrist`, so `b_*_wrist`'s own offset is only 0.03 m and comparing it to SMPL's 0.25 m forearm would be meaningless). Normalising by stature makes the columns frame- and scale-invariant, so they validate the mapping without needing the Step 3 frame calibration.

Length-ratio agreement over the 16 non-root pairs: median **1.021**, range 0.711–2.039 (1.000 = perfect proportional agreement).

### Where the irreducible residual will be — predicted before fitting

Pairs whose proportional length disagrees by more than 15% are structural mismatches, not fitting difficulty: no optimizer can place a joint correctly if the two skeletons put it at a different fraction of body height. Flagged automatically:

| Quest | SMPL | ratio | interpretation |
|---|---|---|---|
| `b_l_upleg` | `L_Hip` | 0.711 | Quest's hips sit closer to the root (narrower pelvis) than SMPL's |
| `b_r_upleg` | `R_Hip` | 0.723 | as above (right side) |
| `b_l_shoulder` | `L_Thorax` | 1.208 | Quest's clavicle attaches higher up its longer spine |
| `b_r_shoulder` | `R_Thorax` | 1.207 | as above (right side) |
| `b_head` | `Head` | 0.829 | Quest's neck→head segment is shorter relative to stature |
| `b_l_arm` | `L_Shoulder` | 2.039 | Quest routes clavicle→shoulder through the discarded `p_l_scap`, making that span ~2x SMPL's Thorax→Shoulder. **Expect the largest residual at the shoulders.** |
| `b_r_arm` | `R_Shoulder` | 1.930 | as above (right side) |

Everything else (legs, elbows, wrists, neck) agrees within ~8%. Notably the **wrist ratio is 1.021 / 0.997**, which confirms that merging `b_*_wrist_twist` into the forearm is the correct interpretation — the elbow→wrist span matches SMPL almost exactly once the twist joint is absorbed rather than mapped.

## 2. Merged (chain resample, NOT a joint mapping)

- Quest `['b_spine0', 'b_spine1', 'b_spine2', 'b_spine3']` (4 segments)
- SMPL  `['Torso', 'Spine', 'Chest']` (3 segments)

Cumulative chain length: Quest **0.4462 m** vs SMPL **0.3062 m** (0.301 vs 0.212 of stature).

> **This is the single largest structural mismatch.** Quest's spine is both longer in absolute terms and has one more segment. There is no joint-to-joint correspondence; the retargeter must fit the SMPL spine to the Quest chain's *shape*, and a non-zero residual here is unavoidable regardless of optimizer quality.

## 3. Discarded (Quest joints with no SMPL counterpart)

| Quest idx | Quest joint | reason |
|---|---|---|
| 8 | `p_l_scap` | scapula; SMPL has no scapular DOF (absorbed into L_Thorax/L_Shoulder) |
| 14 | `p_r_scap` | scapula; as above |
| 11 | `b_l_wrist_twist` | forearm twist DOF; SMPL models forearm as a single L_Elbow->L_Wrist bone |
| 17 | `b_r_wrist_twist` | forearm twist DOF; as above |

Discarding these is lossy but well-posed: they are twist/auxiliary DOFs whose *effect* on the child joint's world position is preserved, because the child is a direct-mapped fit target. Only the intermediate joint's own orientation is lost.

## 4. Synthesized (SMPL joints with no Quest source)

| SMPL idx | SMPL joint | reason |
|---|---|---|
| 10 | `L_Toe` | Quest's leg chain ends at b_l_foot_twist; there is no toe/sole joint |
| 11 | `R_Toe` | as above |
| 22 | `L_Hand` | Quest's arm chain ends at b_l_wrist; there is no hand joint |
| 23 | `R_Hand` | as above |

> These carry **no information from Quest**. They must be held at the SMPL rest pose (or driven by a prior) and must **never** be reported as retargeting accuracy — there is no ground truth for them.

## 5. Fit targets and validation protocol

- **Fit targets** (Step 5 optimization): 16 joints — the direct-mapped set minus the root: `['b_l_upleg', 'b_r_upleg', 'b_l_leg', 'b_r_leg', 'b_l_foot_twist', 'b_r_foot_twist', 'b_neck0', 'b_head', 'b_l_shoulder', 'b_r_shoulder', 'b_l_arm', 'b_r_arm', 'b_l_forearm', 'b_r_forearm', 'b_l_wrist', 'b_r_wrist']`

### Validation is per-joint and grouped, NOT a single global threshold

**Group A — high confidence** (11 joints, chain ratio within ~8%): `['head', 'l_elbow', 'r_elbow', 'l_wrist', 'r_wrist', 'l_hip', 'r_hip', 'l_knee', 'r_knee', 'l_ankle', 'r_ankle']`

**Group B — structurally mismatched** (5 joints, chain ratio off by >=15%): `['l_shoulder', 'r_shoulder', 'l_thorax', 'r_thorax', 'neck']`

**Excluded from accuracy entirely**: `['L_Toe', 'R_Toe', 'L_Hand', 'R_Hand']` — synthesized, no Quest source.

Reported outputs: (1) a **per-joint residual table** for all 16 mapped validation joints, and (2) a **mean over all mapped joints**, with the two groups summarised separately.

#### Why a single global threshold would be misleading

A pooled mean silently averages two error sources that call for different responses:

- **Optimizer convergence error** — fixable by better initialization, more iterations, or reweighting. Only Group A can isolate this, because those joints sit at nearly the same fraction of body height in both skeletons (ratios 0.96–1.08).
- **Irreducible structural mismatch** — a property of the two skeletons that no fit can remove. Four specific causes, all quantified in Section 1:
  - **Spine 4→3.** Quest's chain is 0.4462 m over four segments; SMPL's is 0.3062 m over three. The intermediate joints have no counterpart at all, so the SMPL spine can only approximate the Quest chain's shape — and `neck`, sitting on top of it, inherits that error.
  - **Discarded scapulae.** Quest's clavicle→shoulder span runs through `p_*_scap`, making it ~2.0x SMPL's Thorax→Shoulder (ratio 2.039 / 1.930). The shoulder cannot be placed correctly relative to both its parent and its child.
  - **Different shoulder topology.** `b_*_shoulder`→`*_Thorax` is off by ~21% for the same reason, so the whole shoulder girdle carries a floor.
  - **Synthesized hands/toes.** Zero Quest information. Including them would report the SMPL rest pose's agreement with itself and would *flatter* the result — hence the explicit exclusion.

Pooling these means a large, unavoidable shoulder residual could mask a genuine convergence failure in the legs, or a good leg fit could hide that the shoulders are unusable. Grouping makes the pass criterion actionable: **Group A residuals are a bug signal; Group B residuals are a design cost to be quantified and declared.**

> ⚠️ Still true regardless of grouping: the validation joints are a **subset of the fit targets**, so these are *fit-quality* numbers, not held-out generalization. No holdout is possible — every Quest joint with an SMPL counterpart is needed as a target.

## 6. Defect found in existing repo code

`phc/phc/learning/distill_agent.py::_load_smpl_rest_offsets()` computes

```python
J_reg = np.array(body_data['J_regressor'])[:24]
rest_joints = J_reg @ np.array(body_data['v_template'])
```

but the asset is **SMPL-H with 52 joints** laid out as 0–21 body, 22–36 left hand, 37–51 right hand. So rows 22–23 are the first two **left-hand** joints, not `L_Hand`/`R_Hand`. Measured consequence in the neutral model:

| SMPL-24 idx | name | row used | position | correct row | correct position |
|---|---|---|---|---|---|
| 22 | `L_Hand` | 22 | `[0.7838, 0.2132, -0.0221]` | 22 | (same — correct by luck) |
| 23 | `R_Hand` | 23 | `[0.8156, 0.2161, -0.0188]` | 37 | `[-0.7835, 0.2109, -0.022]` |

Row 23 sits at **+X** (the left side) like row 22, giving a bogus `R_Wrist→R_Hand` bone of **1.5000 m** instead of **0.1027 m**.

**Blast radius (checked, not assumed):**

- `_build_hmd_features` uses `_SMPL_HMD_JOINT_IDS = [15, 20, 21]` — all ≤ 21, **unaffected**.
- `DistillPlayer._accumulate_qbase_gpen` FKs `range(1,24)` — **affected**, but it applies the same wrong offsets to both prediction and ground truth, so the difference largely cancels in `qref_MPJPE`; only `gpen` could shift if the bogus joint crosses the floor.
- Quest pipeline: **unaffected** (never uses SMPL rest offsets).

This retargeter therefore uses `SMPLH_ROW_FOR_SMPL24 = list(range(22)) + [22, 37]` rather than `[:24]`. Since `L_Hand`/`R_Hand` are *synthesized* joints here, the fix does not change any fit target — it is corrected so the bug is not inherited, and flagged so the SMPL pipeline can be fixed separately.

