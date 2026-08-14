# Quest Rotation Convention Verification (Step 1)

Skeleton: `phc/phc/data/assets/mjcf/quest_humanoid.xml` — 25 joints
Parents: `[-1, 0, 1, 2, 3, 4, 5, 4, 7, 8, 9, 10, 11, 4, 13, 14, 15, 16, 17, 0, 19, 20, 0, 22, 23]`

## Clip 1: `SUBJECT008.drinks.pkl_0_397.pkl`

### Stored fields

| key | shape / value | dtype |
|---|---|---|
| `pose_quat_global` | (398, 25, 4) | float32 |
| `pose_quat` | (398, 25, 4) | float32 |
| `trans_orig` | (398, 3) | float32 |
| `root_trans_offset` | (398, 3) | float32 |
| `fps` | 30 | int |
| `scale` | (398, 13) | float32 |
| `head_gt_3d` | (398, 25, 3) | float32 |
| `head_fk_gt_3d` | (398, 25, 3) | float64 |

- frames: **398**, fps: **30**
- `pose_quat` norms: mean 1.000000 (unit-quaternion check)
- component means over all frames/joints: [0.0282, -0.0016, 0.0241, 0.9430] — a dominant LAST component implies **xyzw** (w last)

### Primary hypothesis — `is_local=True`, xyzw, root inside `pose_quat[:,0]`

```
rot: is_local=True, xyzw               mean   0.00250 deg   max   0.06853 deg   worst frame 60 joint 0 (b_root)
pos vs head_fk_gt_3d (raw/world)       mean 1419.26660 mm   max 2332.76367 mm   worst frame 0 joint 18 (b_r_wrist)
  -> max bone-length difference          0.00015 mm (0 => identical skeleton)
  -> after per-frame rigid align         0.00007 mm (0 => identical FK, frame-only difference)
pos vs head_gt_3d (raw/world)          mean 1585.83386 mm   max 2457.67505 mm   worst frame 24 joint 18 (b_r_wrist)
  -> max bone-length difference          1.82456 mm (0 => identical skeleton)
  -> after per-frame rigid align        21.74748 mm (0 => identical FK, frame-only difference)
```

Per-joint mean angular error (deg):

| idx | joint | mean deg |
|---|---|---|
| 0 | `b_root` | 0.014273 |
| 1 | `b_spine0` | 0.002187 |
| 2 | `b_spine1` | 0.000994 |
| 3 | `b_spine2` | 0.001588 |
| 4 | `b_spine3` | 0.002012 |
| 5 | `b_neck0` | 0.001995 |
| 6 | `b_head` | 0.001772 |
| 7 | `b_l_shoulder` | 0.003785 |
| 8 | `p_l_scap` | 0.003559 |
| 9 | `b_l_arm` | 0.002633 |
| 10 | `b_l_forearm` | 0.002252 |
| 11 | `b_l_wrist_twist` | 0.002012 |
| 12 | `b_l_wrist` | 0.001821 |
| 13 | `b_r_shoulder` | 0.002252 |
| 14 | `p_r_scap` | 0.003668 |
| 15 | `b_r_arm` | 0.001896 |
| 16 | `b_r_forearm` | 0.001639 |
| 17 | `b_r_wrist_twist` | 0.001532 |
| 18 | `b_r_wrist` | 0.001176 |
| 19 | `b_l_upleg` | 0.002386 |
| 20 | `b_l_leg` | 0.001491 |
| 21 | `b_l_foot_twist` | 0.001234 |
| 22 | `b_r_upleg` | 0.001831 |
| 23 | `b_r_leg` | 0.001515 |
| 24 | `b_r_foot_twist` | 0.000895 |

### Failure-mode sweep (clip 1)

Alternative interpretations, to confirm the primary one is not accidentally close:

```
is_local=True, xyzw (PRIMARY)          mean   0.00250 deg   max   0.06853 deg
is_local=False, xyzw                   mean 152.49345 deg   max 179.98978 deg
is_local=True, stored read as wxyz     mean 109.35909 deg   max 179.99759 deg
root forced to identity                mean 168.47104 deg   max 179.86354 deg
root x upright [.5,.5,.5,.5]           mean 120.00000 deg   max 120.00003 deg
root x quest->smpl [-.5,-.5,-.5,.5]    mean 120.00000 deg   max 120.00003 deg
```

Sanity: feeding `pose_quat_global` back with `is_local=False` reproduces itself to 0.003658 deg mean — confirms the comparison target is self-consistent.

## Clip 2: `SUBJECT008.drinks.pkl_1193_1589.pkl`

### Stored fields

| key | shape / value | dtype |
|---|---|---|
| `pose_quat_global` | (397, 25, 4) | float32 |
| `pose_quat` | (397, 25, 4) | float32 |
| `trans_orig` | (397, 3) | float32 |
| `root_trans_offset` | (397, 3) | float32 |
| `fps` | 30 | int |
| `scale` | (397, 13) | float32 |
| `head_gt_3d` | (397, 25, 3) | float32 |
| `head_fk_gt_3d` | (397, 25, 3) | float64 |

- frames: **397**, fps: **30**
- `pose_quat` norms: mean 1.000000 (unit-quaternion check)
- component means over all frames/joints: [0.0359, 0.0038, -0.0004, 0.9341] — a dominant LAST component implies **xyzw** (w last)

### Primary hypothesis — `is_local=True`, xyzw, root inside `pose_quat[:,0]`

```
rot: is_local=True, xyzw               mean   0.00223 deg   max   0.06853 deg   worst frame 108 joint 7 (b_l_shoulder)
pos vs head_fk_gt_3d (raw/world)       mean 1417.75952 mm   max 2000.17810 mm   worst frame 1 joint 17 (b_r_wrist_twist)
  -> max bone-length difference          0.00009 mm (0 => identical skeleton)
  -> after per-frame rigid align         0.00008 mm (0 => identical FK, frame-only difference)
pos vs head_gt_3d (raw/world)          mean 1583.63635 mm   max 2104.47217 mm   worst frame 1 joint 17 (b_r_wrist_twist)
  -> max bone-length difference          1.57762 mm (0 => identical skeleton)
  -> after per-frame rigid align        24.65111 mm (0 => identical FK, frame-only difference)
```

Per-joint mean angular error (deg):

| idx | joint | mean deg |
|---|---|---|
| 0 | `b_root` | 0.014785 |
| 1 | `b_spine0` | 0.001794 |
| 2 | `b_spine1` | 0.001138 |
| 3 | `b_spine2` | 0.001976 |
| 4 | `b_spine3` | 0.001337 |
| 5 | `b_neck0` | 0.002134 |
| 6 | `b_head` | 0.001021 |
| 7 | `b_l_shoulder` | 0.002846 |
| 8 | `p_l_scap` | 0.004557 |
| 9 | `b_l_arm` | 0.002059 |
| 10 | `b_l_forearm` | 0.000938 |
| 11 | `b_l_wrist_twist` | 0.000979 |
| 12 | `b_l_wrist` | 0.000880 |
| 13 | `b_r_shoulder` | 0.001461 |
| 14 | `p_r_scap` | 0.002433 |
| 15 | `b_r_arm` | 0.002059 |
| 16 | `b_r_forearm` | 0.001096 |
| 17 | `b_r_wrist_twist` | 0.002158 |
| 18 | `b_r_wrist` | 0.000839 |
| 19 | `b_l_upleg` | 0.001636 |
| 20 | `b_l_leg` | 0.000997 |
| 21 | `b_l_foot_twist` | 0.000797 |
| 22 | `b_r_upleg` | 0.001976 |
| 23 | `b_r_leg` | 0.002017 |
| 24 | `b_r_foot_twist` | 0.001718 |

## Clip 3: `SUBJECT008.drinks.pkl_1590_1986.pkl`

### Stored fields

| key | shape / value | dtype |
|---|---|---|
| `pose_quat_global` | (397, 25, 4) | float32 |
| `pose_quat` | (397, 25, 4) | float32 |
| `trans_orig` | (397, 3) | float32 |
| `root_trans_offset` | (397, 3) | float32 |
| `fps` | 30 | int |
| `scale` | (397, 13) | float32 |
| `head_gt_3d` | (397, 25, 3) | float32 |
| `head_fk_gt_3d` | (397, 25, 3) | float64 |

- frames: **397**, fps: **30**
- `pose_quat` norms: mean 1.000000 (unit-quaternion check)
- component means over all frames/joints: [0.0258, 0.0026, -0.0055, 0.9338] — a dominant LAST component implies **xyzw** (w last)

### Primary hypothesis — `is_local=True`, xyzw, root inside `pose_quat[:,0]`

```
rot: is_local=True, xyzw               mean   0.00230 deg   max   0.06853 deg   worst frame 5 joint 0 (b_root)
pos vs head_fk_gt_3d (raw/world)       mean 1413.33496 mm   max 2141.87744 mm   worst frame 216 joint 17 (b_r_wrist_twist)
  -> max bone-length difference          0.00006 mm (0 => identical skeleton)
  -> after per-frame rigid align         0.00008 mm (0 => identical FK, frame-only difference)
pos vs head_gt_3d (raw/world)          mean 1579.16479 mm   max 2270.44043 mm   worst frame 216 joint 17 (b_r_wrist_twist)
  -> max bone-length difference          1.70261 mm (0 => identical skeleton)
  -> after per-frame rigid align        24.18953 mm (0 => identical FK, frame-only difference)
```

Per-joint mean angular error (deg):

| idx | joint | mean deg |
|---|---|---|
| 0 | `b_root` | 0.012692 |
| 1 | `b_spine0` | 0.001536 |
| 2 | `b_spine1` | 0.001278 |
| 3 | `b_spine2` | 0.001478 |
| 4 | `b_spine3` | 0.001859 |
| 5 | `b_neck0` | 0.003677 |
| 6 | `b_head` | 0.001735 |
| 7 | `b_l_shoulder` | 0.003361 |
| 8 | `p_l_scap` | 0.003378 |
| 9 | `b_l_arm` | 0.001735 |
| 10 | `b_l_forearm` | 0.000839 |
| 11 | `b_l_wrist_twist` | 0.001577 |
| 12 | `b_l_wrist` | 0.000382 |
| 13 | `b_r_shoulder` | 0.001760 |
| 14 | `p_r_scap` | 0.002175 |
| 15 | `b_r_arm` | 0.002798 |
| 16 | `b_r_forearm` | 0.001237 |
| 17 | `b_r_wrist_twist` | 0.001735 |
| 18 | `b_r_wrist` | 0.003014 |
| 19 | `b_l_upleg` | 0.002433 |
| 20 | `b_l_leg` | 0.001495 |
| 21 | `b_l_foot_twist` | 0.000399 |
| 22 | `b_r_upleg` | 0.001096 |
| 23 | `b_r_leg` | 0.002175 |
| 24 | `b_r_foot_twist` | 0.001660 |

## Verdict

| clip | mean rot err (deg) | max rot err (deg) | pos err after rigid align (mm) | max bone-len diff (mm) |
|---|---|---|---|---|
| `SUBJECT008.drinks.pkl_0_397.pkl` | 0.002496 | 0.068528 | 0.00007 | 0.00015 |
| `SUBJECT008.drinks.pkl_1193_1589.pkl` | 0.002225 | 0.068528 | 0.00008 | 0.00009 |
| `SUBJECT008.drinks.pkl_1590_1986.pkl` | 0.002300 | 0.068528 | 0.00008 | 0.00006 |

- Pass condition mean angular < 0.1 deg: **PASS** (worst clip mean 0.002496 deg)
- Pass condition mean position < 1 mm: **PASS** (worst clip 0.000080 mm, measured after removing the reference field's per-frame rigid transform -- see note below)


## Note on the reference-position fields

The raw world-frame comparison against `head_fk_gt_3d` / `head_gt_3d` shows ~1.4 m error. That is
**a frame mismatch in the stored fields, not an FK error**. Evidence:

- `head_fk_gt_3d[:, 6]` (the head joint) is exactly `[0,0,0]` — the field is stored **head-anchored**.
- Head-centring alone leaves ~605 mm, and transforming into the head's own rotation frame leaves
  ~364 mm, so it is neither a pure translation nor a simple head-local frame.
- **Bone lengths agree to 1.5e-4 mm** — it is provably the same skeleton with the same proportions.
- **A per-frame rigid transform (rotation + translation, no scale) reconciles it to 8e-5 mm.**

So `head_fk_gt_3d` *is* this skeleton's forward kinematics, expressed in an undocumented per-frame
anchored frame (most likely the headset/camera frame used by SimXR's vision pipeline). Our FK
reproduces it exactly once that frame is removed.

`head_gt_3d` is different: bone lengths differ by ~1.6–1.8 mm and it does not reduce below
~22–25 mm after rigid alignment. It appears to be a measured/noisy variant rather than pure FK, and
should **not** be used as an exact validation target.

## Conclusions

1. **Confirmed convention:** `pose_quat` is **local (parent-relative), xyzw, with the root rotation
   at index 0**, paired with `root_trans_offset` as the root translation. Reconstruct with
   `SkeletonState.from_rotation_and_root_translation(tree, pose_quat, root_trans_offset,
   is_local=True)`.
2. **Residual is at the numerical floor.** Mean angular error 0.0022–0.0025°, max 0.069°. For
   reference, feeding `pose_quat_global` back through the tree with `is_local=False` reproduces
   *itself* to 0.0037° mean — so our residual is at or below the file's own float32 round-trip
   noise. The 0.069° max is a single-frame artifact of that same float32 storage, not a systematic
   error.
3. **Every alternative interpretation is decisively wrong** (109°–168° mean), so this is not an
   accidental near-match:

   | interpretation | mean err |
   |---|---|
   | `is_local=True`, xyzw (**correct**) | **0.0025°** |
   | `is_local=False`, xyzw | 152.49° |
   | stored read as wxyz | 109.36° |
   | root forced to identity | 168.47° |
   | root × upright `[.5,.5,.5,.5]` | 120.00° |
   | root × quest→smpl `[-.5,-.5,-.5,.5]` | 120.00° |

   Note the two fixed base-frame rotations give exactly 120.00° — the signature of an unwanted
   constant rotation, confirming **no base-frame correction should be applied** at this stage.
4. `trans_orig` and `root_trans_offset` are **identical** in these clips, so either works here —
   but `root_trans_offset` is what the rest of the pipeline uses and should remain the choice.

## Remaining ambiguity

- The exact definition of the `head_fk_gt_3d` anchor frame is **not resolved** — only that it is a
  per-frame rigid transform away from our FK. This does not block retargeting (we use world-frame
  FK), but it means that field cannot be used as a drop-in validation target without first solving
  for its transform.
- `head_gt_3d`'s ~22–25 mm residual is unexplained; treat it as approximate.
- Verified on 3 clips from a single subject (SUBJECT008). Worth re-running across subjects before
  relying on it globally — the command below takes `--n-clips`.
