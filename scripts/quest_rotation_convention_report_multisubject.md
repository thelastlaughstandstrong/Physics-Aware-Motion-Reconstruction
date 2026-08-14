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

## Clip 2: `SUBJECT015.teaching.pkl_0_431.pkl`

### Stored fields

| key | shape / value | dtype |
|---|---|---|

- frames: **432**, fps: **30**
- `pose_quat` norms: mean 1.000000 (unit-quaternion check)
- component means over all frames/joints: [0.0227, -0.0227, 0.0281, 0.9525] — a dominant LAST component implies **xyzw** (w last)

### Primary hypothesis — `is_local=True`, xyzw, root inside `pose_quat[:,0]`

```
rot: is_local=True, xyzw               mean   0.00172 deg   max   0.07913 deg   worst frame 2 joint 0 (b_root)
pos vs head_fk_gt_3d (raw/world)       mean 2111.19678 mm   max 3667.17871 mm   worst frame 382 joint 18 (b_r_wrist)
  -> max bone-length difference          0.00006 mm (0 => identical skeleton)
  -> after per-frame rigid align         0.00009 mm (0 => identical FK, frame-only difference)
pos vs head_gt_3d (raw/world)          mean 2223.88525 mm   max 3735.80298 mm   worst frame 382 joint 17 (b_r_wrist_twist)
  -> max bone-length difference          2.39271 mm (0 => identical skeleton)
  -> after per-frame rigid align        16.84725 mm (0 => identical FK, frame-only difference)
```

Per-joint mean angular error: min 0.000679, max 0.010246 deg (worst joint `b_root`)

## Clip 3: `SUBJECT034.workout.pkl_0_405.pkl`

### Stored fields

| key | shape / value | dtype |
|---|---|---|

- frames: **406**, fps: **30**
- `pose_quat` norms: mean 1.000000 (unit-quaternion check)
- component means over all frames/joints: [0.0223, -0.0080, 0.0148, 0.9177] — a dominant LAST component implies **xyzw** (w last)

### Primary hypothesis — `is_local=True`, xyzw, root inside `pose_quat[:,0]`

```
rot: is_local=True, xyzw               mean   0.00187 deg   max   0.06853 deg   worst frame 221 joint 0 (b_root)
pos vs head_fk_gt_3d (raw/world)       mean 1005.03204 mm   max 2301.82642 mm   worst frame 33 joint 18 (b_r_wrist)
  -> max bone-length difference          0.00003 mm (0 => identical skeleton)
  -> after per-frame rigid align         0.00008 mm (0 => identical FK, frame-only difference)
pos vs head_gt_3d (raw/world)          mean 1163.29041 mm   max 2424.07593 mm   worst frame 33 joint 18 (b_r_wrist)
  -> max bone-length difference          1.55106 mm (0 => identical skeleton)
  -> after per-frame rigid align        19.16565 mm (0 => identical FK, frame-only difference)
```

Per-joint mean angular error: min 0.000723, max 0.013919 deg (worst joint `b_root`)

## Clip 4: `SUBJECT057.yoga.pkl_0_410.pkl`

### Stored fields

| key | shape / value | dtype |
|---|---|---|

- frames: **411**, fps: **30**
- `pose_quat` norms: mean 1.000000 (unit-quaternion check)
- component means over all frames/joints: [0.0176, -0.0071, 0.0070, 0.9144] — a dominant LAST component implies **xyzw** (w last)

### Primary hypothesis — `is_local=True`, xyzw, root inside `pose_quat[:,0]`

```
rot: is_local=True, xyzw               mean   0.00211 deg   max   0.07913 deg   worst frame 394 joint 0 (b_root)
pos vs head_fk_gt_3d (raw/world)       mean 1269.42407 mm   max 2496.13477 mm   worst frame 88 joint 18 (b_r_wrist)
  -> max bone-length difference          0.00009 mm (0 => identical skeleton)
  -> after per-frame rigid align         0.00008 mm (0 => identical FK, frame-only difference)
pos vs head_gt_3d (raw/world)          mean 1432.87073 mm   max 2606.42090 mm   worst frame 88 joint 18 (b_r_wrist)
  -> max bone-length difference          1.74859 mm (0 => identical skeleton)
  -> after per-frame rigid align        13.71874 mm (0 => identical FK, frame-only difference)
```

Per-joint mean angular error: min 0.000193, max 0.012547 deg (worst joint `b_root`)

## Clip 5: `SUBJECT059.martialarts.pkl_0_444.pkl`

### Stored fields

| key | shape / value | dtype |
|---|---|---|

- frames: **445**, fps: **30**
- `pose_quat` norms: mean 1.000000 (unit-quaternion check)
- component means over all frames/joints: [0.0272, 0.0025, 0.0345, 0.9252] — a dominant LAST component implies **xyzw** (w last)

### Primary hypothesis — `is_local=True`, xyzw, root inside `pose_quat[:,0]`

```
rot: is_local=True, xyzw               mean   0.00216 deg   max   0.06853 deg   worst frame 11 joint 14 (p_r_scap)
pos vs head_fk_gt_3d (raw/world)       mean 1433.77307 mm   max 2848.65894 mm   worst frame 410 joint 24 (b_r_foot_twist)
  -> max bone-length difference          0.00015 mm (0 => identical skeleton)
  -> after per-frame rigid align         0.00008 mm (0 => identical FK, frame-only difference)
pos vs head_gt_3d (raw/world)          mean 1591.57056 mm   max 2992.86987 mm   worst frame 410 joint 24 (b_r_foot_twist)
  -> max bone-length difference          3.79521 mm (0 => identical skeleton)
  -> after per-frame rigid align        15.13522 mm (0 => identical FK, frame-only difference)
```

Per-joint mean angular error: min 0.000822, max 0.011662 deg (worst joint `b_root`)

## Clip 6: `SUBJECT061.stretching.pkl_0_438.pkl`

### Stored fields

| key | shape / value | dtype |
|---|---|---|

- frames: **439**, fps: **30**
- `pose_quat` norms: mean 1.000000 (unit-quaternion check)
- component means over all frames/joints: [0.0315, 0.0021, 0.0550, 0.9143] — a dominant LAST component implies **xyzw** (w last)

### Primary hypothesis — `is_local=True`, xyzw, root inside `pose_quat[:,0]`

```
rot: is_local=True, xyzw               mean   0.00225 deg   max   0.07913 deg   worst frame 438 joint 0 (b_root)
pos vs head_fk_gt_3d (raw/world)       mean 1210.18994 mm   max 2195.17969 mm   worst frame 0 joint 18 (b_r_wrist)
  -> max bone-length difference          0.00009 mm (0 => identical skeleton)
  -> after per-frame rigid align         0.00008 mm (0 => identical FK, frame-only difference)
pos vs head_gt_3d (raw/world)          mean 1383.75745 mm   max 2306.83423 mm   worst frame 1 joint 18 (b_r_wrist)
  -> max bone-length difference          4.53103 mm (0 => identical skeleton)
  -> after per-frame rigid align        15.35905 mm (0 => identical FK, frame-only difference)
```

Per-joint mean angular error: min 0.000631, max 0.013307 deg (worst joint `b_root`)

## Clip 7: `SUBJECT064.standing.pkl_0_444.pkl`

### Stored fields

| key | shape / value | dtype |
|---|---|---|

- frames: **445**, fps: **30**
- `pose_quat` norms: mean 1.000000 (unit-quaternion check)
- component means over all frames/joints: [0.0289, 0.0030, 0.0589, 0.9442] — a dominant LAST component implies **xyzw** (w last)

### Primary hypothesis — `is_local=True`, xyzw, root inside `pose_quat[:,0]`

```
rot: is_local=True, xyzw               mean   0.00232 deg   max   0.06853 deg   worst frame 82 joint 9 (b_l_arm)
pos vs head_fk_gt_3d (raw/world)       mean 1450.07861 mm   max 2314.97290 mm   worst frame 8 joint 18 (b_r_wrist)
  -> max bone-length difference          0.00015 mm (0 => identical skeleton)
  -> after per-frame rigid align         0.00008 mm (0 => identical FK, frame-only difference)
pos vs head_gt_3d (raw/world)          mean 1622.00330 mm   max 2432.63110 mm   worst frame 1 joint 18 (b_r_wrist)
  -> max bone-length difference          1.02416 mm (0 => identical skeleton)
  -> after per-frame rigid align        28.46096 mm (0 => identical FK, frame-only difference)
```

Per-joint mean angular error: min 0.000800, max 0.011115 deg (worst joint `b_root`)

## Clip 8: `SUBJECT066.standing.pkl_0_423.pkl`

### Stored fields

| key | shape / value | dtype |
|---|---|---|

- frames: **424**, fps: **30**
- `pose_quat` norms: mean 1.000000 (unit-quaternion check)
- component means over all frames/joints: [0.0340, 0.0030, 0.0542, 0.9416] — a dominant LAST component implies **xyzw** (w last)

### Primary hypothesis — `is_local=True`, xyzw, root inside `pose_quat[:,0]`

```
rot: is_local=True, xyzw               mean   0.00242 deg   max   0.06853 deg   worst frame 31 joint 0 (b_root)
pos vs head_fk_gt_3d (raw/world)       mean 1433.85718 mm   max 2345.79053 mm   worst frame 18 joint 18 (b_r_wrist)
  -> max bone-length difference          0.00006 mm (0 => identical skeleton)
  -> after per-frame rigid align         0.00007 mm (0 => identical FK, frame-only difference)
pos vs head_gt_3d (raw/world)          mean 1605.72571 mm   max 2461.72119 mm   worst frame 0 joint 18 (b_r_wrist)
  -> max bone-length difference          0.50265 mm (0 => identical skeleton)
  -> after per-frame rigid align        17.22194 mm (0 => identical FK, frame-only difference)
```

Per-joint mean angular error: min 0.000280, max 0.013125 deg (worst joint `b_root`)

## Clip 9: `SUBJECT069.standing.pkl_0_425.pkl`

### Stored fields

| key | shape / value | dtype |
|---|---|---|

- frames: **426**, fps: **30**
- `pose_quat` norms: mean 1.000000 (unit-quaternion check)
- component means over all frames/joints: [0.0261, -0.0009, 0.0569, 0.9458] — a dominant LAST component implies **xyzw** (w last)

### Primary hypothesis — `is_local=True`, xyzw, root inside `pose_quat[:,0]`

```
rot: is_local=True, xyzw               mean   0.00210 deg   max   0.06853 deg   worst frame 25 joint 0 (b_root)
pos vs head_fk_gt_3d (raw/world)       mean 1441.69739 mm   max 2313.39893 mm   worst frame 6 joint 18 (b_r_wrist)
  -> max bone-length difference          0.00003 mm (0 => identical skeleton)
  -> after per-frame rigid align         0.00008 mm (0 => identical FK, frame-only difference)
pos vs head_gt_3d (raw/world)          mean 1619.58276 mm   max 2413.83276 mm   worst frame 6 joint 18 (b_r_wrist)
  -> max bone-length difference          0.91344 mm (0 => identical skeleton)
  -> after per-frame rigid align        25.11161 mm (0 => identical FK, frame-only difference)
```

Per-joint mean angular error: min 0.000279, max 0.011147 deg (worst joint `b_root`)

## Clip 10: `SUBJECT070.ROM.pkl_0_435.pkl`

### Stored fields

| key | shape / value | dtype |
|---|---|---|

- frames: **436**, fps: **30**
- `pose_quat` norms: mean 1.000000 (unit-quaternion check)
- component means over all frames/joints: [0.0312, 0.0030, 0.0217, 0.9400] — a dominant LAST component implies **xyzw** (w last)

### Primary hypothesis — `is_local=True`, xyzw, root inside `pose_quat[:,0]`

```
rot: is_local=True, xyzw               mean   0.00258 deg   max   0.06853 deg   worst frame 70 joint 18 (b_r_wrist)
pos vs head_fk_gt_3d (raw/world)       mean 1479.58264 mm   max 2414.58667 mm   worst frame 57 joint 18 (b_r_wrist)
  -> max bone-length difference          0.00006 mm (0 => identical skeleton)
  -> after per-frame rigid align         0.00008 mm (0 => identical FK, frame-only difference)
pos vs head_gt_3d (raw/world)          mean 1637.53833 mm   max 2520.90674 mm   worst frame 57 joint 18 (b_r_wrist)
  -> max bone-length difference          0.04239 mm (0 => identical skeleton)
  -> after per-frame rigid align        22.75222 mm (0 => identical FK, frame-only difference)
```

Per-joint mean angular error: min 0.000272, max 0.012664 deg (worst joint `b_root`)

## Clip 11: `SUBJECT072.ROM.pkl_0_440.pkl`

### Stored fields

| key | shape / value | dtype |
|---|---|---|

- frames: **441**, fps: **30**
- `pose_quat` norms: mean 1.000000 (unit-quaternion check)
- component means over all frames/joints: [0.0214, -0.0028, 0.0265, 0.9323] — a dominant LAST component implies **xyzw** (w last)

### Primary hypothesis — `is_local=True`, xyzw, root inside `pose_quat[:,0]`

```
rot: is_local=True, xyzw               mean   0.00237 deg   max   0.07913 deg   worst frame 283 joint 12 (b_l_wrist)
pos vs head_fk_gt_3d (raw/world)       mean 1484.86646 mm   max 2248.95728 mm   worst frame 134 joint 18 (b_r_wrist)
  -> max bone-length difference          0.00012 mm (0 => identical skeleton)
  -> after per-frame rigid align         0.00008 mm (0 => identical FK, frame-only difference)
pos vs head_gt_3d (raw/world)          mean 1660.46362 mm   max 2383.78296 mm   worst frame 102 joint 18 (b_r_wrist)
  -> max bone-length difference          0.07650 mm (0 => identical skeleton)
  -> after per-frame rigid align        14.34705 mm (0 => identical FK, frame-only difference)
```

Per-joint mean angular error: min 0.000449, max 0.014798 deg (worst joint `b_root`)

## Clip 12: `SUBJECT073.ROM.pkl_0_406.pkl`

### Stored fields

| key | shape / value | dtype |
|---|---|---|

- frames: **407**, fps: **30**
- `pose_quat` norms: mean 1.000000 (unit-quaternion check)
- component means over all frames/joints: [0.0354, -0.0030, 0.0271, 0.9356] — a dominant LAST component implies **xyzw** (w last)

### Primary hypothesis — `is_local=True`, xyzw, root inside `pose_quat[:,0]`

```
rot: is_local=True, xyzw               mean   0.00245 deg   max   0.06853 deg   worst frame 117 joint 18 (b_r_wrist)
pos vs head_fk_gt_3d (raw/world)       mean 1484.31079 mm   max 2315.76416 mm   worst frame 0 joint 18 (b_r_wrist)
  -> max bone-length difference          0.00006 mm (0 => identical skeleton)
  -> after per-frame rigid align         0.00008 mm (0 => identical FK, frame-only difference)
pos vs head_gt_3d (raw/world)          mean 1652.87805 mm   max 2413.91479 mm   worst frame 0 joint 18 (b_r_wrist)
  -> max bone-length difference          0.37420 mm (0 => identical skeleton)
  -> after per-frame rigid align        22.98688 mm (0 => identical FK, frame-only difference)
```

Per-joint mean angular error: min 0.000389, max 0.012196 deg (worst joint `b_root`)

## Clip 13: `SUBJECT074.ROM.pkl_0_439.pkl`

### Stored fields

| key | shape / value | dtype |
|---|---|---|

- frames: **440**, fps: **30**
- `pose_quat` norms: mean 1.000000 (unit-quaternion check)
- component means over all frames/joints: [0.0253, -0.0008, 0.0260, 0.9399] — a dominant LAST component implies **xyzw** (w last)

### Primary hypothesis — `is_local=True`, xyzw, root inside `pose_quat[:,0]`

```
rot: is_local=True, xyzw               mean   0.00268 deg   max   0.06853 deg   worst frame 23 joint 17 (b_r_wrist_twist)
pos vs head_fk_gt_3d (raw/world)       mean 1468.04639 mm   max 2322.02588 mm   worst frame 47 joint 18 (b_r_wrist)
  -> max bone-length difference          0.00012 mm (0 => identical skeleton)
  -> after per-frame rigid align         0.00008 mm (0 => identical FK, frame-only difference)
pos vs head_gt_3d (raw/world)          mean 1641.71814 mm   max 2456.28320 mm   worst frame 47 joint 18 (b_r_wrist)
  -> max bone-length difference          0.26444 mm (0 => identical skeleton)
  -> after per-frame rigid align        20.60776 mm (0 => identical FK, frame-only difference)
```

Per-joint mean angular error: min 0.000614, max 0.015921 deg (worst joint `b_root`)

## Clip 14: `SUBJECT075.ROM.pkl_0_424.pkl`

### Stored fields

| key | shape / value | dtype |
|---|---|---|

- frames: **425**, fps: **30**
- `pose_quat` norms: mean 1.000000 (unit-quaternion check)
- component means over all frames/joints: [0.0285, -0.0010, -0.0047, 0.9277] — a dominant LAST component implies **xyzw** (w last)

### Primary hypothesis — `is_local=True`, xyzw, root inside `pose_quat[:,0]`

```
rot: is_local=True, xyzw               mean   0.00239 deg   max   0.06853 deg   worst frame 252 joint 0 (b_root)
pos vs head_fk_gt_3d (raw/world)       mean 1509.47314 mm   max 2395.76758 mm   worst frame 27 joint 18 (b_r_wrist)
  -> max bone-length difference          0.00009 mm (0 => identical skeleton)
  -> after per-frame rigid align         0.00008 mm (0 => identical FK, frame-only difference)
pos vs head_gt_3d (raw/world)          mean 1676.65002 mm   max 2514.39819 mm   worst frame 0 joint 18 (b_r_wrist)
  -> max bone-length difference          0.34735 mm (0 => identical skeleton)
  -> after per-frame rigid align        18.35246 mm (0 => identical FK, frame-only difference)
```

Per-joint mean angular error: min 0.000465, max 0.010222 deg (worst joint `b_root`)

## Clip 15: `SUBJECT076.ROM.pkl_0_435.pkl`

### Stored fields

| key | shape / value | dtype |
|---|---|---|

- frames: **436**, fps: **30**
- `pose_quat` norms: mean 1.000000 (unit-quaternion check)
- component means over all frames/joints: [0.0320, 0.0038, 0.0319, 0.9389] — a dominant LAST component implies **xyzw** (w last)

### Primary hypothesis — `is_local=True`, xyzw, root inside `pose_quat[:,0]`

```
rot: is_local=True, xyzw               mean   0.00235 deg   max   0.06853 deg   worst frame 49 joint 0 (b_root)
pos vs head_fk_gt_3d (raw/world)       mean 1496.38000 mm   max 2359.70923 mm   worst frame 86 joint 18 (b_r_wrist)
  -> max bone-length difference          0.00006 mm (0 => identical skeleton)
  -> after per-frame rigid align         0.00009 mm (0 => identical FK, frame-only difference)
pos vs head_gt_3d (raw/world)          mean 1672.21265 mm   max 2486.91260 mm   worst frame 86 joint 18 (b_r_wrist)
  -> max bone-length difference          0.07069 mm (0 => identical skeleton)
  -> after per-frame rigid align        20.69968 mm (0 => identical FK, frame-only difference)
```

Per-joint mean angular error: min 0.000363, max 0.012716 deg (worst joint `b_root`)

## Clip 16: `SUBJECT078.ROM.pkl_0_404.pkl`

### Stored fields

| key | shape / value | dtype |
|---|---|---|

- frames: **405**, fps: **30**
- `pose_quat` norms: mean 1.000000 (unit-quaternion check)
- component means over all frames/joints: [0.0276, -0.0015, 0.0155, 0.9366] — a dominant LAST component implies **xyzw** (w last)

### Primary hypothesis — `is_local=True`, xyzw, root inside `pose_quat[:,0]`

```
rot: is_local=True, xyzw               mean   0.00229 deg   max   0.06853 deg   worst frame 1 joint 0 (b_root)
pos vs head_fk_gt_3d (raw/world)       mean 1461.61230 mm   max 2298.68994 mm   worst frame 127 joint 18 (b_r_wrist)
  -> max bone-length difference          0.00003 mm (0 => identical skeleton)
  -> after per-frame rigid align         0.00008 mm (0 => identical FK, frame-only difference)
pos vs head_gt_3d (raw/world)          mean 1630.21631 mm   max 2423.22900 mm   worst frame 126 joint 18 (b_r_wrist)
  -> max bone-length difference          0.29749 mm (0 => identical skeleton)
  -> after per-frame rigid align        22.80920 mm (0 => identical FK, frame-only difference)
```

Per-joint mean angular error: min 0.000586, max 0.012248 deg (worst joint `b_root`)

## Clip 17: `SUBJECT079.ROM.pkl_0_428.pkl`

### Stored fields

| key | shape / value | dtype |
|---|---|---|

- frames: **429**, fps: **30**
- `pose_quat` norms: mean 1.000000 (unit-quaternion check)
- component means over all frames/joints: [0.0224, 0.0002, 0.0180, 0.9380] — a dominant LAST component implies **xyzw** (w last)

### Primary hypothesis — `is_local=True`, xyzw, root inside `pose_quat[:,0]`

```
rot: is_local=True, xyzw               mean   0.00258 deg   max   0.06853 deg   worst frame 27 joint 0 (b_root)
pos vs head_fk_gt_3d (raw/world)       mean 1520.69800 mm   max 2331.83667 mm   worst frame 87 joint 18 (b_r_wrist)
  -> max bone-length difference          0.00006 mm (0 => identical skeleton)
  -> after per-frame rigid align         0.00009 mm (0 => identical FK, frame-only difference)
pos vs head_gt_3d (raw/world)          mean 1695.51892 mm   max 2459.88672 mm   worst frame 72 joint 18 (b_r_wrist)
  -> max bone-length difference          0.05969 mm (0 => identical skeleton)
  -> after per-frame rigid align        14.26089 mm (0 => identical FK, frame-only difference)
```

Per-joint mean angular error: min 0.000369, max 0.013455 deg (worst joint `b_root`)

## Clip 18: `SUBJECT080.ROM.pkl_0_419.pkl`

### Stored fields

| key | shape / value | dtype |
|---|---|---|

- frames: **420**, fps: **30**
- `pose_quat` norms: mean 1.000000 (unit-quaternion check)
- component means over all frames/joints: [0.0228, -0.0018, 0.0052, 0.9286] — a dominant LAST component implies **xyzw** (w last)

### Primary hypothesis — `is_local=True`, xyzw, root inside `pose_quat[:,0]`

```
rot: is_local=True, xyzw               mean   0.00246 deg   max   0.06853 deg   worst frame 13 joint 13 (b_r_shoulder)
pos vs head_fk_gt_3d (raw/world)       mean 1513.84619 mm   max 2409.92041 mm   worst frame 98 joint 18 (b_r_wrist)
  -> max bone-length difference          0.00006 mm (0 => identical skeleton)
  -> after per-frame rigid align         0.00007 mm (0 => identical FK, frame-only difference)
pos vs head_gt_3d (raw/world)          mean 1682.72839 mm   max 2522.22900 mm   worst frame 98 joint 18 (b_r_wrist)
  -> max bone-length difference          0.04240 mm (0 => identical skeleton)
  -> after per-frame rigid align        27.12094 mm (0 => identical FK, frame-only difference)
```

Per-joint mean angular error: min 0.000283, max 0.013089 deg (worst joint `b_root`)

## Clip 19: `SUBJECT081.ROM.pkl_0_413.pkl`

### Stored fields

| key | shape / value | dtype |
|---|---|---|

- frames: **414**, fps: **30**
- `pose_quat` norms: mean 1.000000 (unit-quaternion check)
- component means over all frames/joints: [0.0265, 0.0050, 0.0106, 0.9256] — a dominant LAST component implies **xyzw** (w last)

### Primary hypothesis — `is_local=True`, xyzw, root inside `pose_quat[:,0]`

```
rot: is_local=True, xyzw               mean   0.00245 deg   max   0.06853 deg   worst frame 101 joint 5 (b_neck0)
pos vs head_fk_gt_3d (raw/world)       mean 1504.48315 mm   max 2298.48560 mm   worst frame 148 joint 18 (b_r_wrist)
  -> max bone-length difference          0.00009 mm (0 => identical skeleton)
  -> after per-frame rigid align         0.00008 mm (0 => identical FK, frame-only difference)
pos vs head_gt_3d (raw/world)          mean 1677.59302 mm   max 2417.32568 mm   worst frame 147 joint 18 (b_r_wrist)
  -> max bone-length difference          0.24420 mm (0 => identical skeleton)
  -> after per-frame rigid align        21.39692 mm (0 => identical FK, frame-only difference)
```

Per-joint mean angular error: min 0.000573, max 0.011630 deg (worst joint `b_root`)

## Clip 20: `SUBJECT083.ROM.pkl_0_401.pkl`

### Stored fields

| key | shape / value | dtype |
|---|---|---|

- frames: **402**, fps: **30**
- `pose_quat` norms: mean 1.000000 (unit-quaternion check)
- component means over all frames/joints: [0.0285, -0.0004, 0.0147, 0.9389] — a dominant LAST component implies **xyzw** (w last)

### Primary hypothesis — `is_local=True`, xyzw, root inside `pose_quat[:,0]`

```
rot: is_local=True, xyzw               mean   0.00272 deg   max   0.06853 deg   worst frame 31 joint 0 (b_root)
pos vs head_fk_gt_3d (raw/world)       mean 1487.57556 mm   max 2328.97656 mm   worst frame 14 joint 18 (b_r_wrist)
  -> max bone-length difference          0.00006 mm (0 => identical skeleton)
  -> after per-frame rigid align         0.00007 mm (0 => identical FK, frame-only difference)
pos vs head_gt_3d (raw/world)          mean 1654.47961 mm   max 2436.93506 mm   worst frame 14 joint 18 (b_r_wrist)
  -> max bone-length difference          0.36317 mm (0 => identical skeleton)
  -> after per-frame rigid align        26.04520 mm (0 => identical FK, frame-only difference)
```

Per-joint mean angular error: min 0.000394, max 0.014356 deg (worst joint `b_root`)

## Clip 21: `SUBJECT084.ROM.pkl_0_426.pkl`

### Stored fields

| key | shape / value | dtype |
|---|---|---|

- frames: **427**, fps: **30**
- `pose_quat` norms: mean 1.000000 (unit-quaternion check)
- component means over all frames/joints: [0.0437, -0.0014, 0.0291, 0.9425] — a dominant LAST component implies **xyzw** (w last)

### Primary hypothesis — `is_local=True`, xyzw, root inside `pose_quat[:,0]`

```
rot: is_local=True, xyzw               mean   0.00230 deg   max   0.06853 deg   worst frame 86 joint 0 (b_root)
pos vs head_fk_gt_3d (raw/world)       mean 1463.02112 mm   max 2419.04370 mm   worst frame 4 joint 18 (b_r_wrist)
  -> max bone-length difference          0.00012 mm (0 => identical skeleton)
  -> after per-frame rigid align         0.00009 mm (0 => identical FK, frame-only difference)
pos vs head_gt_3d (raw/world)          mean 1627.37256 mm   max 2510.95850 mm   worst frame 4 joint 18 (b_r_wrist)
  -> max bone-length difference          0.14377 mm (0 => identical skeleton)
  -> after per-frame rigid align        14.73832 mm (0 => identical FK, frame-only difference)
```

Per-joint mean angular error: min 0.000371, max 0.013762 deg (worst joint `b_root`)

## Clip 22: `SUBJECT086.ROM.pkl_0_437.pkl`

### Stored fields

| key | shape / value | dtype |
|---|---|---|

- frames: **438**, fps: **30**
- `pose_quat` norms: mean 1.000000 (unit-quaternion check)
- component means over all frames/joints: [0.0287, -0.0018, 0.0111, 0.9305] — a dominant LAST component implies **xyzw** (w last)

### Primary hypothesis — `is_local=True`, xyzw, root inside `pose_quat[:,0]`

```
rot: is_local=True, xyzw               mean   0.00255 deg   max   0.06853 deg   worst frame 29 joint 8 (p_l_scap)
pos vs head_fk_gt_3d (raw/world)       mean 1495.13538 mm   max 2370.43140 mm   worst frame 118 joint 18 (b_r_wrist)
  -> max bone-length difference          0.00006 mm (0 => identical skeleton)
  -> after per-frame rigid align         0.00008 mm (0 => identical FK, frame-only difference)
pos vs head_gt_3d (raw/world)          mean 1672.13110 mm   max 2494.74390 mm   worst frame 118 joint 18 (b_r_wrist)
  -> max bone-length difference          0.11352 mm (0 => identical skeleton)
  -> after per-frame rigid align        21.31017 mm (0 => identical FK, frame-only difference)
```

Per-joint mean angular error: min 0.000361, max 0.014225 deg (worst joint `b_root`)

## Clip 23: `SUBJECT088.ROM.pkl_0_438.pkl`

### Stored fields

| key | shape / value | dtype |
|---|---|---|

- frames: **439**, fps: **30**
- `pose_quat` norms: mean 1.000000 (unit-quaternion check)
- component means over all frames/joints: [0.0252, 0.0062, 0.0463, 0.9367] — a dominant LAST component implies **xyzw** (w last)

### Primary hypothesis — `is_local=True`, xyzw, root inside `pose_quat[:,0]`

```
rot: is_local=True, xyzw               mean   0.00244 deg   max   0.06853 deg   worst frame 26 joint 21 (b_l_foot_twist)
pos vs head_fk_gt_3d (raw/world)       mean 1500.89319 mm   max 2435.76147 mm   worst frame 102 joint 18 (b_r_wrist)
  -> max bone-length difference          0.00009 mm (0 => identical skeleton)
  -> after per-frame rigid align         0.00007 mm (0 => identical FK, frame-only difference)
pos vs head_gt_3d (raw/world)          mean 1683.85425 mm   max 2573.63232 mm   worst frame 102 joint 18 (b_r_wrist)
  -> max bone-length difference          0.84993 mm (0 => identical skeleton)
  -> after per-frame rigid align        20.00697 mm (0 => identical FK, frame-only difference)
```

Per-joint mean angular error: min 0.000270, max 0.014636 deg (worst joint `b_root`)

## Verdict

| clip | mean rot err (deg) | max rot err (deg) | pos err after rigid align (mm) | max bone-len diff (mm) |
|---|---|---|---|---|
| `SUBJECT008.drinks.pkl_0_397.pkl` | 0.002496 | 0.068528 | 0.00007 | 0.00015 |
| `SUBJECT015.teaching.pkl_0_431.pkl` | 0.001719 | 0.079129 | 0.00009 | 0.00006 |
| `SUBJECT034.workout.pkl_0_405.pkl` | 0.001865 | 0.068528 | 0.00008 | 0.00003 |
| `SUBJECT057.yoga.pkl_0_410.pkl` | 0.002105 | 0.079129 | 0.00008 | 0.00009 |
| `SUBJECT059.martialarts.pkl_0_444.pkl` | 0.002157 | 0.068528 | 0.00008 | 0.00015 |
| `SUBJECT061.stretching.pkl_0_438.pkl` | 0.002252 | 0.079129 | 0.00008 | 0.00009 |
| `SUBJECT064.standing.pkl_0_444.pkl` | 0.002319 | 0.068528 | 0.00008 | 0.00015 |
| `SUBJECT066.standing.pkl_0_423.pkl` | 0.002421 | 0.068528 | 0.00007 | 0.00006 |
| `SUBJECT069.standing.pkl_0_425.pkl` | 0.002101 | 0.068528 | 0.00008 | 0.00003 |
| `SUBJECT070.ROM.pkl_0_435.pkl` | 0.002583 | 0.068528 | 0.00008 | 0.00006 |
| `SUBJECT072.ROM.pkl_0_440.pkl` | 0.002367 | 0.079129 | 0.00008 | 0.00012 |
| `SUBJECT073.ROM.pkl_0_406.pkl` | 0.002449 | 0.068528 | 0.00008 | 0.00006 |
| `SUBJECT074.ROM.pkl_0_439.pkl` | 0.002685 | 0.068528 | 0.00008 | 0.00012 |
| `SUBJECT075.ROM.pkl_0_424.pkl` | 0.002385 | 0.068528 | 0.00008 | 0.00009 |
| `SUBJECT076.ROM.pkl_0_435.pkl` | 0.002354 | 0.068528 | 0.00009 | 0.00006 |
| `SUBJECT078.ROM.pkl_0_404.pkl` | 0.002290 | 0.068528 | 0.00008 | 0.00003 |
| `SUBJECT079.ROM.pkl_0_428.pkl` | 0.002584 | 0.068528 | 0.00009 | 0.00006 |
| `SUBJECT080.ROM.pkl_0_419.pkl` | 0.002456 | 0.068528 | 0.00007 | 0.00006 |
| `SUBJECT081.ROM.pkl_0_413.pkl` | 0.002449 | 0.068528 | 0.00008 | 0.00009 |
| `SUBJECT083.ROM.pkl_0_401.pkl` | 0.002722 | 0.068528 | 0.00007 | 0.00006 |
| `SUBJECT084.ROM.pkl_0_426.pkl` | 0.002302 | 0.068528 | 0.00009 | 0.00012 |
| `SUBJECT086.ROM.pkl_0_437.pkl` | 0.002547 | 0.068528 | 0.00008 | 0.00006 |
| `SUBJECT088.ROM.pkl_0_438.pkl` | 0.002444 | 0.068528 | 0.00007 | 0.00009 |

- Pass condition mean angular < 0.1 deg: **PASS** (worst clip mean 0.002722 deg)
- Pass condition mean position < 1 mm: **PASS** (worst clip 0.000092 mm, measured after removing the reference field's per-frame rigid transform -- see note below)

