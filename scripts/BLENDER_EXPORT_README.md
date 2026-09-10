# Blender export — 3 representative clips

Prepared from the 10-clip characterization study. Export and validation only; no render produced.

| role | clip | source | net |
|---|---|---|---|
| beneficial | `sports` | `SUBJECT070.sports.pkl_0_417` | +2.41 |
| harmful | `gestures` | `SUBJECT070.gestures.pkl_0_423` | −0.79 |
| mixed | `martialarts` | `SUBJECT059.martialarts.pkl_0_444` | +0.42 |

---

## THE rendering path (only this one)

```
phc_positions_for_eval.pkl
  -> scripts/export_blender_mesh_motion.py     ->  mesh_<clip>.npz
  -> scripts/blender_render_addon.py           ->  posed SMPL-X rig in Blender
```

```
blender --background --python scripts/blender_render_addon.py -- \
    --npz data/quest_smpl_retarget/blender_export/mesh_<clip>.npz \
    [--out out.blend] [--offset 1.4]
```

`blender_render_addon.py` builds the official SMPL-X addon's rig/mesh/skinning but drives it
itself: **PHC's Z-up data is used directly**, and each bone gets the per-bone correction
`pose[i] = rest_relative[i]⁻¹ · local_smpl[i]`.

**No Y-up conversion. No `target_pelvis_zup_frame1` calibration. Neither is needed on this route** —
verified numerically below (root orientation error 0.000°, pelvis position error 0.0001 mm).

`quest_gt_<clip>.npz` carries the third stream (Quest ground truth) as joint positions for a marker
cloud / drawn skeleton. Use `positions_zup` — it is already in the same world and clock as the rig.

### Why the other path is not used

`amass_*.npz` → `object.smplx_add_animation` (Y-up + `target_pelvis_zup_frame1`) is **abandoned**.
Per `blender_render_addon.py`'s own docstring, that operator introduced an unexplained ~0.74 m
translation offset, and after calibrating it out the head still sat ~0.3 m too low, because
`set_pose_from_rodrigues` assumes the fed rotations already match each bone's non-uniform rest
orientation. The direct-rig route fixes that properly with the per-bone rest-relative correction.

The `amass_*.npz` files remain on disk but **are not an input to the final render**. Their Y-up
conversion is redundant on this route: this rig expresses its Y-up-ness as a `root` bone carrying a
+90° X rest rotation, so the frame change is already inside the rig.

---

## Validation

Two different things were validated. Both matter, and they are not interchangeable.

### 1. Export fidelity — parameters vs PHC skeleton — **sub-mm**

Reconstructing from the exported files and running SMPL FK reproduces PHC's trajectory at
**0.077 mm mean / 0.212 mm max** over the 22 rendered joints, root position exact, root orientation
3e-06°. 48/48 checks — [validate_blender_export.py](validate_blender_export.py).

### 2. Rig fidelity — Blender's posed rig vs PHC — **pelvis exact, joints ~34 mm**

`blender_render_addon.py`'s own `build_rig`/`animate_rig` driven inside Blender 5.2, world bone
positions read back for **every frame** of all three clips. 54/54 checks —
[validate_addon_rig_report.py](validate_addon_rig_report.py).

| | sports | gestures | martialarts |
|---|---|---|---|
| pelvis/root world position (max) | 0.0001 mm | 0.0001 mm | 0.0001 mm |
| **root orientation** (max) | **0.000°** | **0.000°** | **0.000°** |
| 22 rendered joints (mean) | 34.5 mm | 33.5 mm | ~34 mm |
| proximal chain (pelvis/hips/spine1) | 11.4 mm | 11.4 mm | 11.4 mm |
| left/right ordering | correct | correct | correct |
| frames @ fps | 417 @ 30 | 423 @ 30 | 444 @ 30 |
| temporal continuity Δ | 2.8 mm | 1.9 mm | ~3 mm |
| ground plane Δ (rig − PHC) | −29.6 mm | −20.5 mm | ~−25 mm |
| corr(per-joint error, rest-skeleton difference) | **+0.979** | **+0.988** | **+0.98** |

**The ~34 mm is the rig's own body, not a pipeline error.** SMPL-X neutral's rest skeleton differs
from the SMPL-H skeleton our rotations were fitted against by **mean 31.4 mm / max 74.4 mm**
(stature pelvis→head 62.0 vs 57.6 cm), and the per-joint error correlates with that difference at
**r ≈ 0.98**. Sub-mm is therefore *not attainable on this rig* — it would require driving an
SMPL/SMPL-H body. The addon supports `body_model = "smplh"`, but its model file
(`smplh_model_20260511.blend`) is **not installed**; installing it is the only way to close this gap.

### Bug found and fixed during this validation

The rig has an extra `root` bone above `pelvis` (`pelvis.parent = root`) carrying a **+90° about X**
rest rotation. `root` is not in `SMPL_TO_SMPLX_BONE`, so it is never posed. Using the
*parent-relative* rest for the pelvis left that 90° uncancelled:
`pelvis_arm_rot = root_rest · local_smpl` — the entire body rendered rotated 90° about X.

Measured before the fix: **568 mm mean / 1383 mm max** joint error, best-fit residual rotation
**91.05° about X**. Fix (one branch in `build_rig`): correct the pelvis against its **absolute**
rest rotation, since its parent is outside the SMPL chain. After: 34.5 mm, residual rotation 0.000°.

## Caveats for scene setup

1. **Hands are not driven.** SMPL-24's `L_Hand`/`R_Hand` have no SMPL-X equivalent and PHC runs
   `robot.freeze_hand=True`, so the addon's hands stay at rest.
2. **The rig's feet sit ~20–30 mm below PHC's.** A consequence of SMPL-X's longer legs, not a
   grounding error. If the ground plane must look exact, shift the armature, do not re-fit.
3. **Quest GT is not a mesh** — a different 25-joint rig. Only its 17 `ik_target_joints` were
   fitted; divergence on the other 8 is structural mismatch, not error.
4. **All three streams share one world and one clock** (frame shift −1, Z offset −44 to −55 mm,
   both properties of PHC's loader).
5. **Betas are zero / neutral**, matching the humanoid PHC simulated.
6. `--offset` in the render script separates the two rigs horizontally for viewing; it is set to 0
   during validation so world coordinates match PHC directly.

## Files — `data/quest_smpl_retarget/blender_export/`

| file | role |
|---|---|
| `mesh_<clip>.npz` | **render input** — positions + local quats, both streams |
| `quest_gt_<clip>.npz` | **render input** — Quest GT joint positions, `parents`, `ik_target_joints` |
| `amass_gt_*.npz`, `amass_pred_*.npz` | superseded, not used by the final render |

---

# Step 7 — final renders

```
blender --background --python scripts/blender_render_comparison.py -- \
    --npz mesh_<clip>.npz --quest quest_gt_<clip>.npz --tag <clip> --outdir <dir>
# then encode the PNG sequence (this Blender build has no FFMPEG output enum):
$CONDA_PREFIX/bin/ffmpeg -framerate 30 -i <dir>/_<clip>_frames/f_%04d.png \
    -c:v libx264 -pix_fmt yuv420p -crf 18 <dir>/<clip>_comparison.mp4
```
Or all three at once: `bash scripts/render_step7_all.sh`, then `bash scripts/make_step7_summary.sh`.

`blender_render_comparison.py` reuses the validated `build_rig`/`animate_rig` **unchanged** and only
adds scene, camera, labels and materials. Layout: Quest GT (skinned marker skeleton, x=−2.3) |
pre-physics (x=0) | post-PHC (x=+2.3), on a 0.5 m checker floor at Z=0. The camera follows one
shared reference (pre-physics root XY, smoothed over 31 frames) — it moves the **camera only**;
every column keeps its own full trajectory, including the post-PHC root drift.

Keyframes are forced to **LINEAR** interpolation. Blender's default Bezier easing would smooth the
motion, which would alter exactly what the render is evidence for.

## Outputs — `data/quest_smpl_retarget/renders/`

| file | contents |
|---|---|
| `sports_comparison.mp4` | 1920×1080, 417 frames @ 30 fps, 13.900 s |
| `gestures_comparison.mp4` | 1920×1080, 423 frames @ 30 fps, 14.100 s |
| `martialarts_comparison.mp4` | 1920×1080, 444 frames @ 30 fps, 14.800 s |
| `summary_all_three.mp4` | all three with title cards, 1509 frames, 50.300 s |
| `<clip>_still_f*.png` | 3 stills per clip, 1920×1080 |

18/18 verification checks pass ([verify_step7_renders.py](verify_step7_renders.py)): frame counts
match the motions exactly, PNG count == motion frames (no drops or duplicates), 30 fps, correct
durations, presentation resolution.

## Ground contact, measured on the motion itself

| clip | pre-physics frames with feet below floor | worst | post-PHC |
|---|---|---|---|
| sports | 217/417 (52%) | −59.2 mm | **0/417** |
| gestures | 41/423 (10%) | −26.6 mm | **0/423** |
| martialarts | 204/444 (46%) | −67.8 mm | **0/444** |

See [GROUND_PLANE_FINDING.md](GROUND_PLANE_FINDING.md) for why the rendered mesh sole sits ~12 mm
below Z=0 on every post-PHC clip and why that is rig geometry, not a floor error, and was left
uncorrected.
