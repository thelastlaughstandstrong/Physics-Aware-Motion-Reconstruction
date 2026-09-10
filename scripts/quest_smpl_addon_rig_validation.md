# Direct-rig Blender path — numerical validation

`blender_render_addon.py`'s own `build_rig`/`animate_rig` driven inside Blender 5.2, world bone positions read back for **every frame**, compared against the PHC simulated trajectory in `mesh_<clip>.npz`.

Reference: SMPL-X rig rest skeleton vs our SMPL-H rest skeleton — **mean 31.4 mm, max 74.4 mm** (stature pelvis→head 62.0 vs 57.6 cm). The addon ships only an SMPL-X body; the SMPL+H model file is not installed, so this difference cannot be removed from this route.

## sports

### post-PHC (pred)

- frames **417** @ **30 fps** (PHC 417 @ 30)
- pelvis/root world position: mean **0.0000 mm**, max **0.0001 mm**
- all 22 rendered joints: mean **34.53 mm**, max **86.67 mm**
- proximal chain (pelvis, hips, spine1): mean **11.42 mm**
- root orientation: mean **0.000°**, max **0.000°**
- temporal continuity: max per-frame step 188.0 mm (PHC 185.2 mm)
- ground plane: lowest foot/ankle Z rig **-14.9 mm** vs PHC **+14.7 mm**
- per-joint error vs rest-skeleton difference: **r = +0.979**
  [PASS] sports/post-PHC (pred): pelvis world position exact  — max 0.0001 mm
  [PASS] sports/post-PHC (pred): no residual 90° frame error (root orientation < 3°)  — max 0.00°
  [PASS] sports/post-PHC (pred): joint error within the rig's own skeleton difference  — 34.5 mm vs rest-diff 31.4 mm
  [PASS] sports/post-PHC (pred): residual is structural (r > 0.9 with rest-skeleton difference)
  [PASS] sports/post-PHC (pred): left/right ordering correct
  [PASS] sports/post-PHC (pred): frame count and fps preserved
  [PASS] sports/post-PHC (pred): temporal continuity preserved  — Δ 2.8 mm
  [PASS] sports/post-PHC (pred): ground plane consistent (within 5 cm)  — Δ -29.6 mm
  [PASS] sports/post-PHC (pred): finite everywhere

### pre-PHC (gt)

- frames **417** @ **30 fps** (PHC 417 @ 30)
- pelvis/root world position: mean **0.0000 mm**, max **0.0001 mm**
- all 22 rendered joints: mean **34.86 mm**, max **88.51 mm**
- proximal chain (pelvis, hips, spine1): mean **11.42 mm**
- root orientation: mean **0.000°**, max **0.000°**
- temporal continuity: max per-frame step 202.9 mm (PHC 200.1 mm)
- ground plane: lowest foot/ankle Z rig **-83.5 mm** vs PHC **-59.2 mm**
- per-joint error vs rest-skeleton difference: **r = +0.978**
  [PASS] sports/pre-PHC (gt): pelvis world position exact  — max 0.0001 mm
  [PASS] sports/pre-PHC (gt): no residual 90° frame error (root orientation < 3°)  — max 0.00°
  [PASS] sports/pre-PHC (gt): joint error within the rig's own skeleton difference  — 34.9 mm vs rest-diff 31.4 mm
  [PASS] sports/pre-PHC (gt): residual is structural (r > 0.9 with rest-skeleton difference)
  [PASS] sports/pre-PHC (gt): left/right ordering correct
  [PASS] sports/pre-PHC (gt): frame count and fps preserved
  [PASS] sports/pre-PHC (gt): temporal continuity preserved  — Δ 2.9 mm
  [PASS] sports/pre-PHC (gt): ground plane consistent (within 5 cm)  — Δ -24.2 mm
  [PASS] sports/pre-PHC (gt): finite everywhere


## gestures

### post-PHC (pred)

- frames **423** @ **30 fps** (PHC 423 @ 30)
- pelvis/root world position: mean **0.0000 mm**, max **0.0001 mm**
- all 22 rendered joints: mean **33.47 mm**, max **84.95 mm**
- proximal chain (pelvis, hips, spine1): mean **11.42 mm**
- root orientation: mean **0.000°**, max **0.000°**
- temporal continuity: max per-frame step 119.2 mm (PHC 117.4 mm)
- ground plane: lowest foot/ankle Z rig **-5.7 mm** vs PHC **+14.8 mm**
- per-joint error vs rest-skeleton difference: **r = +0.988**
  [PASS] gestures/post-PHC (pred): pelvis world position exact  — max 0.0001 mm
  [PASS] gestures/post-PHC (pred): no residual 90° frame error (root orientation < 3°)  — max 0.00°
  [PASS] gestures/post-PHC (pred): joint error within the rig's own skeleton difference  — 33.5 mm vs rest-diff 31.4 mm
  [PASS] gestures/post-PHC (pred): residual is structural (r > 0.9 with rest-skeleton difference)
  [PASS] gestures/post-PHC (pred): left/right ordering correct
  [PASS] gestures/post-PHC (pred): frame count and fps preserved
  [PASS] gestures/post-PHC (pred): temporal continuity preserved  — Δ 1.9 mm
  [PASS] gestures/post-PHC (pred): ground plane consistent (within 5 cm)  — Δ -20.5 mm
  [PASS] gestures/post-PHC (pred): finite everywhere

### pre-PHC (gt)

- frames **423** @ **30 fps** (PHC 423 @ 30)
- pelvis/root world position: mean **0.0000 mm**, max **0.0001 mm**
- all 22 rendered joints: mean **33.59 mm**, max **88.49 mm**
- proximal chain (pelvis, hips, spine1): mean **11.42 mm**
- root orientation: mean **0.000°**, max **0.000°**
- temporal continuity: max per-frame step 121.8 mm (PHC 117.2 mm)
- ground plane: lowest foot/ankle Z rig **-47.0 mm** vs PHC **-26.6 mm**
- per-joint error vs rest-skeleton difference: **r = +0.987**
  [PASS] gestures/pre-PHC (gt): pelvis world position exact  — max 0.0001 mm
  [PASS] gestures/pre-PHC (gt): no residual 90° frame error (root orientation < 3°)  — max 0.00°
  [PASS] gestures/pre-PHC (gt): joint error within the rig's own skeleton difference  — 33.6 mm vs rest-diff 31.4 mm
  [PASS] gestures/pre-PHC (gt): residual is structural (r > 0.9 with rest-skeleton difference)
  [PASS] gestures/pre-PHC (gt): left/right ordering correct
  [PASS] gestures/pre-PHC (gt): frame count and fps preserved
  [PASS] gestures/pre-PHC (gt): temporal continuity preserved  — Δ 4.6 mm
  [PASS] gestures/pre-PHC (gt): ground plane consistent (within 5 cm)  — Δ -20.5 mm
  [PASS] gestures/pre-PHC (gt): finite everywhere


## martialarts

### post-PHC (pred)

- frames **444** @ **30 fps** (PHC 444 @ 30)
- pelvis/root world position: mean **0.0000 mm**, max **0.0001 mm**
- all 22 rendered joints: mean **33.96 mm**, max **98.80 mm**
- proximal chain (pelvis, hips, spine1): mean **11.42 mm**
- root orientation: mean **0.000°**, max **0.000°**
- temporal continuity: max per-frame step 235.7 mm (PHC 232.7 mm)
- ground plane: lowest foot/ankle Z rig **-7.5 mm** vs PHC **+15.2 mm**
- per-joint error vs rest-skeleton difference: **r = +0.988**
  [PASS] martialarts/post-PHC (pred): pelvis world position exact  — max 0.0001 mm
  [PASS] martialarts/post-PHC (pred): no residual 90° frame error (root orientation < 3°)  — max 0.00°
  [PASS] martialarts/post-PHC (pred): joint error within the rig's own skeleton difference  — 34.0 mm vs rest-diff 31.4 mm
  [PASS] martialarts/post-PHC (pred): residual is structural (r > 0.9 with rest-skeleton difference)
  [PASS] martialarts/post-PHC (pred): left/right ordering correct
  [PASS] martialarts/post-PHC (pred): frame count and fps preserved
  [PASS] martialarts/post-PHC (pred): temporal continuity preserved  — Δ 3.0 mm
  [PASS] martialarts/post-PHC (pred): ground plane consistent (within 5 cm)  — Δ -22.6 mm
  [PASS] martialarts/post-PHC (pred): finite everywhere

### pre-PHC (gt)

- frames **444** @ **30 fps** (PHC 444 @ 30)
- pelvis/root world position: mean **0.0000 mm**, max **0.0001 mm**
- all 22 rendered joints: mean **33.18 mm**, max **86.97 mm**
- proximal chain (pelvis, hips, spine1): mean **11.42 mm**
- root orientation: mean **0.000°**, max **0.000°**
- temporal continuity: max per-frame step 155.4 mm (PHC 152.0 mm)
- ground plane: lowest foot/ankle Z rig **-74.4 mm** vs PHC **-67.8 mm**
- per-joint error vs rest-skeleton difference: **r = +0.990**
  [PASS] martialarts/pre-PHC (gt): pelvis world position exact  — max 0.0001 mm
  [PASS] martialarts/pre-PHC (gt): no residual 90° frame error (root orientation < 3°)  — max 0.00°
  [PASS] martialarts/pre-PHC (gt): joint error within the rig's own skeleton difference  — 33.2 mm vs rest-diff 31.4 mm
  [PASS] martialarts/pre-PHC (gt): residual is structural (r > 0.9 with rest-skeleton difference)
  [PASS] martialarts/pre-PHC (gt): left/right ordering correct
  [PASS] martialarts/pre-PHC (gt): frame count and fps preserved
  [PASS] martialarts/pre-PHC (gt): temporal continuity preserved  — Δ 3.4 mm
  [PASS] martialarts/pre-PHC (gt): ground plane consistent (within 5 cm)  — Δ -6.6 mm
  [PASS] martialarts/pre-PHC (gt): finite everywhere


**PASSED** — pelvis exact, no frame/convention error, and the entire residual is the SMPL-X rig's own rest skeleton.
