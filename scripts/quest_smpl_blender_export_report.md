# Blender export — validation

Each clip reconstructed from the exported AMASS npz alone and compared against the PHC simulated rigid-body trajectory in `phc_positions_for_eval.pkl`.

## sports

### post-PHC (pred) — `amass_pred_cs_sports.npz`

- frames **417** (dump 417), fps **30**, duration **13.867 s**, betas 10 (all zero: True), gender `neutral`
- joint position error, 22 rendered joints: mean **0.077 mm**, max **0.212 mm**
- hands (not driven by the addon): mean 39.5 mm — PHC's mjcf hand bone differs from real SMPL's by 40.8/38.0 mm
- root position error: mean **0.0000 mm**, max 0.0000 mm
- root orientation error: mean **3.259e-06°**, max 9.119e-06°
  [PASS] sports/post-PHC (pred): rendered-joint reconstruction < 1 mm  — max 0.212 mm
  [PASS] sports/post-PHC (pred): root position exact  — max 0.0000 mm
  [PASS] sports/post-PHC (pred): root orientation exact  — max 9.12e-06°
  [PASS] sports/post-PHC (pred): frame count and fps preserved
  [PASS] sports/post-PHC (pred): finite everywhere
  [PASS] sports/post-PHC (pred): joint ordering correct  — worst joint Head 0.193 mm
  [PASS] sports/post-PHC (pred): left/right correct (hips & shoulders never cross)
- temporal continuity: max per-frame joint step 185.2 mm (source 185.2 mm)
  [PASS] sports/post-PHC (pred): no discontinuity introduced by the export  — Δ 0.015 mm

### pre-PHC (retarget) — `amass_gt_cs_sports.npz`

- frames **417** (dump 417), fps **30**, duration **13.867 s**, betas 10 (all zero: True), gender `neutral`
- joint position error, 22 rendered joints: mean **0.083 mm**, max **0.783 mm**
- hands (not driven by the addon): mean 39.5 mm — PHC's mjcf hand bone differs from real SMPL's by 40.8/38.0 mm
- root position error: mean **0.0000 mm**, max 0.0000 mm
- root orientation error: mean **3.338e-06°**, max 8.859e-06°
  [PASS] sports/pre-PHC (retarget): rendered-joint reconstruction < 1 mm  — max 0.783 mm
  [PASS] sports/pre-PHC (retarget): root position exact  — max 0.0000 mm
  [PASS] sports/pre-PHC (retarget): root orientation exact  — max 8.86e-06°
  [PASS] sports/pre-PHC (retarget): frame count and fps preserved
  [PASS] sports/pre-PHC (retarget): finite everywhere
  [PASS] sports/pre-PHC (retarget): joint ordering correct  — worst joint Head 0.202 mm
  [PASS] sports/pre-PHC (retarget): left/right correct (hips & shoulders never cross)
- temporal continuity: max per-frame joint step 200.1 mm (source 200.1 mm)
  [PASS] sports/pre-PHC (retarget): no discontinuity introduced by the export  — Δ 0.015 mm


## gestures

### post-PHC (pred) — `amass_pred_cs_gestures.npz`

- frames **423** (dump 423), fps **30**, duration **14.067 s**, betas 10 (all zero: True), gender `neutral`
- joint position error, 22 rendered joints: mean **0.076 mm**, max **0.204 mm**
- hands (not driven by the addon): mean 39.4 mm — PHC's mjcf hand bone differs from real SMPL's by 40.8/38.0 mm
- root position error: mean **0.0000 mm**, max 0.0000 mm
- root orientation error: mean **3.565e-06°**, max 9.448e-06°
  [PASS] gestures/post-PHC (pred): rendered-joint reconstruction < 1 mm  — max 0.204 mm
  [PASS] gestures/post-PHC (pred): root position exact  — max 0.0000 mm
  [PASS] gestures/post-PHC (pred): root orientation exact  — max 9.45e-06°
  [PASS] gestures/post-PHC (pred): frame count and fps preserved
  [PASS] gestures/post-PHC (pred): finite everywhere
  [PASS] gestures/post-PHC (pred): joint ordering correct  — worst joint Head 0.190 mm
  [PASS] gestures/post-PHC (pred): left/right correct (hips & shoulders never cross)
- temporal continuity: max per-frame joint step 117.3 mm (source 117.4 mm)
  [PASS] gestures/post-PHC (pred): no discontinuity introduced by the export  — Δ 0.008 mm

### pre-PHC (retarget) — `amass_gt_cs_gestures.npz`

- frames **423** (dump 423), fps **30**, duration **14.067 s**, betas 10 (all zero: True), gender `neutral`
- joint position error, 22 rendered joints: mean **0.082 mm**, max **0.687 mm**
- hands (not driven by the addon): mean 39.5 mm — PHC's mjcf hand bone differs from real SMPL's by 40.8/38.0 mm
- root position error: mean **0.0000 mm**, max 0.0000 mm
- root orientation error: mean **3.534e-06°**, max 9.567e-06°
  [PASS] gestures/pre-PHC (retarget): rendered-joint reconstruction < 1 mm  — max 0.687 mm
  [PASS] gestures/pre-PHC (retarget): root position exact  — max 0.0000 mm
  [PASS] gestures/pre-PHC (retarget): root orientation exact  — max 9.57e-06°
  [PASS] gestures/pre-PHC (retarget): frame count and fps preserved
  [PASS] gestures/pre-PHC (retarget): finite everywhere
  [PASS] gestures/pre-PHC (retarget): joint ordering correct  — worst joint Head 0.192 mm
  [PASS] gestures/pre-PHC (retarget): left/right correct (hips & shoulders never cross)
- temporal continuity: max per-frame joint step 117.2 mm (source 117.2 mm)
  [PASS] gestures/pre-PHC (retarget): no discontinuity introduced by the export  — Δ 0.011 mm


## martialarts

### post-PHC (pred) — `amass_pred_cs_martialarts.npz`

- frames **444** (dump 444), fps **30**, duration **14.767 s**, betas 10 (all zero: True), gender `neutral`
- joint position error, 22 rendered joints: mean **0.078 mm**, max **0.213 mm**
- hands (not driven by the addon): mean 39.5 mm — PHC's mjcf hand bone differs from real SMPL's by 40.8/38.0 mm
- root position error: mean **0.0000 mm**, max 0.0000 mm
- root orientation error: mean **3.391e-06°**, max 9.270e-06°
  [PASS] martialarts/post-PHC (pred): rendered-joint reconstruction < 1 mm  — max 0.213 mm
  [PASS] martialarts/post-PHC (pred): root position exact  — max 0.0000 mm
  [PASS] martialarts/post-PHC (pred): root orientation exact  — max 9.27e-06°
  [PASS] martialarts/post-PHC (pred): frame count and fps preserved
  [PASS] martialarts/post-PHC (pred): finite everywhere
  [PASS] martialarts/post-PHC (pred): joint ordering correct  — worst joint Head 0.190 mm
  [PASS] martialarts/post-PHC (pred): left/right correct (hips & shoulders never cross)
- temporal continuity: max per-frame joint step 232.7 mm (source 232.7 mm)
  [PASS] martialarts/post-PHC (pred): no discontinuity introduced by the export  — Δ 0.023 mm

### pre-PHC (retarget) — `amass_gt_cs_martialarts.npz`

- frames **444** (dump 444), fps **30**, duration **14.767 s**, betas 10 (all zero: True), gender `neutral`
- joint position error, 22 rendered joints: mean **0.086 mm**, max **0.705 mm**
- hands (not driven by the addon): mean 39.5 mm — PHC's mjcf hand bone differs from real SMPL's by 40.8/38.0 mm
- root position error: mean **0.0000 mm**, max 0.0000 mm
- root orientation error: mean **3.352e-06°**, max 1.056e-05°
  [PASS] martialarts/pre-PHC (retarget): rendered-joint reconstruction < 1 mm  — max 0.705 mm
  [PASS] martialarts/pre-PHC (retarget): root position exact  — max 0.0000 mm
  [PASS] martialarts/pre-PHC (retarget): root orientation exact  — max 1.06e-05°
  [PASS] martialarts/pre-PHC (retarget): frame count and fps preserved
  [PASS] martialarts/pre-PHC (retarget): finite everywhere
  [PASS] martialarts/pre-PHC (retarget): joint ordering correct  — worst joint Head 0.197 mm
  [PASS] martialarts/pre-PHC (retarget): left/right correct (hips & shoulders never cross)
- temporal continuity: max per-frame joint step 152.0 mm (source 152.0 mm)
  [PASS] martialarts/pre-PHC (retarget): no discontinuity introduced by the export  — Δ 0.012 mm


**VALIDATION PASSED** — every export reconstructs the PHC trajectory to sub-millimetre.
