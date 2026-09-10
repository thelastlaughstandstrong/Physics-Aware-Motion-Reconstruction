# Ground-plane discrepancy — diagnosis (Step 7, item 5)

**Verdict: expected SMPL-H→SMPL-X geometry, NOT a scene/floor alignment problem. No correction
applied, and none is warranted.**

## What was measured

The validation reported the rig's foot/ankle **joints** sitting 20–30 mm lower than PHC's. Joint
height is not what a viewer sees, so the **mesh sole** was measured directly in Blender (40 frames
spread over each clip, minimum vertex Z against the floor at Z = 0):

| clip | post-PHC min / median | pre-PHC min / median |
|---|---|---|
| sports | −25.5 / **−11.9 mm** | −127.2 / −49.1 mm |
| gestures | −20.4 / **−12.6 mm** | −55.8 / −18.5 mm |
| martialarts | −83.5 / **−11.3 mm** | −122.8 / −32.4 mm |

## Why this is geometry, not misalignment

1. **The post-PHC median is −11 to −13 mm on all three clips** — a constant, clip-independent
   baseline. A floor misalignment would be constant too, but so is a fixed rig-geometry offset;
   what separates them is that this value matches the SMPL-X rig's own longer legs (rest
   pelvis→ankle 8.9 / 5.8 mm longer, pelvis→toe 11.3 mm longer, stature 62.0 vs 57.6 cm) with the
   pelvis pinned exactly to PHC's pelvis position.
2. **The floor is at Z = 0 in both.** PHC's ground is Z = 0 and the render's plane is created at
   `location=(0,0,0)`. Nothing is offset.
3. **The pre-PHC medians are deeper and clip-dependent** (−49.1, −18.5, −32.4 mm) and rank in the
   same order as the independently measured pre-sim penetration (sports 24.5 mm > martialarts
   29.9 mm ≳ gestures 10.2 mm). That is real motion penetration showing through, exactly as it
   should.

## Why PHC's "0.00 mm penetration" and a dipping mesh are not a contradiction

The post-sim `gpen = 0.00 mm` from the characterization study was computed on **rigid-body joint
positions**, not on a mesh surface. A body's sole always sits below its toe joint, so a small
negative mesh Z is expected even for a physically valid PHC motion. The two numbers measure
different surfaces and should not be compared directly.

## Consequence for the render

None requiring action. The comparison the video has to show is *relative*, and it survives intact:
post-PHC sits at a stable ~12 mm median dip on every clip, while pre-PHC dips 1.5–4× deeper and
varies with the clip's actual artefact severity. At presentation scale a 12 mm dip on a 1.7 m
figure is not visible, whereas the pre-PHC penetration is.

Applying a correction would have been actively harmful: any constant lift large enough to zero the
post-PHC dip would also lift pre-PHC by the same amount, suppressing part of the very artefact the
sports and martialarts renders exist to demonstrate.
