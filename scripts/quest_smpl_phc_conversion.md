# Step 6C — Quest→SMPL sequences in PHC motion format

```
========================================================================================
0. TRACED FORMAT (from the repo, not invented)
========================================================================================
  skeleton   /tmp/claude-1000/-home-sia-irl-AvatarPoser/198ae13f-92be-4147-9256-2b72f67cdfbf/scratchpad/smpl_humanoid_upright.xml
             (LocalRobot, upright_start=True — the tree humanoid.py builds at runtime,
              whose rest offsets are R_up @ smpl_offsets, not the static mjcf's)
  node order == MUJOCO: True   local_translation[0] = [-0.0018 -0.2233  0.0282]
  upright_start=True (cfg/robot/smpl_humanoid.yaml) -> post-multiply by inv(quat[0.5,0.5,0.5,0.5]), xyzw
  velocities: derived by PHC at load (SkeletonMotion + compute_motion_dof_vels), never stored, never copied from Quest
  [PASS] smpl_2_mujoco / mujoco_2_smpl are mutual inverses

  wrote data/quest_smpl_retarget/quest_smpl_standing_phc.pkl  (key 'quest_retarget_standing', 426 frames)
========================================================================================
STANDING  round trip: PHC file -> PHC loader -> reconstructed motion
========================================================================================
  frames 426 (source 426) | fps 30 (source 30) | duration 14.1667 s
  [PASS] standing: frame count preserved
  [PASS] standing: fps preserved
  [PASS] standing: zeroing the hand rotations moves no joint position (both are leaves)  — max 0.0e+00 m over all 24 joints
  joint-rotation discrepancy, fitted joints 1..21: mean 4.700e-08° max 2.415e-06°
  root-orientation discrepancy: mean 1.688e-07° max 1.708e-06°
  hands (deliberately zeroed by AMASS convention): max 0.000°
  [PASS] standing: local body rotations survive the round trip  — max 2.41e-06°
  [PASS] standing: root orientation survives the round trip  — max 1.71e-06°
  root-position discrepancy: mean 2.765e-05 mm max 6.060e-05 mm
  [PASS] standing: root translation survives the round trip (metres, native Z-up)  — max 6.06e-05 mm
  [PASS] standing: format invariant root_trans_offset == trans_orig + local_translation[0]  — max 2.98e-08 m
  joint-position discrepancy, all 24: mean 3.3572 mm max 40.9976 mm
    excluding the two hand leaves: mean 0.0739 mm max 0.1918 mm
    L_Hand 40.98 mm | R_Hand 37.96 mm — PHC's humanoid carries its own hand bone (40.8 / 38.0 mm from real SMPL's); both are leaves, so this moves no other joint
  [PASS] standing: joint positions preserved (excluding hand leaves)  — max 0.1918 mm
  [PASS] standing: hand offset is exactly the known skeleton difference, not a conversion error  — L 40.98 mm vs 40.8, R 37.96 mm vs 38.0
  |linear velocity| max 1.155 m/s | |angular velocity| max 2.192 rad/s | dof_vels (426, 23, 3)
  [PASS] standing: linear velocity is PHC's own derivative of the reconstructed positions at this file's fps  — max residual 0.00e+00 m/s
  [PASS] standing: velocities are physically plausible (not Quest-frame leftovers)
  [PASS] standing: velocities/dof_vels finite
  [PASS] standing: no NaN/Inf anywhere in the reconstructed motion
  head-above-pelvis in Z: min 0.573 m | Z extent 1.508 m
  [PASS] standing: native Z-up preserved — head above pelvis every frame  — min 0.573 m
  L/R hand lateral separation: mean +456.0 mm; L_Hand-to-L_Wrist 85.7 mm, R_Hand-to-R_Wrist 85.4 mm
  [PASS] standing: left/right ordering preserved (hips, shoulders, hands)
  temporal continuity: max per-frame joint step 40.5 mm (median 3.3 mm), 0 discontinuities
  [PASS] standing: no temporal discontinuities introduced by the conversion

  wrote data/quest_smpl_retarget/quest_smpl_walking_phc.pkl  (key 'quest_retarget_walking', 427 frames)
========================================================================================
WALKING  round trip: PHC file -> PHC loader -> reconstructed motion
========================================================================================
  frames 427 (source 427) | fps 30 (source 30) | duration 14.2000 s
  [PASS] walking: frame count preserved
  [PASS] walking: fps preserved
  [PASS] walking: zeroing the hand rotations moves no joint position (both are leaves)  — max 0.0e+00 m over all 24 joints
  joint-rotation discrepancy, fitted joints 1..21: mean 4.933e-08° max 2.091e-06°
  root-orientation discrepancy: mean 1.201e-07° max 1.708e-06°
  hands (deliberately zeroed by AMASS convention): max 0.000°
  [PASS] walking: local body rotations survive the round trip  — max 2.09e-06°
  [PASS] walking: root orientation survives the round trip  — max 1.71e-06°
  root-position discrepancy: mean 2.795e-05 mm max 1.185e-04 mm
  [PASS] walking: root translation survives the round trip (metres, native Z-up)  — max 1.18e-04 mm
  [PASS] walking: format invariant root_trans_offset == trans_orig + local_translation[0]  — max 2.24e-07 m
  joint-position discrepancy, all 24: mean 3.3569 mm max 41.0070 mm
    excluding the two hand leaves: mean 0.0737 mm max 0.1933 mm
    L_Hand 40.96 mm | R_Hand 37.99 mm — PHC's humanoid carries its own hand bone (40.8 / 38.0 mm from real SMPL's); both are leaves, so this moves no other joint
  [PASS] walking: joint positions preserved (excluding hand leaves)  — max 0.1933 mm
  [PASS] walking: hand offset is exactly the known skeleton difference, not a conversion error  — L 40.96 mm vs 40.8, R 37.99 mm vs 38.0
  |linear velocity| max 3.533 m/s | |angular velocity| max 5.067 rad/s | dof_vels (427, 23, 3)
  [PASS] walking: linear velocity is PHC's own derivative of the reconstructed positions at this file's fps  — max residual 0.00e+00 m/s
  [PASS] walking: velocities are physically plausible (not Quest-frame leftovers)
  [PASS] walking: velocities/dof_vels finite
  [PASS] walking: no NaN/Inf anywhere in the reconstructed motion
  head-above-pelvis in Z: min 0.565 m | Z extent 1.545 m
  [PASS] walking: native Z-up preserved — head above pelvis every frame  — min 0.565 m
  L/R hand lateral separation: mean +831.4 mm; L_Hand-to-L_Wrist 85.7 mm, R_Hand-to-R_Wrist 85.4 mm
  [PASS] walking: left/right ordering preserved (hips, shoulders, hands)
  temporal continuity: max per-frame joint step 135.9 mm (median 36.1 mm), 0 discontinuities
  [PASS] walking: no temporal discontinuities introduced by the conversion
========================================================================================
SCHEMA CONFORMANCE vs sample_data/avatarposer_all.pkl
========================================================================================
  field               reference                         ours
  pose_quat_global    ndarray (65, 24, 4) float64       ndarray (426, 24, 4) float64
  pose_quat           ndarray (65, 24, 4) float32       ndarray (426, 24, 4) float32
  trans_orig          ndarray (65, 3) float32           ndarray (426, 3) float32
  root_trans_offset   Tensor (65, 3) float32            Tensor (426, 3) float32
  beta                ndarray (16,) float64             ndarray (16,) float64
  gender              str neutral                       str neutral
  pose_aa             ndarray (65, 72) float64          ndarray (426, 72) float64
  fps                 float 30.0                        float 30.0
  [PASS] all reference fields present with matching types/dtypes
========================================================================================
```
