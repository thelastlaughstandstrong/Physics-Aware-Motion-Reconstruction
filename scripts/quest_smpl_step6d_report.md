# Step 6D — frozen PHC tracking the retargeted Quest motions

- checkpoint: `output/HumanoidIm/phc_kp_pnn_iccv/Humanoid.pth` (frozen, not fine-tuned)
- policy/config: `learning=im_mcp env=env_im_getup_mcp robot=smpl_humanoid env.obs_v=7 test=True im_eval=True epoch=-1`
- motions: `data/quest_smpl_retarget/quest_smpl_both_phc.pkl`
- overall success_rate reported by PHC: **1.000**


## quest_retarget_walking

- reference frames **426** (PHC drops the source's final frame: 427 -> 426), simulated frames **426**, survival ratio **1.000**
- survived: **YES**, early termination: **NO** — tracked to the end of the clip

| metric | value |
|---|---|
| MPJPE (22 joints) | **3.29 cm** |
| — root drift | 2.45 cm |
| — root-relative pose error | 2.63 cm |
| MPJRE (global, 24 joints) | **14.21°** |
| MPJVE | 13.95 cm/s |
| jitter | 71.4 m/s³ |
| foot skate | 0.72 cm/s |
| ground penetration (mean / max / rate) | 0.00 mm / 0.00 mm / 0.000 |
| root trajectory error (mean / max) | 2.45 cm / 6.85 cm |

Simulated trajectory saved: `data/quest_smpl_retarget/step6d_sim/quest_retarget_walking_sim.pkl`

## quest_retarget_standing

- reference frames **425** (PHC drops the source's final frame: 426 -> 425), simulated frames **425**, survival ratio **1.000**
- survived: **YES**, early termination: **NO** — tracked to the end of the clip

| metric | value |
|---|---|
| MPJPE (22 joints) | **3.66 cm** |
| — root drift | 2.56 cm |
| — root-relative pose error | 2.81 cm |
| MPJRE (global, 24 joints) | **19.76°** |
| MPJVE | 4.01 cm/s |
| jitter | 9.1 m/s³ |
| foot skate | 0.16 cm/s |
| ground penetration (mean / max / rate) | 0.00 mm / 0.00 mm / 0.000 |
| root trajectory error (mean / max) | 2.56 cm / 3.63 cm |

Simulated trajectory saved: `data/quest_smpl_retarget/step6d_sim/quest_retarget_standing_sim.pkl`

## Refinement check — pre-sim (retargeted reference) vs post-sim (frozen PHC)

| sequence | metric | pre-sim | post-sim | |
|---|---|---|---|---|
| walking | ground penetration mean (mm) | 3.913 | 0.000 | improved |
| walking | ground penetration max (mm) | 11.288 | 0.000 | improved |
| walking | ground penetration rate | 0.061 | 0.000 | improved |
| walking | foot skate (cm/s) | 2.406 | 0.716 | improved |
| walking | jitter (m/s^3) | 35.314 | 71.366 | WORSE |
| standing | ground penetration mean (mm) | 0.000 | 0.000 | unchanged |
| standing | ground penetration max (mm) | 0.000 | 0.000 | unchanged |
| standing | ground penetration rate | 0.000 | 0.000 | unchanged |
| standing | foot skate (cm/s) | 0.492 | 0.161 | improved |
| standing | jitter (m/s^3) | 10.784 | 9.104 | improved |
