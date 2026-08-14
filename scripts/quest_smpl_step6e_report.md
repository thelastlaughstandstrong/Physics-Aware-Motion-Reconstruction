# Step 6E — effect of the frozen-PHC physics pass

Analysis only: no IK rerun, no PHC rerun, no controller or reward change. Inputs are the Step 6D saved trajectories.

## standing

### Kinematic fidelity vs the Quest ground truth (the objective that matters)

Both sides scored against the same independent Quest targets (17 mapped joints, frame alignment -1, the known -64.2 mm fix_trans_height shift removed from both).

| | pre-sim (retargeted) | post-sim (PHC) | change | verdict |
|---|---|---|---|---|
| MPJPE vs Quest GT | **34.0 mm** | **50.7 mm** | +16.7 mm | **worsened** |

### Physical-motion quality (absolute, measurable on both)

| metric | pre-sim | post-sim | verdict |
|---|---|---|---|
| jitter (m/s³) | 10.784 | 9.104 | **improved** |
| foot skate (cm/s) | 0.492 | 0.161 | **improved** |
| ground penetration mean (mm) | 0.000 | 0.000 | **unchanged** |
| ground penetration max (mm) | 0.000 | 0.000 | **unchanged** |
| ground penetration rate | 0.000 | 0.000 | **unchanged** |

### Cost relative to the tracked reference (pre-sim is 0 by construction)

| metric | value | note |
|---|---|---|
| MPJPE | 3.66 cm | how far PHC moved from the retarget |
| — root/global displacement | 2.56 cm | 46% of squared error |
| — root-relative pose | 2.81 cm | articulated-pose change |
| MPJRE | 19.76° | decomposed below |
| MPJVE | 4.01 cm/s | |
| survival | 1.000 | no early termination |

### Rotation-error diagnosis — MPJRE 19.76°

Worst joints (mean°): R_Shoulder 43.9, L_Elbow 43.9, R_Wrist 43.3, L_Shoulder 42.3, R_Hand 37.5, L_Hand 37.5
Mean over jointed (non-leaf) bones: **twist 7.7°** about the bone axis vs **swing 14.7°** of bone direction (34% twist).

| joint | MPJRE | twist (bone axis) | swing (bone direction) | twist share |
|---|---|---|---|---|
| R_Shoulder | 43.9° | 4.8° | 22.6° | 18% |
| L_Elbow | 43.9° | 23.5° | 35.1° | 40% |
| R_Wrist | 43.3° | 9.5° | 35.9° | 21% |
| L_Shoulder | 42.3° | 8.9° | 20.7° | 30% |
| R_Hand (leaf) | 37.5° | — | — | inherits parent |
| L_Hand (leaf) | 37.5° | — | — | inherits parent |
| L_Wrist | 23.2° | 5.0° | 43.4° | 10% |
| Head (leaf) | 22.9° | — | — | inherits parent |

Arm chain: mean error rotation **27.2°** (a constant postural bias), residual scatter about it **20.6°** -- mostly a STATIC offset

### Jitter diagnosis — 10.8 → 9.1 m/s³ (0.84×)

- double support:  100% of frames | jitter   10.8 → **   9.1** (0.85×)
- single support:    0% of frames | jitter   15.3 → **   8.0** (0.52×)
- within ±2 frames of a contact change (2% of frames): 13.2 → **9.3** (0.71×); elsewhere 10.7 → **9.1** (0.85×)
- concentration: worst 1% of frames carry **56%** of the increase, worst 5% carry **76%** (uniform would be 1% / 5%)
- high-frequency power (>7.5 Hz, i.e. chatter): 16.8% → **39.2%** of jerk power
- root vs articulation: root jitter 13.3 → **5.4** (0.40×); root-relative 13.0 → **8.2** (0.63×)

## walking

### Kinematic fidelity vs the Quest ground truth (the objective that matters)

Both sides scored against the same independent Quest targets (17 mapped joints, frame alignment -1, the known -20.0 mm fix_trans_height shift removed from both).

| | pre-sim (retargeted) | post-sim (PHC) | change | verdict |
|---|---|---|---|---|
| MPJPE vs Quest GT | **35.6 mm** | **49.9 mm** | +14.2 mm | **worsened** |

### Physical-motion quality (absolute, measurable on both)

| metric | pre-sim | post-sim | verdict |
|---|---|---|---|
| jitter (m/s³) | 35.314 | 71.366 | **worsened** |
| foot skate (cm/s) | 2.406 | 0.716 | **improved** |
| ground penetration mean (mm) | 3.913 | 0.000 | **improved** |
| ground penetration max (mm) | 11.288 | 0.000 | **improved** |
| ground penetration rate | 0.061 | 0.000 | **improved** |

### Cost relative to the tracked reference (pre-sim is 0 by construction)

| metric | value | note |
|---|---|---|
| MPJPE | 3.29 cm | how far PHC moved from the retarget |
| — root/global displacement | 2.45 cm | 46% of squared error |
| — root-relative pose | 2.63 cm | articulated-pose change |
| MPJRE | 14.21° | decomposed below |
| MPJVE | 13.95 cm/s | |
| survival | 1.000 | no early termination |

### Rotation-error diagnosis — MPJRE 14.21°

Worst joints (mean°): L_Elbow 28.1, R_Shoulder 28.1, L_Shoulder 26.8, R_Wrist 21.6, R_Hand 19.9, L_Hand 19.9
Mean over jointed (non-leaf) bones: **twist 5.3°** about the bone axis vs **swing 10.8°** of bone direction (33% twist).

| joint | MPJRE | twist (bone axis) | swing (bone direction) | twist share |
|---|---|---|---|---|
| L_Elbow | 28.1° | 8.6° | 24.6° | 26% |
| R_Shoulder | 28.1° | 4.5° | 16.5° | 21% |
| L_Shoulder | 26.8° | 4.7° | 16.8° | 22% |
| R_Wrist | 21.6° | 4.9° | 18.7° | 21% |
| R_Hand (leaf) | 19.9° | — | — | inherits parent |
| L_Hand (leaf) | 19.9° | — | — | inherits parent |
| L_Knee | 18.7° | 8.9° | 7.0° | 56% |
| Head (leaf) | 17.9° | — | — | inherits parent |

Arm chain: mean error rotation **10.0°** (a constant postural bias), residual scatter about it **17.6°** -- mostly FLUCTUATION

### Jitter diagnosis — 35.3 → 71.4 m/s³ (2.02×)

- double support:   60% of frames | jitter   26.3 → **  47.2** (1.79×)
- single support:   39% of frames | jitter   48.7 → ** 107.3** (2.20×)
- flight:    1% of frames | jitter   47.8 → ** 110.4** (2.31×)
- within ±2 frames of a contact change (37% of frames): 52.5 → **100.0** (1.90×); elsewhere 25.1 → **54.3** (2.17×)
- concentration: worst 1% of frames carry **9%** of the increase, worst 5% carry **33%** (uniform would be 1% / 5%)
- high-frequency power (>7.5 Hz, i.e. chatter): 4.1% → **18.3%** of jerk power
- root vs articulation: root jitter 23.5 → **42.1** (1.79×); root-relative 33.2 → **65.2** (1.96×)
