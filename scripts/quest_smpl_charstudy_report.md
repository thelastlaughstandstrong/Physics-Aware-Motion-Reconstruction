# Characterization study — should PHC physics be applied selectively?

10 clips, frozen `phc_kp_pnn_iccv` checkpoint, Step 6E protocol (pre- and post-physics both scored against Quest ground truth).

## Per-clip results

| clip | class | frames | surv | MPJPE pre→post (mm) | jitter pre→post | HF%% pre→post | skate pre→post (cm/s) | penetration pre→post (mm) |
|---|---|---|---|---|---|---|---|---|
| sports | dynamic | 417 | Y | 43.0 → 85.3 (**+42.3**) | 112.9 → 208.2 (1.84×) | 5.1 → 24.6 | 27.93 → 1.03 | 24.49 → 0.00 |
| martialarts | dynamic | 444 | Y | 43.5 → 91.8 (**+48.2**) | 64.4 → 193.0 (2.99×) | 3.5 → 27.8 | 14.37 → 0.71 | 29.92 → 0.00 |
| stretching | transition | 438 | Y | 51.1 → 91.4 (**+40.3**) | 19.3 → 20.4 (1.06×) | 7.7 → 11.2 | 1.54 → 0.69 | 46.48 → 0.00 |
| dancing | dynamic | 428 | Y | 35.3 → 49.1 (**+13.8**) | 57.5 → 101.5 (1.76×) | 5.2 → 25.2 | 16.00 → 0.95 | 9.58 → 0.00 |
| jogging | locomotion | 435 | Y | 46.8 → 75.9 (**+29.1**) | 193.9 → 215.4 (1.11×) | 4.8 → 19.8 | 9.46 → 0.65 | 16.35 → 0.00 |
| gestures | upper-body | 423 | Y | 43.7 → 62.9 (**+19.2**) | 64.0 → 164.2 (2.56×) | 6.5 → 37.4 | 5.88 → 0.95 | 10.18 → 0.00 |
| whiteboard | upper-body | 424 | Y | 40.1 → 54.4 (**+14.3**) | 48.6 → 50.4 (1.04×) | 56.2 → 23.4 | 1.53 → 0.43 | 12.37 → 0.00 |
| walking | locomotion | 426 | Y | 36.5 → 49.4 (**+12.9**) | 35.3 → 71.2 (2.02×) | 4.1 → 17.9 | 2.41 → 0.72 | 3.91 → 0.00 |
| standing | static | 425 | Y | 34.5 → 50.5 (**+16.0**) | 10.8 → 8.9 (0.83×) | 16.8 → 39.6 | 0.49 → 0.16 | 0.00 → 0.00 |
| rom | transition | 435 | Y | 35.1 → 52.3 (**+17.2**) | 15.4 → 14.0 (0.91×) | 6.4 → 7.8 | 0.07 → 0.16 | 0.00 → 0.00 |

## Benefit vs cost

Each term normalised by its across-clip pre-sim mean, so all four are dimensionless relative changes. Positive net = physics worth applying.

| clip | pre-sim violation | benefit | cost | **net** | verdict |
|---|---|---|---|---|---|
| sports | 5.10 (high) | +4.97 | +2.57 | **+2.41** | **beneficial** |
| martialarts | 3.76 (high) | +3.67 | +3.24 | **+0.42** | **beneficial** |
| stretching | 3.23 (high) | +3.14 | +1.00 | **+2.14** | **beneficial** |
| dancing | 2.63 (high) | +2.51 | +1.04 | **+1.47** | **beneficial** |
| jogging | 2.25 (high) | +2.17 | +1.06 | **+1.12** | **beneficial** |
| gestures | 1.40 (low) | +1.28 | +2.08 | **-0.79** | **harmful** |
| whiteboard | 1.00 (low) | +0.94 | +0.38 | **+0.57** | **beneficial** |
| walking | 0.56 (low) | +0.47 | +0.89 | **-0.43** | **harmful** |
| standing | 0.06 (low) | +0.04 | +0.36 | **-0.32** | **harmful** |
| rom | 0.01 (low) | -0.01 | +0.40 | **-0.41** | **harmful** |

## Low- vs high-violation groups

| group | n | pre-sim violation | ΔMPJPE (mm) | Δjitter (×) | Δskate (cm/s) | Δpenetration (mm) | **net** |
|---|---|---|---|---|---|---|---|
| low | 5 | 0.61 | +15.9 | 1.47 | -1.59 | -5.29 | **-0.28** |
| high | 5 | 3.39 | +34.7 | 1.75 | -13.05 | -25.36 | **+1.51** |

## Correlations

- pre-sim violation vs benefit: **r = +1.000** — this is TAUTOLOGICAL, not a finding: physics drives penetration to exactly 0 and skate below 1 cm/s on every clip, so the reduction is mechanically equal to the pre-existing violation. It is reported only to make that circularity explicit.
- pre-sim violation vs ΔMPJPE (fidelity cost): **r = +0.831**
- pre-sim violation vs net: **r = +0.811**
- net-positive clips have violation ≥ 1.00; net-negative ≤ 1.40 → **the groups overlap; no clean threshold**

## Fidelity-first view (the project's stated priority 1)

ΔMPJPE vs Quest GT is **positive on 10 of 10 clips** (range +12.9 to +48.2 mm, mean +25.3 mm / +60%).

**There is no clip on which physics improves kinematic fidelity.** Any recommendation to run it therefore rests entirely on valuing contact realism above pose accuracy; the composite `net` above encodes one particular exchange rate (equal weight per normalised term) and should not be read as objective.

## Recommended clips for Blender visualization

| role | clip | source | net | why |
|---|---|---|---|---|
| clearly beneficial | **sports** | `SUBJECT070.sports.pkl_0_417.pkl` | +2.41 | penetration 24.5→0.0 mm, skate 27.93→1.03, MPJPE +42.3 mm |
| clearly harmful | **gestures** | `SUBJECT070.gestures.pkl_0_423.pkl` | -0.79 | penetration 10.2→0.0 mm, skate 5.88→0.95, MPJPE +19.2 mm |
| mixed | **martialarts** | `SUBJECT059.martialarts.pkl_0_444.pkl` | +0.42 | penetration 29.9→0.0 mm, skate 14.37→0.71, MPJPE +48.2 mm |
