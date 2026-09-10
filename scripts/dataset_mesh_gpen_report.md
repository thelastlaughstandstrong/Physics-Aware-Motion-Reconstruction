# Full-dataset mesh-level Ground Penetration — Quest2 test set

Metric: `utils.physical_metrics.compute_ground_penetration`, unchanged — GPen Mean is the
mean depth over **penetrating (vertex, frame) pairs only**; GPen Rate is the fraction of
**frames** with at least one vertex below the floor; tolerance 1e-4 m; floor Z = 0;
SMPL-H neutral, betas = 0, 6890 vertices. Every stream goes through the identical
Quest→SMPL-H position IK, the same vertical calibration, and no per-method alignment.

Aggregation is **equal-weight per clip**. Pooled frame-weighted values appear only as a
secondary diagnostic.

## Main aggregate

| Method | Stage | Clips | GPen Mean (mm) | Median | GPen Rate | % clips penetrating | IK residual (mm) |
|---|---|---|---|---|---|---|---|
| Ground Truth (retarget control) | pre | 1072 | **1.72** | 0.00 | **0.017** | 15.2% | 40.0 ± 5.2 |
| Phase A / Kinematics Only | pre | 1072 | **3.27** | 0.00 | **0.018** | 16.7% | 40.1 ± 5.3 |
| Finetuned Backbone (q_base) | pre | 1072 | **4.48** | 0.00 | **0.024** | 19.4% | 39.3 ± 5.0 |
| Finetuned Backbone | post | 1072 | **6.03** | 0.00 | **0.041** | 47.3% | 40.4 ± 5.1 |
| Naive PHC Baseline | post | 1072 | **2.56** | 0.00 | **0.020** | 21.5% | 40.5 ± 5.1 |
| SimXR no-vision | post | 1072 | **3.30** | 0.00 | **0.031** | 31.1% | 40.2 ± 3.6 |

Spread (equal-weight per clip):

| Method | GPen Mean sd | GPen Mean IQR | GPen Rate median | mean GPen Max (mm) |
|---|---|---|---|---|
| Ground Truth (retarget control) | 5.80 | 0.00 | 0.000 | 5.04 |
| Phase A / Kinematics Only | 12.23 | 0.00 | 0.000 | 10.65 |
| Finetuned Backbone (q_base) | 16.50 | 0.00 | 0.000 | 13.35 |
| Finetuned Backbone | 9.96 | 8.54 | 0.000 | 19.79 |
| Naive PHC Baseline | 7.20 | 0.00 | 0.000 | 8.11 |
| SimXR no-vision | 7.66 | 2.73 | 0.000 | 9.62 |

## By motion category (equal-weight per clip)

| Method | upper-body GPen mean / rate | lower-body GPen mean / rate | full-body/dynamic GPen mean / rate |
|---|---|---|---|
| Ground Truth (retarget control) | 0.03 / 0.000 | 0.75 / 0.003 | 5.31 / 0.056 |
| Phase A / Kinematics Only | 0.11 / 0.001 | 1.29 / 0.005 | 10.20 / 0.059 |
| Finetuned Backbone (q_base) | 0.41 / 0.004 | 1.16 / 0.007 | 14.19 / 0.073 |
| Finetuned Backbone | 3.16 / 0.008 | 7.74 / 0.036 | 10.32 / 0.102 |
| Naive PHC Baseline | 0.15 / 0.001 | 2.47 / 0.008 | 6.56 / 0.064 |
| SimXR no-vision | 0.80 / 0.003 | 4.10 / 0.026 | 6.81 / 0.081 |

## By activity (largest categories)

| Method | gestures | whiteboard | walking | jogging | standing | ROM | interactions | sports |
|---|---|---|---|---|---|---|---|---|
| Ground Truth (retarget control) | 0.0 | 0.0 | 0.0 | 1.5 | 0.0 | 3.6 | 1.3 | 8.3 |
| Phase A / Kinematics Only | 0.0 | 0.0 | 0.4 | 2.3 | 0.0 | 7.8 | 4.1 | 10.6 |
| Finetuned Backbone (q_base) | 0.3 | 0.3 | 0.2 | 2.2 | 0.0 | 11.1 | 9.1 | 13.3 |
| Finetuned Backbone | 4.0 | 1.6 | 5.5 | 10.1 | 0.2 | 5.4 | 10.4 | 16.1 |
| Naive PHC Baseline | 0.1 | 0.0 | 0.7 | 4.3 | 0.0 | 4.8 | 2.3 | 10.5 |
| SimXR no-vision | 0.5 | 0.9 | 2.2 | 6.1 | 0.2 | 5.1 | 4.9 | 9.3 |

## Pre- vs post-physics (same mesh metric, stages labelled)

| Comparison | pre-sim GPen mean | post-sim GPen mean | change |
|---|---|---|---|
| Finetuned Backbone | 4.48 | 6.03 | +1.55 mm (increased) |
| Naive PHC (Phase A → frozen teacher) | 3.27 | 2.56 | -0.71 mm (reduced) |
| SimXR no-vision | N/A (no intermediate pose by design) | 3.30 | — |
