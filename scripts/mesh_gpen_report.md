# Mesh-level ground penetration — simulator-independent

Definition (imported unchanged from `utils/physical_metrics.compute_ground_penetration`, the
same function the AvatarPoser baseline uses):

- **GPen Mean (mm)** — mean depth over **penetrating (vertex, frame) pairs only**
- **GPen Rate** — fraction of **frames** with at least one vertex below the floor
- **GPen Max (mm)** — worst single-vertex depth
- tolerance 1e-4 m (0.1 mm) · floor `ground_z = 0.0` · SMPL-H **neutral**, betas = 0, 6890 verts

## Step 1+2 — diagnostic clips and the GT retargeting floor

| Clip | IK residual (mm) | GPen Mean (mm) | GPen Rate | GPen Max (mm) |
|---|---|---|---|---|
| SUBJECT059.martialarts_0_444 | 42.8 | 31.32 | 0.175 | 94.02 |
| SUBJECT070.sports_0_417 | 42.5 | 7.34 | 0.134 | 25.08 |
| SUBJECT070.gestures_0_423 | 44.2 | 0.00 | 0.000 | 0.00 |

## Step 3 — four-way comparison

| Motion | Method | IK Residual (mm) | GPen Mean (mm) | GPen Rate | GPen Max (mm) |
|---|---|---|---|---|---|
| gestures_427_853 (upper-body) | Ground Truth | 37.1 | 0.00 | 0.000 | 0.00 |
| gestures_427_853 (upper-body) | SimXR | 37.2 | 0.00 | 0.000 | 0.00 |
| gestures_427_853 (upper-body) | Finetuned Backbone | 38.9 | 0.00 | 0.000 | 0.00 |
| gestures_427_853 (upper-body) | Naive PHC Baseline | 38.3 | 0.00 | 0.000 | 0.00 |
| walking_0_412 (lower-body) | Ground Truth | 33.6 | 0.00 | 0.000 | 0.00 |
| walking_0_412 (lower-body) | SimXR | 35.9 | 1.60 | 0.002 | 2.52 |
| walking_0_412 (lower-body) | Finetuned Backbone | 33.7 | 5.77 | 0.015 | 17.40 |
| walking_0_412 (lower-body) | Naive PHC Baseline | 34.0 | 0.00 | 0.000 | 0.00 |
| ROM_3820_4243 (full-body) | Ground Truth | 40.4 | 18.13 | 0.106 | 48.24 |
| ROM_3820_4243 (full-body) | SimXR | 41.9 | 26.46 | 0.144 | 98.05 |
| ROM_3820_4243 (full-body) | Finetuned Backbone | 45.9 | 22.09 | 0.210 | 70.15 |
| ROM_3820_4243 (full-body) | Naive PHC Baseline | 37.5 | 25.22 | 0.066 | 58.78 |

## Step 6 — old simulator GPen vs new mesh GPen

| Motion | Method | Simulator GPen mean (mm) | Mesh GPen mean (mm) |
|---|---|---|---|
| gestures_427_853 (upper-body) | SimXR | 0.000 | 0.00 |
| gestures_427_853 (upper-body) | Finetuned Backbone | 0.000 | 0.00 |
| gestures_427_853 (upper-body) | Naive PHC Baseline | 0.000 | 0.00 |
| walking_0_412 (lower-body) | SimXR | 0.000 | 1.60 |
| walking_0_412 (lower-body) | Finetuned Backbone | 0.000 | 5.77 |
| walking_0_412 (lower-body) | Naive PHC Baseline | 0.000 | 0.00 |
| ROM_3820_4243 (full-body) | SimXR | 0.000 | 26.46 |
| ROM_3820_4243 (full-body) | Finetuned Backbone | 0.000 | 22.09 |
| ROM_3820_4243 (full-body) | Naive PHC Baseline | 0.000 | 25.22 |

## Fairness — IK residual spread per motion

| Motion | min | max | spread | flag |
|---|---|---|---|---|
| gestures_427_853 (upper-body) | 37.1 | 38.9 | 1.8 | none |
| walking_0_412 (lower-body) | 33.6 | 35.9 | 2.3 | none |
| ROM_3820_4243 (full-body) | 37.5 | 45.9 | 8.4 | none |
