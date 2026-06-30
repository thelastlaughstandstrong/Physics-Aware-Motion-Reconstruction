# Results Comparison

## PHC Physics Evaluation — Primary Results

Full student (backbone + action head) in IsaacGym physics simulation on PHC test set (536 sequences).
Results saved to `phc/output/HumanoidIm/<EXP>/phc_eval.log`.

| Model | Success rate | MPJPE (cm) | MPJVE (cm/s) | MPJRE (°) | Jitter (m/s³) | Foot Skate (cm/s) |
|---|---|---|---|---|---|---|
| Distill recon+smooth (`recon2`) | 0.537 | 38.98 | 95.00 | 41.06 | 1236 | 2.33 |
| Distill recon+smooth+gpen (`recon3`) | 0.450 | 46.86 | 106.95 | 43.23 | 1488 | 2.72 |
| Distill frozen backbone (`frozen_bb`) | — | — | — | — | — | — |

> GPen is not reported — physics sim enforces non-penetration regardless of model quality.
> AvatarPoser kinematic baseline (for reference): MPJPE 2.51 cm, Jitter 1357 m/s³, Foot Skate 33.67 cm/s.

---

## Backbone Kinematic Quality (main_test, backbone only)

Measures backbone pose predictions against AMASS GT — independent of the physics sim.
With a frozen backbone this should always return ~2.51 cm.

| Model | Backbone MPJPE (cm) | Notes |
|---|---|---|
| AvatarPoser (pretrained) | 2.51 | baseline |
| recon2 (post-distill, Stage D+E) | **75.35** | catastrophic forgetting — λ_recon=1.0 insufficient once backbone unfrozen |
| frozen_bb | ~2.51 (expected) | backbone never unfrozen |

---

## Eval commands

```bash
# Step 1: run student in sim
conda run -n rlgpu --no-capture-output bash -c "
unset PYTHONPATH
cd /home/sia-irl/AvatarPoser/phc && python phc/run_hydra.py \
  learning=distill exp_name=<EXP> test=True im_eval=True \
  env=env_im_getup_mcp robot=smpl_humanoid robot.freeze_hand=True robot.box_body=False \
  env.motion_file=sample_data/phc_test.pkl \
  'env.models=[output/HumanoidIm/phc_kp_pnn_iccv/Humanoid.pth]' \
  env.num_envs=16 env.obs_v=7 headless=True \
  learning.params.config.student_checkpoint=output/HumanoidIm/<EXP>/student/<CKPT>.pth"

# Step 2: compute metrics
python scripts/evaluate_phc_metrics.py \
  --pos_dump phc/output/HumanoidIm/<EXP>/phc_positions_for_eval.pkl \
  --motion_file phc/sample_data/phc_test.pkl
```
