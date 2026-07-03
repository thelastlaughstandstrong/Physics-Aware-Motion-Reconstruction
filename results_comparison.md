# Results Comparison

## PHC Physics Evaluation — Primary Results

Full student (backbone + action head) in IsaacGym physics simulation on PHC test set (536 sequences).
Results saved to `phc/output/HumanoidIm/<EXP>/phc_eval.log`.

| Model | Checkpoint | Success | MPJPE (cm) | MPJVE (cm/s) | MPJRE (°) | Jitter (m/s³) | Foot Skate (cm/s) |
|---|---|---|---|---|---|---|---|
| `recon2` — recon+smooth | 489.5k | 0.537 | 38.98 | 95.00 | 41.06 | 1236 | 2.33 |
| `recon3` — recon+smooth+gpen | 489.5k | 0.450 | 46.86 | 106.95 | 43.23 | 1488 | 2.72 |
| `frozen_bb` — frozen backbone | 489.5k | 0.584 | 34.83 | 89.39 | 38.73 | 1188 | 2.66 |
| `v2` — trainable heads, z.detach() | 1.95M | 0.455 | 50.71 | 113.59 | 45.02 | 1612 | 2.28 |
| `v3` — refinement head, no staged training | 431k | 0.440 | 47.18 | 105.24 | 43.07 | 1418 | 2.63 |
| **`v4` — staged training 1→2→3→E** | **750k** | **0.498** | **41.14** | **99.59** | **40.86** | **1345** | **2.51** |

> GPen is not reported — physics sim enforces non-penetration regardless of model quality.
> AvatarPoser kinematic baseline (for reference): MPJPE 2.51 cm, Jitter 1357 m/s³, Foot Skate 33.67 cm/s.
> v4 peaked at 750k; 1M regressed on all metrics.

---

## Backbone Kinematic Quality (main_test, backbone only)

Measures backbone pose predictions against AMASS GT — independent of the physics sim.

| Model | Backbone MPJPE (cm) | Notes |
|---|---|---|
| AvatarPoser (pretrained) | 2.51 | baseline |
| recon2 (post-distill) | 75.35 | catastrophic forgetting — λ_recon=1.0 insufficient once backbone unfrozen |
| frozen_bb | ~2.51 | backbone never modified |

---

## Eval commands

```bash
# Automated (watches for checkpoint, runs both steps):
bash scripts/auto_eval.sh <exp_name> <step>

# Manual 

#Step 1: run student in sim (rlgpu env, from phc/)
/home/sia-irl/anaconda3/bin/conda run -n rlgpu --no-capture-output bash -c "
unset PYTHONPATH && export WANDB_MODE=offline
cd /home/sia-irl/AvatarPoser/phc && python phc/run_hydra.py \
  learning=distill exp_name=<EXP> test=True im_eval=True \
  env=env_im_getup_mcp robot=smpl_humanoid robot.freeze_hand=True robot.box_body=False \
  env.motion_file=sample_data/phc_test.pkl \
  'env.models=[output/HumanoidIm/phc_kp_pnn_iccv/Humanoid.pth]' \
  env.num_envs=16 env.obs_v=7 headless=True \
  learning.params.config.student_checkpoint=output/HumanoidIm/<EXP>/student/<CKPT>.pth"

# Step 2: compute metrics (avatarposer env, from repo root)
python scripts/evaluate_phc_metrics.py \
  --pos_dump phc/output/HumanoidIm/<EXP>/phc_positions_for_eval.pkl \
  --motion_file phc/sample_data/phc_test.pkl
```
