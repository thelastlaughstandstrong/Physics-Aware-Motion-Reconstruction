#!/bin/bash
# Step 6D — frozen pretrained SMPL PHC tracking the two validated Quest->SMPL motions.
#
# This is the repository's OWN documented frozen-teacher recipe (evaluate_phc_metrics.py's
# module docstring, the same one that produced the 536-sequence naive baseline in
# results_comparison.md). Only two things differ:
#   * env.motion_file  -> the Step 6C converted motions
#   * exp_name         -> a separate output dir, so the naive baseline's 273 MB dump in
#                         output/HumanoidIm/phc_kp_mcp_iccv/ is NOT overwritten.
#                         Humanoid.pth there is a symlink to that same known-good checkpoint.
#
# Nothing is trained or fine-tuned: learning=im_mcp, epoch=-1, test=True, im_eval=True.
set -e

REPO=/home/sia-irl/AvatarPoser
EXP=quest_smpl_frozen_phc
MOTION=$REPO/data/quest_smpl_retarget/quest_smpl_both_phc.pkl
LOG=$REPO/output/HumanoidIm/$EXP/rollout.log

echo "[6D] motion file : $MOTION"
echo "[6D] checkpoint  : output/HumanoidIm/phc_kp_pnn_iccv/Humanoid.pth (frozen)"
echo "[6D] output dir  : output/HumanoidIm/$EXP"

/home/sia-irl/anaconda3/bin/conda run -n rlgpu --no-capture-output bash -c "
unset PYTHONPATH && export WANDB_MODE=offline
cd $REPO/phc && python phc/run_hydra.py \
  learning=im_mcp exp_name=$EXP test=True \
  env=env_im_getup_mcp robot=smpl_humanoid robot.freeze_hand=True robot.box_body=False \
  env.z_activation=relu \
  'env.motion_file=$MOTION' \
  \"env.models=['output/HumanoidIm/phc_kp_pnn_iccv/Humanoid.pth']\" \
  env.num_envs=2 env.obs_v=7 headless=True epoch=-1 im_eval=True" 2>&1 | tee "$LOG"

echo "[6D] rollout done -> output/HumanoidIm/$EXP/phc_positions_for_eval.pkl"
