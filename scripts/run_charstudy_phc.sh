#!/bin/bash
# Characterization study, stage 2: frozen-PHC rollout over all study clips in one pass.
# Identical recipe to Step 6D (the repo's documented frozen-teacher eval); only the motion file,
# exp_name and num_envs differ. Nothing is trained or fine-tuned.
set -e
REPO=/home/sia-irl/AvatarPoser
EXP=quest_smpl_charstudy
MOTION=$REPO/data/quest_smpl_retarget/charstudy/charstudy_all_phc.pkl
N=$(/home/sia-irl/anaconda3/envs/avatarposer/bin/python -c "import joblib;print(len(joblib.load('$MOTION')))")
echo "[charstudy] $N clips -> $EXP"
/home/sia-irl/anaconda3/bin/conda run -n rlgpu --no-capture-output bash -c "
unset PYTHONPATH && export WANDB_MODE=offline
cd $REPO/phc && python phc/run_hydra.py \
  learning=im_mcp exp_name=$EXP test=True \
  env=env_im_getup_mcp robot=smpl_humanoid robot.freeze_hand=True robot.box_body=False \
  env.z_activation=relu \
  'env.motion_file=$MOTION' \
  \"env.models=['output/HumanoidIm/phc_kp_pnn_iccv/Humanoid.pth']\" \
  env.num_envs=$N env.obs_v=7 headless=True epoch=-1 im_eval=True" 2>&1 | tr '\r' '\n' | grep -vE "^Terminated:" | tail -12
echo "[charstudy] rollout done"
