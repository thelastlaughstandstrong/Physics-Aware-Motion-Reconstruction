#!/bin/bash
# auto_eval.sh — waits for a distill checkpoint then runs physics eval + metrics
#
# Usage:
#   bash scripts/auto_eval.sh <exp_name> <target_step> [extra_hydra_overrides]
#
# Example:
#   bash scripts/auto_eval.sh ap_phc_distill_v2 500000
#   bash scripts/auto_eval.sh ap_phc_distill_direct_pd 260000 "env.raw_action_passthrough=True"

set -e

EXP_NAME=${1:?Usage: auto_eval.sh <exp_name> <target_step> [extra_hydra_overrides]}
TARGET_STEP=${2:?Usage: auto_eval.sh <exp_name> <target_step> [extra_hydra_overrides]}
EXTRA_OVERRIDES=${3:-}

REPO=/home/sia-irl/AvatarPoser
CKPT_DIR=$REPO/phc/output/HumanoidIm/$EXP_NAME/student
CKPT=$CKPT_DIR/$(printf "student_%07d.pth" $TARGET_STEP)
POS_DUMP=$REPO/phc/output/HumanoidIm/$EXP_NAME/phc_positions_for_eval.pkl
MOTION_FILE=$REPO/phc/sample_data/phc_test.pkl

echo "[auto_eval] Waiting for $CKPT ..."
while [ ! -f "$CKPT" ]; do
    sleep 60
done
echo "[auto_eval] Checkpoint found. Starting physics eval..."

/home/sia-irl/anaconda3/bin/conda run -n rlgpu --no-capture-output bash -c "
unset PYTHONPATH
export WANDB_MODE=offline
cd $REPO/phc && python phc/run_hydra.py \
  learning=distill exp_name=$EXP_NAME test=True im_eval=True \
  env=env_im_getup_mcp robot=smpl_humanoid robot.freeze_hand=True robot.box_body=False \
  env.motion_file=sample_data/phc_test.pkl \
  'env.models=[output/HumanoidIm/phc_kp_pnn_iccv/Humanoid.pth]' \
  env.num_envs=16 env.obs_v=7 headless=True \
  learning.params.config.student_checkpoint=$CKPT $EXTRA_OVERRIDES"

echo "[auto_eval] Physics eval done. Computing metrics..."

/home/sia-irl/anaconda3/bin/conda run -n avatarposer --no-capture-output python $REPO/scripts/evaluate_phc_metrics.py \
    --pos_dump "$POS_DUMP" \
    --motion_file "$MOTION_FILE"

# Save a checkpoint-specific copy of the logs
STEP_TAG=$(printf "%07d" $TARGET_STEP)
EXP_DIR=$REPO/phc/output/HumanoidIm/$EXP_NAME
cp "$EXP_DIR/phc_eval.log"   "$EXP_DIR/phc_eval_${STEP_TAG}.log"   2>/dev/null && \
    echo "[auto_eval] Saved -> phc_eval_${STEP_TAG}.log"
cp "$EXP_DIR/qref_gpen.log"  "$EXP_DIR/qref_gpen_${STEP_TAG}.log"  2>/dev/null && \
    echo "[auto_eval] Saved -> qref_gpen_${STEP_TAG}.log"

echo "[auto_eval] Done."
