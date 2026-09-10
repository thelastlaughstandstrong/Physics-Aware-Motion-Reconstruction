#!/bin/bash
# SimXR no-vision ablation — EVALUATION on the 1072-clip held-out test set.
# Reproduces: Success Rate 0.7173507463, mpjpe_g 112.527 mm.
#
# WARNING: this writes phc_positions_for_eval.pkl into output/HumanoidIm/$EXP/, OVERWRITING any
# existing dump there. Set EXP to a fresh name to keep the current one. (This exact overwrite has
# already destroyed run records twice in this project.)
set -euo pipefail

SIMXR=${SIMXR:-/home/sia-irl/SimXR}
DATA=${DATA:-/home/sia-irl/AvatarPoser/data/quest2_synthetic}
CONDA=${CONDA:-/home/sia-irl/anaconda3/bin/conda}
ENV_NAME=${ENV_NAME:-isaac}
EXP=${EXP:-simxr_quest_novision_lrdecay}
EPOCH=${EPOCH:-1500}          # peak checkpoint -> Humanoid_00001500.pth (name + zfill(8))
LOG=${LOG:-$SIMXR/output/HumanoidIm/$EXP/eval_epoch${EPOCH}.log}

cd "$SIMXR"
"$CONDA" run -n "$ENV_NAME" --no-capture-output python phc/run_hydra.py \
  exp_name="$EXP" \
  env=env_simxr_quest learning=im_simxr robot=quest_humanoid \
  env.motion_file="$DATA/test_seg_motion/" \
  +env.pretrain=True \
  env.cycle_motion=False \
  epoch="$EPOCH" test=True im_eval=True \
  headless=True no_virtual_display=True no_log=True \
  2>&1 | tee "$LOG"

echo
echo "[eval] result:"; grep -E "Success Rate|^All:" "$LOG" | tail -2
