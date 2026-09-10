#!/bin/bash
# SimXR no-vision ablation — TRAINING (reproduces the 71.7%-success checkpoint)
#
# Reconstructed from output/HumanoidIm/simxr_quest_novision_lrdecay/{train.log, ablation_results.md}.
# Hydra rewrites .hydra/ on every run in the same output dir, so the original training override
# record was overwritten by later evals -- this script IS the record. See README.md "Provenance".
#
# The two LR flags are COUPLED: rl_games' a2c_common.py:122 sets is_adaptive_lr from
# lr_schedule=='adaptive', then line 127 reads config['kl_threshold'] unconditionally. That key is
# absent from im_simxr.yaml, so 'adaptive' WITHOUT '+...kl_threshold' raises KeyError.
set -euo pipefail

SIMXR=${SIMXR:-/home/sia-irl/SimXR}
DATA=${DATA:-/home/sia-irl/AvatarPoser/data/quest2_synthetic}
CONDA=${CONDA:-/home/sia-irl/anaconda3/bin/conda}
ENV_NAME=${ENV_NAME:-isaac}
EXP=${EXP:-simxr_quest_novision_lrdecay}

# 6104 = 100,000,000 / (32 x 512) -- matches the paper's stated 0.1B-interaction budget.
MAX_EPOCHS=${MAX_EPOCHS:-6104}
NUM_ENVS=${NUM_ENVS:-512}

cd "$SIMXR"
"$CONDA" run -n "$ENV_NAME" --no-capture-output python phc/run_hydra.py \
  exp_name="$EXP" \
  env=env_simxr_quest learning=im_simxr robot=quest_humanoid \
  env.motion_file="$DATA/train_seg_motion/" \
  +env.pretrain=True \
  env.num_envs="$NUM_ENVS" \
  learning.params.config.lr_schedule=adaptive \
  +learning.params.config.kl_threshold=0.008 \
  learning.params.config.max_epochs="$MAX_EPOCHS" \
  headless=True no_virtual_display=True \
  2>&1 | tee "$SIMXR/output/HumanoidIm/$EXP/train.log"

echo "[train] done. Checkpoints every 500 epochs (save_frequency in im_simxr.yaml)."
echo "[train] The reported result uses Humanoid_00001500.pth -- the PEAK, not the final."
