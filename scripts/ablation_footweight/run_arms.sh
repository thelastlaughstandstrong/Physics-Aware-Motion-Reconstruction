#!/bin/bash
# Foot-weight ablation: train the w=2.0 and w=4.0 arms SEQUENTIALLY.
#
# Baseline (w=1.0) is NOT retrained -- foot_pos_weight=1.0 is a bitwise no-op, so the existing
# quest_distill_trackerr run is the baseline arm (see PROTOCOL.md section 4).
#
# Uses train_distill_quest_with_auto_recovery.sh, NOT the periodic-eval variant: the latter runs a
# concurrent eval on the FULL test_seg_motion set, which would inspect the 701 held-out clips that
# PROTOCOL.md section 2 puts off-limits until a final candidate exists. Both scripts issue an
# identical training command; the only difference is the eval side-channel.
#
# Everything except foot_pos_weight is left at the baseline recipe: same script, same env/robot,
# same motion file, same num_envs, same seed (0, not overridden), no other Hydra override.
set -uo pipefail

REPO=/home/sia-irl/AvatarPoser
D=$REPO/scripts/ablation_footweight
MAX_STEP=${MAX_STEP:-400000}     # protocol evaluates 330k-400k; checkpoints in that range are
                                 # identical whether or not training continues past 400k
NUM_ENVS=${NUM_ENVS:-128}
BLOCK_PID=${BLOCK_PID:-}         # optional: wait for this PID (the unrelated 3DGS job) to exit

if [ -n "$BLOCK_PID" ]; then
  echo "[arms] waiting for PID $BLOCK_PID (unrelated GPU job) to exit ..."
  while kill -0 "$BLOCK_PID" 2>/dev/null; do sleep 60; done
  echo "[arms] PID $BLOCK_PID gone; GPU free: $(nvidia-smi --query-gpu=memory.free --format=csv,noheader)"
fi

run_arm () {
  local w=$1 exp=$2
  echo "=== [arms] $exp : foot_pos_weight=$w -> $MAX_STEP steps ==="
  mkdir -p "$REPO/output/HumanoidIm/$exp"
  # Hydra rewrites .hydra/ on every later invocation into this dir -- both quest_distill_trackerr
  # and SimXR lost their training records that way. Write the command where it cannot be clobbered.
  cat > "$REPO/output/HumanoidIm/$exp/RUN_CONFIG.md" <<INNER
# RUN_CONFIG — $exp

Foot-weighted kinematic-loss ablation arm. Written at launch; .hydra/ in this directory will be
overwritten by any later eval and is NOT the training record.

    bash scripts/train_distill_quest_with_auto_recovery.sh \\
        $exp $MAX_STEP $NUM_ENVS 10 10000 \\
        "learning.params.config.foot_pos_weight=$w"

- foot_pos_weight = **$w** (the ONLY deviation from the baseline recipe)
- baseline arm = quest_distill_trackerr (foot_pos_weight 1.0 is a bitwise no-op)
- seed 0 (config default, not overridden); env=env_quest_im_basic; robot=quest_humanoid
- motion_file = data/quest2_synthetic/train_seg_motion; num_envs=$NUM_ENVS
- lambda_action 0.1, lambda_kinematic 1.0, kinematic_root_weight 0.02 — all unchanged
- protocol: scripts/ablation_footweight/PROTOCOL.md (frozen before training)
INNER
  bash "$REPO/scripts/train_distill_quest_with_auto_recovery.sh" \
      "$exp" "$MAX_STEP" "$NUM_ENVS" 10 10000 \
      "learning.params.config.foot_pos_weight=$w"
  echo "=== [arms] $exp finished with status $? ==="
}

run_arm 2.0 quest_distill_footw2
run_arm 4.0 quest_distill_footw4
echo "[arms] both arms done"
