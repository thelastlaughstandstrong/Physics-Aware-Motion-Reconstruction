#!/bin/bash
# run_frozen_backbone_probe.sh — sequentially trains a FRESH state_enc/action_head against each of
# two frozen, already-distilled Quest2 backbones (finetuned_backbone's vs Option2's), for a
# matched step budget, then scans both retroactively.
#
# WHY: finetuned_backbone shows a real post-sim advantage over Option2 (31.7% vs 29.4% success)
# despite their backbones decoding to statistically identical pre-sim pose accuracy (~14.3 deg
# rotation error, matched) -- so the advantage isn't "better pose," it's something about the
# backbone's raw z representation that doesn't show up in pose accuracy. A direct raw-z comparison
# was inconclusive (basis misalignment between independently-trained networks), and a raw
# component swap ("frankenstein" hybrid checkpoints) failed completely in both directions for the
# same reason (each action_head only understands vectors in its OWN network's coordinate system).
#
# This sidesteps that: freeze_backbone=True means kinematic_loss can never move the backbone (it's
# frozen), so state_enc/action_head are trained completely fresh against a FIXED z, learning to
# interpret it from scratch -- no cross-network coordinate mismatch. If the run seeded from
# finetuned_backbone's frozen z reaches meaningfully higher success than the one seeded from
# Option2's, that cleanly localizes the advantage in the backbone's representation itself.
#
# Usage: bash scripts/run_frozen_backbone_probe.sh [max_distill_step] [num_envs]

set -uo pipefail

REPO=/home/sia-irl/AvatarPoser
MAX_STEP=${1:-50000}
NUM_ENVS=${2:-128}

FB_BACKBONE="$REPO/model_zoo/quest_backbone_finetuned_backbone_extracted.pth"
O2_BACKBONE="$REPO/model_zoo/quest_backbone_option2_extracted.pth"

LOG="$REPO/phc/output/HumanoidIm/frozen_backbone_probe_orchestration.log"
mkdir -p "$REPO/phc/output/HumanoidIm"
echo "[probe] starting sequential frozen-backbone probe pair, max_step=$MAX_STEP num_envs=$NUM_ENVS" | tee -a "$LOG"

echo "[probe] === Phase 1/4: training fresh heads against finetuned_backbone's frozen backbone ===" | tee -a "$LOG"
bash "$REPO/scripts/train_distill_quest_with_auto_recovery.sh" quest_distill_frozen_probe_fb "$MAX_STEP" "$NUM_ENVS" 10 10000 \
  "learning.params.config.freeze_backbone=True learning.params.config.avatarposer_checkpoint=$FB_BACKBONE" \
  2>&1 | tee -a "$LOG"

echo "[probe] === Phase 2/4: training fresh heads against option2's frozen backbone ===" | tee -a "$LOG"
bash "$REPO/scripts/train_distill_quest_with_auto_recovery.sh" quest_distill_frozen_probe_o2 "$MAX_STEP" "$NUM_ENVS" 10 10000 \
  "learning.params.config.freeze_backbone=True learning.params.config.avatarposer_checkpoint=$O2_BACKBONE" \
  2>&1 | tee -a "$LOG"

echo "[probe] === Phase 3/4: full-range retroactive scan of the finetuned_backbone-frozen run ===" | tee -a "$LOG"
bash "$REPO/scripts/quest_checkpoint_scan.sh" quest_distill_frozen_probe_fb 5000 999 5000 2>&1 | tee -a "$LOG"

echo "[probe] === Phase 4/4: full-range retroactive scan of the option2-frozen run ===" | tee -a "$LOG"
bash "$REPO/scripts/quest_checkpoint_scan.sh" quest_distill_frozen_probe_o2 5000 999 5000 2>&1 | tee -a "$LOG"

echo "[probe] ALL DONE." | tee -a "$LOG"
