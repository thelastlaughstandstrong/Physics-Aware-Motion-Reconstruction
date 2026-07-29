#!/bin/bash
# quest_checkpoint_scan.sh — retroactively walks an ALREADY-FINISHED quest_distill run's saved
# checkpoints at fixed intervals, tracking the best success_rate_phc_survival and stopping the
# scan (there's no live training process here) once <patience> consecutive checks fail to beat
# the best. Reconstructs, after the fact, the same "which checkpoint would live periodic eval
# have picked" answer that train_distill_quest_with_periodic_eval.sh gives future runs --
# built specifically for quest_distill_option2_scratch and quest_distill_option3_actionloss,
# neither of which had periodic eval running live during training (both only got a single
# end-of-run eval on the final checkpoint).
#
# Usage:
#   bash scripts/quest_checkpoint_scan.sh <exp_name> [eval_interval] [patience] [start_step] \
#        [extra_hydra_overrides]
#
# extra_hydra_overrides is forwarded verbatim to each auto_eval_quest.sh call -- REQUIRED for
# any run that wasn't trained with every learning.params.config default (e.g.
# "learning.params.config.explicit_pose_action_input=True"), since without it the student gets
# rebuilt at the wrong architecture and fails to load the checkpoint's action_head/value_head
# (shape mismatch), just like auto_eval_quest.sh itself.

set -uo pipefail

EXP_NAME=${1:?Usage: quest_checkpoint_scan.sh <exp_name> [eval_interval] [patience] [start_step] [extra_hydra_overrides]}
EVAL_INTERVAL=${2:-10000}
PATIENCE=${3:-2}
NEXT_CHECK=${4:-$EVAL_INTERVAL}
EXTRA_OVERRIDES=${5:-}

REPO=/home/sia-irl/AvatarPoser
EXP_DIR="$REPO/phc/output/HumanoidIm/$EXP_NAME"
SCAN_LOG="$EXP_DIR/checkpoint_scan.log"

MAX_STEP=$(ls -1 "$EXP_DIR/student/"student_*.pth 2>/dev/null | sed -E 's#.*student_0*([0-9]+)\.pth#\1#' | sort -n | tail -1)
if [ -z "$MAX_STEP" ]; then
    echo "[scan] ERROR: no checkpoints found for $EXP_NAME in $EXP_DIR/student/"
    exit 1
fi
echo "[scan] $EXP_NAME: scanning up to step $MAX_STEP, every $EVAL_INTERVAL steps, patience=$PATIENCE, starting at $NEXT_CHECK" | tee -a "$SCAN_LOG"

BEST_SUCCESS="-1"
BEST_STEP=0
FAILS=0

while [ "$NEXT_CHECK" -le "$MAX_STEP" ]; do
    echo "[scan] evaluating near step $NEXT_CHECK ..." | tee -a "$SCAN_LOG"
    EVAL_OUT="$EXP_DIR/scan_eval_${NEXT_CHECK}.log"
    bash "$REPO/scripts/auto_eval_quest.sh" "$EXP_NAME" "$NEXT_CHECK" "$EXTRA_OVERRIDES" > "$EVAL_OUT" 2>&1
    METRICS_LOG=$(grep "^\[auto_eval_quest\] METRICS_LOG=" "$EVAL_OUT" | tail -1 | cut -d= -f2-)
    SUCCESS=""
    if [ -n "$METRICS_LOG" ] && [ -f "$METRICS_LOG" ]; then
        SUCCESS=$(grep "success_rate_phc_survival" "$METRICS_LOG" | awk -F: '{print $2}' | tr -d ' ')
    fi

    if [ -z "$SUCCESS" ]; then
        echo "[scan] WARNING: could not read success_rate near step $NEXT_CHECK, see $EVAL_OUT" | tee -a "$SCAN_LOG"
        NEXT_CHECK=$((NEXT_CHECK + EVAL_INTERVAL))
        continue
    fi

    echo "[scan] step ~$NEXT_CHECK success_rate=$SUCCESS (best so far: $BEST_SUCCESS @ $BEST_STEP)" | tee -a "$SCAN_LOG"
    IMPROVED=$(python3 -c "print(1 if float('$SUCCESS') > float('$BEST_SUCCESS') else 0)")
    if [ "$IMPROVED" = "1" ]; then
        BEST_SUCCESS="$SUCCESS"; BEST_STEP="$NEXT_CHECK"; FAILS=0
    else
        FAILS=$((FAILS + 1))
        echo "[scan] no improvement ($FAILS/$PATIENCE)" | tee -a "$SCAN_LOG"
    fi

    if [ "$FAILS" -ge "$PATIENCE" ]; then
        echo "[scan] patience exceeded — stopping scan early (would have stopped training here)." | tee -a "$SCAN_LOG"
        break
    fi
    NEXT_CHECK=$((NEXT_CHECK + EVAL_INTERVAL))
done

echo "[scan] DONE. Best checkpoint for $EXP_NAME: student_$(printf '%07d' $BEST_STEP).pth (success_rate=$BEST_SUCCESS)" | tee -a "$SCAN_LOG"
