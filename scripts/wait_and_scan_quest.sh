#!/bin/bash
# wait_and_scan_quest.sh — waits for a train_distill_quest_with_auto_recovery.sh run to reach a
# genuine, deliberate stop (not just a crash-then-retry), then automatically launches
# quest_checkpoint_scan.sh over it. Exists so a full-range retroactive best-checkpoint scan
# doesn't require someone to be watching the training run finish in real time.
#
# Distinguishes three outcomes by tailing the SAME auto_recovery.log the training wrapper
# itself writes to (see train_distill_quest_with_auto_recovery.sh):
#   - "run completed normally"          -> launch the scan
#   - "exceeded max_crash_retries"      -> give up, do NOT scan (nothing finished to scan)
#   - wrapper process died with neither message printed -> give up, do NOT scan (unexpected)
#
# Usage:
#   bash scripts/wait_and_scan_quest.sh <exp_name> [eval_interval] [patience] [start_step] \
#        [extra_hydra_overrides] [poll_seconds]
#
# Args after <exp_name> are forwarded verbatim to quest_checkpoint_scan.sh (same meaning/
# defaults as that script) except the last (poll_seconds, default 120), which only controls how
# often this script rechecks auto_recovery.log while waiting.

set -uo pipefail

EXP_NAME=${1:?Usage: wait_and_scan_quest.sh <exp_name> [eval_interval] [patience] [start_step] [extra_hydra_overrides] [poll_seconds]}
EVAL_INTERVAL=${2:-10000}
PATIENCE=${3:-999}
START_STEP=${4:-10000}
EXTRA_OVERRIDES=${5:-}
POLL_SECONDS=${6:-120}

REPO=/home/sia-irl/AvatarPoser
RECOVERY_LOG="$REPO/phc/output/HumanoidIm/.recovery_logs/$EXP_NAME/auto_recovery.log"

echo "[wait_and_scan] waiting for $EXP_NAME training to finish (polling $RECOVERY_LOG every ${POLL_SECONDS}s)"
while true; do
    if [ -f "$RECOVERY_LOG" ] && grep -q "run completed normally" "$RECOVERY_LOG"; then
        echo "[wait_and_scan] training completed normally -- proceeding to scan"
        break
    fi
    if [ -f "$RECOVERY_LOG" ] && grep -q "exceeded max_crash_retries" "$RECOVERY_LOG"; then
        echo "[wait_and_scan] training GAVE UP after exceeding max_crash_retries -- NOT launching scan. Check $RECOVERY_LOG."
        exit 1
    fi
    if ! pgrep -f "train_distill_quest_with_auto_recovery.sh $EXP_NAME " >/dev/null; then
        echo "[wait_and_scan] WARNING: wrapper process for $EXP_NAME is no longer running, but neither completion nor give-up message was found in $RECOVERY_LOG -- NOT launching scan (unexpected state, needs manual investigation)."
        exit 1
    fi
    sleep "$POLL_SECONDS"
done

echo "[wait_and_scan] launching: bash scripts/quest_checkpoint_scan.sh $EXP_NAME $EVAL_INTERVAL $PATIENCE $START_STEP \"$EXTRA_OVERRIDES\""
cd "$REPO"
bash scripts/quest_checkpoint_scan.sh "$EXP_NAME" "$EVAL_INTERVAL" "$PATIENCE" "$START_STEP" "$EXTRA_OVERRIDES"
echo "[wait_and_scan] scan finished."
