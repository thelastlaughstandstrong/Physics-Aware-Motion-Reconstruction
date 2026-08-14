#!/bin/bash
# run_stage_r_after_scan.sh — chains Quest2 Stage R onto a finished Stage E run.
#
# Waits for an in-flight quest_checkpoint_scan.sh (by exact PID, never by pgrep pattern -- a
# pattern containing the exp_name matches this script's OWN command line and would hang forever),
# reads the winning Stage E checkpoint out of that scan's log, seeds Stage R from it, then
# fine-grain scans the Stage R range afterwards.
#
# The trailing fine scan is not optional bookkeeping: Quest2/SMPL Stage R peaks within a few
# thousand steps and then degrades, so a fixed end-of-run checkpoint is the wrong thing to report.
# Interval is 2000, far tighter than Stage E's 10000, for the same reason.
#
# Hyperparameters below are trial 2's (`..._stager_v2`), not the yaml defaults: trial 1 ran the
# defaults and produced the worst MPJRE of any Quest2 row (39.11 deg). Both prior trials still
# traded rotation accuracy for noisy success, so lambda_kinematic_stage_r (the anchor that guards
# rotation) is the first lever to raise if that repeats here.
#
# Usage: bash scripts/run_stage_r_after_scan.sh <scan_pid> [stage_r_steps]

set -uo pipefail

SCAN_PID=${1:?Usage: run_stage_r_after_scan.sh <scan_pid> [stage_r_steps]}
STAGE_R_STEPS=${2:-30000}

REPO=/home/sia-irl/AvatarPoser
SRC_EXP=quest_distill_trackerr
DST_EXP=quest_distill_trackerr_stager
SCAN_LOG="$REPO/phc/output/HumanoidIm/$SRC_EXP/checkpoint_scan.log"
LOG="$REPO/phc/output/HumanoidIm/stage_r_chain.log"

say() { echo "[stage-r-chain] $(date '+%F %T') $*" | tee -a "$LOG"; }

# scan_pid=0 means "the scan already finished, start immediately". Do NOT pass $$ or any shell
# that is still alive -- this waits on liveness, so a still-running parent hangs it forever.
if [ "$SCAN_PID" = "0" ]; then
    say "scan_pid=0 -- scan already complete, proceeding immediately"
else
    say "waiting for scan PID $SCAN_PID to exit"
    while kill -0 "$SCAN_PID" 2>/dev/null; do sleep 60; done
    say "scan exited"
    sleep 90   # let Isaac Gym fully release the GPU before the next sim is created
fi

# The scan log's final line is: "[scan] DONE. Best checkpoint for <exp>: student_XXXXXXX.pth (success_rate=Y)"
BEST_LINE=$(grep -a "DONE. Best checkpoint" "$SCAN_LOG" | tail -1)
BEST_CKPT_NAME=$(echo "$BEST_LINE" | grep -oE "student_[0-9]+\.pth" | tail -1)
if [ -z "$BEST_CKPT_NAME" ]; then
    say "ERROR: could not parse best checkpoint from $SCAN_LOG -- last line was: $BEST_LINE"
    exit 1
fi
BEST_CKPT="$REPO/phc/output/HumanoidIm/$SRC_EXP/student/$BEST_CKPT_NAME"
if [ ! -f "$BEST_CKPT" ]; then
    # Older scan logs name a checkpoint at the rounded scan target (e.g. student_0380000.pth),
    # which never exists -- checkpoints land every save_freq steps. Fall back to the nearest
    # real file, the same one auto_eval_quest.sh actually evaluated.
    WANT=$(echo "$BEST_CKPT_NAME" | grep -oE "[0-9]+" | sed 's/^0*//')
    say "WARN: $BEST_CKPT_NAME does not exist (scan reported a rounded step); resolving nearest"
    BEST_CKPT=$(python3 -c "
import glob, os, re, sys
want = int(sys.argv[1])
cands = glob.glob(os.path.join(sys.argv[2], 'student_*.pth'))
step = lambda p: int(re.search(r'student_(\d+)\.pth', os.path.basename(p)).group(1))
print(min(cands, key=lambda p: abs(step(p) - want)) if cands else '')
" "$WANT" "$REPO/phc/output/HumanoidIm/$SRC_EXP/student")
    BEST_CKPT_NAME=$(basename "$BEST_CKPT" 2>/dev/null)
    say "resolved to $BEST_CKPT_NAME"
fi
[ -n "$BEST_CKPT" ] && [ -f "$BEST_CKPT" ] || { say "ERROR: could not resolve a real checkpoint"; exit 1; }

BEST_STEP=$(echo "$BEST_CKPT_NAME" | grep -oE "[0-9]+" | sed 's/^0*//')
MAX_STEP=$((BEST_STEP + STAGE_R_STEPS))
say "seeding Stage R from $BEST_CKPT_NAME (step $BEST_STEP); cap $MAX_STEP ($STAGE_R_STEPS Stage R steps)"
say "$BEST_LINE"

OVERRIDES="learning.params.config.enable_stage_r=True \
learning.params.config.stage_r_lr=5e-6 \
learning.params.config.kl_threshold=0.03 \
learning.params.config.stage_r_entropy_coef=0.003 \
learning.params.config.stage_r_action_log_std_init=-3.6 \
learning.params.config.lambda_kinematic_stage_r=0.3"

cd "$REPO"
bash scripts/train_distill_quest_with_auto_recovery.sh \
    "$DST_EXP" "$MAX_STEP" 128 10 10000 "$OVERRIDES" "$BEST_CKPT" >> "$LOG" 2>&1
say "Stage R training finished"

sleep 90
say "fine-grain scanning Stage R range from $BEST_STEP at interval 2000"
bash scripts/quest_checkpoint_scan.sh "$DST_EXP" 2000 99 "$BEST_STEP" "$OVERRIDES" >> "$LOG" 2>&1
say "DONE -- Stage R scan complete"
grep -a "DONE. Best checkpoint" "$REPO/phc/output/HumanoidIm/$DST_EXP/checkpoint_scan.log" | tail -1 | tee -a "$LOG"
