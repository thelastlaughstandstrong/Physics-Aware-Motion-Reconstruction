#!/bin/bash
# train_distill_quest_with_periodic_eval.sh — Phase-C quest_humanoid distillation training
# (distill_quest algo, DistillAgentQuest) with automatic OOM-crash recovery (ported from
# train_distill_quest_with_auto_recovery.sh) AND live periodic real-eval-based best-checkpoint
# tracking. Always trains to the full <max_distill_step>; periodic eval is purely for TRACKING
# the best checkpoint along the way, not for cutting training short.
#
# Exists because neither quest_distill_option2_scratch nor quest_distill_option3_actionloss had
# any periodic-eval safety net -- both trained to a fixed step count and were evaluated once,
# after the fact, on the final checkpoint only.
#
# Design note (2026-07-24): an earlier version of this script also killed training early via
# patience-based early stopping, mirroring train_stage_r_with_early_stop.sh's SMPL/Stage-R
# collapse-detection (which peaks once early then monotonically collapses). Retroactively
# scanning quest_distill_option2_scratch/quest_distill_option3_actionloss with that same
# patience=2 logic showed it does NOT transfer to this recipe: both runs' success_rate
# trajectory is noisy/non-monotonic early on (option3: 0.081 -> 0.159 -> 0.095 -> 0.146 over
# its first 40k steps), and the actual best checkpoint for BOTH runs turned out to be at or
# near their FINAL step -- patience=2 stopped searching at an early temporary dip long before
# reaching that. So this version never kills training: it runs the full budget every time, and
# periodic eval only decides what to report as "best," not when to stop.
#
# Usage:
#   bash scripts/train_distill_quest_with_periodic_eval.sh <exp_name> <max_distill_step> \
#        [num_envs] [eval_interval] [max_crash_retries] [gpu_free_mib_threshold] \
#        [extra_hydra_overrides]
#
# Mechanism:
#   - Launches training in the background (same recipe as train_distill_quest_with_auto_
#     recovery.sh: learning=distill_quest, env=env_quest_im_basic, robot=quest_humanoid,
#     env.motion_file=train_seg_motion).
#   - Every <eval_interval> steps, once a checkpoint at/past that step exists, runs a real
#     physics eval (scripts/auto_eval_quest.sh, on the held-out test_seg_motion set) CONCURRENTLY
#     with training still running (same pattern train_stage_r_with_early_stop.sh already uses
#     successfully for the SMPL pipeline -- see the "GPU contention" note in memory: occasional
#     eval crashes from shared-GPU contention are a known, recoverable failure mode there, not a
#     sign the concurrent-eval design itself is broken).
#   - Tracks the best success_rate_phc_survival seen so far, purely for reporting -- training is
#     never killed because of it.
#   - If training dies WITHOUT hitting max_distill_step, treats it as a crash (the documented
#     Isaac-Gym-internal env-stepping memory leak) and resumes from the latest checkpoint, up to
#     max_crash_retries -- preserving the best-checkpoint bookkeeping across the resume via
#     $EXP_DIR/.periodic_eval_state.
#   - Once training hits max_distill_step naturally, runs one more eval on the true final
#     checkpoint (it may not land exactly on an eval_interval boundary) before reporting the
#     best checkpoint found overall.
#
# Example:
#   bash scripts/train_distill_quest_with_periodic_eval.sh quest_distill_option4 300000

set -uo pipefail

EXP_NAME=${1:?Usage: train_distill_quest_with_periodic_eval.sh <exp_name> <max_distill_step> [num_envs] [eval_interval] [max_crash_retries] [gpu_free_mib_threshold] [extra_hydra_overrides]}
MAX_DISTILL_STEP=${2:?Usage: train_distill_quest_with_periodic_eval.sh <exp_name> <max_distill_step> [num_envs] [eval_interval] [max_crash_retries] [gpu_free_mib_threshold] [extra_hydra_overrides]}
NUM_ENVS=${3:-128}
EVAL_INTERVAL=${4:-10000}
MAX_RETRIES=${5:-10}
GPU_FREE_MIB_THRESHOLD=${6:-10000}
EXTRA_OVERRIDES=${7:-}

REPO=/home/sia-irl/AvatarPoser
EXP_DIR="$REPO/phc/output/HumanoidIm/$EXP_NAME"
RECOVERY_DIR="$REPO/phc/output/HumanoidIm/.recovery_logs/$EXP_NAME"
RECOVERY_LOG="$RECOVERY_DIR/auto_recovery.log"
STATE_FILE="$EXP_DIR/.periodic_eval_state"
MOTION_FILE="$REPO/data/quest2_synthetic/train_seg_motion"
PATTERN="run_hydra.py learning=distill_quest exp_name=$EXP_NAME "

mkdir -p "$EXP_DIR/student" "$RECOVERY_DIR"
echo "[periodic-eval] starting supervised run for $EXP_NAME (max $MAX_RETRIES crash retries, cap=$MAX_DISTILL_STEP steps, num_envs=$NUM_ENVS, eval_interval=$EVAL_INTERVAL, no early stopping)" | tee -a "$RECOVERY_LOG"

latest_checkpoint() {
    ls -1 "$EXP_DIR/student/"student_*.pth 2>/dev/null | sort | tail -1
}

wait_for_gpu() {
    local free_mib
    while true; do
        free_mib=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | head -1)
        if [ -n "$free_mib" ] && [ "$free_mib" -ge "$GPU_FREE_MIB_THRESHOLD" ]; then
            echo "[periodic-eval] GPU has ${free_mib}MiB free (>= ${GPU_FREE_MIB_THRESHOLD}MiB threshold) — proceeding" | tee -a "$RECOVERY_LOG"
            return
        fi
        echo "[periodic-eval] waiting for GPU: ${free_mib:-unknown}MiB free, need ${GPU_FREE_MIB_THRESHOLD}MiB — rechecking in 60s" | tee -a "$RECOVERY_LOG"
        sleep 60
    done
}

# checkpoint_at_or_past <step> — newest saved checkpoint whose step is >= <step>, or empty.
checkpoint_at_or_past() {
    local target=$1 f step
    ls -1 "$EXP_DIR/student/"student_*.pth 2>/dev/null | sort | while read -r f; do
        step=$(basename "$f" | sed -E 's/student_0*([0-9]+)\.pth/\1/')
        if [ "$step" -ge "$target" ]; then
            echo "$f"
            break
        fi
    done
}

BEST_SUCCESS="-1"
BEST_STEP=0
NEXT_CHECK=$EVAL_INTERVAL

if [ -f "$STATE_FILE" ]; then
    # shellcheck disable=SC1090
    source "$STATE_FILE"
    echo "[periodic-eval] restored bookkeeping: best=$BEST_SUCCESS @ $BEST_STEP, next_check=$NEXT_CHECK" | tee -a "$RECOVERY_LOG"
fi

save_state() {
    printf 'BEST_SUCCESS=%q\nBEST_STEP=%q\nNEXT_CHECK=%q\n' \
        "$BEST_SUCCESS" "$BEST_STEP" "$NEXT_CHECK" > "$STATE_FILE"
}
save_state

run_eval_check() {
    # Evaluates the checkpoint nearest <step_label>, updates BEST_SUCCESS/BEST_STEP if improved.
    local step_label=$1
    local tag="$RECOVERY_DIR/eval_${step_label}.log"
    echo "[periodic-eval] evaluating near step $step_label ..." | tee -a "$RECOVERY_LOG"
    bash "$REPO/scripts/auto_eval_quest.sh" "$EXP_NAME" "$step_label" > "$tag" 2>&1
    local metrics_log
    metrics_log=$(grep "^\[auto_eval_quest\] METRICS_LOG=" "$tag" | tail -1 | cut -d= -f2-)
    local success=""
    if [ -n "$metrics_log" ] && [ -f "$metrics_log" ]; then
        success=$(grep "success_rate_phc_survival" "$metrics_log" | awk -F: '{print $2}' | tr -d ' ')
    fi
    if [ -z "$success" ]; then
        echo "[periodic-eval] WARNING: could not read success_rate for check @ $step_label, see $tag" | tee -a "$RECOVERY_LOG"
        return
    fi
    echo "[periodic-eval] step ~$step_label success_rate=$success (best so far: $BEST_SUCCESS @ $BEST_STEP)" | tee -a "$RECOVERY_LOG"
    local improved
    improved=$(python3 -c "print(1 if float('$success') > float('$BEST_SUCCESS') else 0)")
    if [ "$improved" = "1" ]; then
        BEST_SUCCESS="$success"; BEST_STEP="$step_label"
    fi
}

ATTEMPT=0
while true; do
    ATTEMPT=$((ATTEMPT + 1))
    ATTEMPT_LOG="$RECOVERY_DIR/attempt_${ATTEMPT}.log"

    RESUME_CKPT=$(latest_checkpoint)
    RESUME_ARG=""
    if [ -z "$RESUME_CKPT" ]; then
        echo "[periodic-eval] attempt $ATTEMPT: no existing checkpoint -- fresh start" | tee -a "$RECOVERY_LOG"
    else
        echo "[periodic-eval] attempt $ATTEMPT: resuming from $RESUME_CKPT" | tee -a "$RECOVERY_LOG"
        RESUME_ARG="learning.params.config.student_checkpoint=$RESUME_CKPT"
    fi

    wait_for_gpu

    cd "$REPO/phc"
    conda run -n rlgpu --no-capture-output python phc/run_hydra.py \
        learning=distill_quest exp_name=$EXP_NAME \
        env=env_quest_im_basic robot=quest_humanoid \
        env.motion_file=$MOTION_FILE \
        env.num_envs=$NUM_ENVS env.cycle_motion=True env.min_length=5 \
        learning.params.config.max_distill_step=$MAX_DISTILL_STEP \
        $RESUME_ARG \
        headless=True no_virtual_display=True no_log=True disable_multiprocessing=False \
        test=False im_eval=False \
        $EXTRA_OVERRIDES \
        > "$ATTEMPT_LOG" 2>&1 &

    echo "[periodic-eval] training launched, attempt $ATTEMPT" | tee -a "$RECOVERY_LOG"

    while true; do
        sleep 30

        if ! pgrep -f "$PATTERN" > /dev/null; then
            echo "[periodic-eval] training process has exited" | tee -a "$RECOVERY_LOG"
            break
        fi

        if [ "$NEXT_CHECK" -gt "$MAX_DISTILL_STEP" ]; then
            continue  # no more scheduled checks -- let training run to its own cap
        fi

        CKPT=$(checkpoint_at_or_past "$NEXT_CHECK")
        if [ -z "$CKPT" ]; then
            continue  # training hasn't reached this checkpoint yet
        fi

        run_eval_check "$NEXT_CHECK"
        NEXT_CHECK=$((NEXT_CHECK + EVAL_INTERVAL))
        save_state
    done

    if grep -q "hit max_distill_step cap" "$ATTEMPT_LOG"; then
        echo "[periodic-eval] training completed normally (hit max_distill_step cap)" | tee -a "$RECOVERY_LOG"
        FINAL_CKPT=$(latest_checkpoint)
        if [ -n "$FINAL_CKPT" ]; then
            FINAL_STEP=$(basename "$FINAL_CKPT" | sed -E 's/student_0*([0-9]+)\.pth/\1/')
            run_eval_check "$FINAL_STEP"
            save_state
        fi
        echo "[periodic-eval] DONE. Best checkpoint: student_$(printf '%07d' $BEST_STEP).pth (success_rate=$BEST_SUCCESS)" | tee -a "$RECOVERY_LOG"
        exit 0
    fi

    echo "[periodic-eval] attempt $ATTEMPT ended without hitting max_distill_step — treating as a crash" | tee -a "$RECOVERY_LOG"
    tail -15 "$ATTEMPT_LOG" | tee -a "$RECOVERY_LOG"

    if [ "$ATTEMPT" -gt "$MAX_RETRIES" ]; then
        echo "[periodic-eval] exceeded max_crash_retries ($MAX_RETRIES) — giving up. Best so far: student_$(printf '%07d' $BEST_STEP).pth (success_rate=$BEST_SUCCESS)" | tee -a "$RECOVERY_LOG"
        exit 1
    fi

    echo "[periodic-eval] pausing 30s before retry $((ATTEMPT + 1))..." | tee -a "$RECOVERY_LOG"
    sleep 30
done
