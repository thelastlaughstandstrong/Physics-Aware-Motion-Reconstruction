#!/bin/bash
# train_stage_r_with_early_stop.sh — runs the FULL pipeline (Stage 1 -> E -> R) in one
# continuous, unattended launch, with periodic real evaluation and patience-based early
# stopping once Stage R begins.
#
# Exists because Stage R has a demonstrated failure mode: it peaks early (~10k steps in
# both tested architectures, v7 and v8) then regresses/collapses with continued training.
# Running to a fixed target step count either wastes compute past the peak or (worse)
# requires a human to notice the regression and kill training manually — which is what
# happened twice before this script existed. Separately: there is no need to train Stage E
# to completion, kill, and manually relaunch Stage R as a second process — DistillAgent's
# train() loop already transitions 1 -> E -> R automatically within one process as long as
# both stage_e_step and stage_r_step are set at launch, exactly like 1 -> E already did on
# its own. Splitting it into two manual launches (as v7 and v8 both did) was unnecessary.
#
# Usage:
#   bash scripts/train_stage_r_with_early_stop.sh <exp_name> <stage_e_step> <stage_r_step> \
#        [resume_checkpoint] [eval_interval] [patience] [max_step] [extra_hydra_overrides] [num_envs]
#
# Note: when resume_checkpoint is set, the exp dir is preserved (not wiped) and this
# script's own best/patience bookkeeping is reloaded from $EXP_DIR/.collapse_state if
# present -- this lets scripts/train_with_auto_recovery.sh resume a crashed run without
# losing track of a best checkpoint that was already found before the crash, or double
# counting patience. Only a genuinely fresh run (no resume_checkpoint) wipes the dir.
#
# Example — fresh run, Stage 1->E->R all in one launch (the normal case):
#   bash scripts/train_stage_r_with_early_stop.sh ap_phc_distill_v9 100000 250000
#
# Example — resume Stage R from an already-trained Stage E checkpoint (only needed for
# cases like v7/v8 where Stage E was already trained before Stage R existed/was decided on):
#   bash scripts/train_stage_r_with_early_stop.sh ap_phc_distill_v9_stager 100000 250000 \
#        /home/sia-irl/AvatarPoser/phc/output/HumanoidIm/ap_phc_distill_v9/student/student_0250000.pth
#
# Mechanism:
#   - Launches training with both stage_e_step and stage_r_step set — the whole 1->E->R
#     sequence runs automatically in one process, no manual restart at any stage boundary.
#   - Every <eval_interval> steps (default 10000) *after* stage_r_step, runs the real
#     auto_eval.sh sweep on the checkpoint at that step and reads back success_rate.
#   - Tracks the best success_rate seen so far and which step achieved it.
#   - If <patience> (default 2) consecutive evals fail to beat the best, kills training
#     and reports the best checkpoint. A hard <max_step> cap (default stage_r_step +
#     200000) also applies regardless, as a safety net.
#   - This is the authoritative (real eval) signal only — it does not also implement
#     the cheaper per-update reward-EMA trip-wire discussed as a possible future
#     refinement; that would catch a collapse faster but is noisier and untested.

set -uo pipefail

EXP_NAME=${1:?Usage: train_stage_r_with_early_stop.sh <exp_name> <stage_e_step> <stage_r_step> [resume_checkpoint] [eval_interval] [patience] [max_step] [extra_hydra_overrides] [num_envs]}
STAGE_E_STEP=${2:?Usage: train_stage_r_with_early_stop.sh <exp_name> <stage_e_step> <stage_r_step> [resume_checkpoint] [eval_interval] [patience] [max_step] [extra_hydra_overrides] [num_envs]}
STAGE_R_STEP=${3:?Usage: train_stage_r_with_early_stop.sh <exp_name> <stage_e_step> <stage_r_step> [resume_checkpoint] [eval_interval] [patience] [max_step] [extra_hydra_overrides] [num_envs]}
RESUME_CKPT=${4:-}
EVAL_INTERVAL=${5:-10000}
PATIENCE=${6:-2}
MAX_STEP=${7:-$((STAGE_R_STEP + 200000))}
EXTRA_OVERRIDES=${8:-}
NUM_ENVS=${9:-16}

REPO=/home/sia-irl/AvatarPoser
EXP_DIR="$REPO/phc/output/HumanoidIm/$EXP_NAME"
LOG="$EXP_DIR/$EXP_NAME.log"
STATE_FILE="$EXP_DIR/.collapse_state"
PATTERN="run_hydra.py learning=distill exp_name=$EXP_NAME"

RESUME_ARG=""
if [ -n "$RESUME_CKPT" ]; then
    if [ ! -f "$RESUME_CKPT" ]; then
        echo "[collapse-detect] ERROR: resume checkpoint not found: $RESUME_CKPT"
        exit 1
    fi
    RESUME_ARG="learning.params.config.student_checkpoint=$RESUME_CKPT"
    echo "[collapse-detect] resuming from $RESUME_CKPT"
else
    echo "[collapse-detect] fresh run from step 0 (Stage 1 -> E -> R all in this launch)"
fi

if [ -n "$RESUME_CKPT" ]; then
    mkdir -p "$EXP_DIR"
else
    rm -rf "$EXP_DIR"
    mkdir -p "$EXP_DIR"
fi
cd "$REPO/phc"
/home/sia-irl/anaconda3/bin/conda run -n rlgpu --no-capture-output bash -c "
unset PYTHONPATH && export WANDB_MODE=offline
cd $REPO/phc && python phc/run_hydra.py \
  learning=distill exp_name=$EXP_NAME env=env_im_getup_mcp robot=smpl_humanoid \
  robot.freeze_hand=True robot.box_body=False env.motion_file=sample_data/phc_train.pkl \
  'env.models=[output/HumanoidIm/phc_kp_pnn_iccv/Humanoid.pth]' \
  env.num_envs=$NUM_ENVS env.obs_v=7 headless=True \
  learning.params.config.avatarposer_checkpoint=../model_zoo/avatarposer.pth \
  learning.params.config.stage_e_step=$STAGE_E_STEP \
  learning.params.config.stage_r_step=$STAGE_R_STEP \
  $RESUME_ARG $EXTRA_OVERRIDES
" > "$LOG.raw" 2>&1 &

TRAIN_WRAPPER_PID=$!
echo "[collapse-detect] training launched (wrapper pid $TRAIN_WRAPPER_PID), num_envs=$NUM_ENVS"
echo "[collapse-detect] stage_e_step=$STAGE_E_STEP stage_r_step=$STAGE_R_STEP"
echo "[collapse-detect] eval every $EVAL_INTERVAL steps after Stage R starts, patience=$PATIENCE, hard cap=$MAX_STEP"

BEST_SUCCESS="-1"
BEST_STEP=0
FAILS=0
NEXT_CHECK=$((STAGE_R_STEP + EVAL_INTERVAL))

if [ -n "$RESUME_CKPT" ] && [ -f "$STATE_FILE" ]; then
    # shellcheck disable=SC1090
    source "$STATE_FILE"
    echo "[collapse-detect] restored bookkeeping from $STATE_FILE: best=$BEST_SUCCESS @ $BEST_STEP, fails=$FAILS, next_check=$NEXT_CHECK"
fi

save_state() {
    printf 'BEST_SUCCESS=%q\nBEST_STEP=%q\nFAILS=%q\nNEXT_CHECK=%q\n' \
        "$BEST_SUCCESS" "$BEST_STEP" "$FAILS" "$NEXT_CHECK" > "$STATE_FILE"
}
save_state

while true; do
    sleep 30

    if ! pgrep -f "$PATTERN" > /dev/null; then
        echo "[collapse-detect] training process is no longer running — stopping watch"
        break
    fi

    if [ "$NEXT_CHECK" -gt "$MAX_STEP" ]; then
        echo "[collapse-detect] hit hard max_step cap ($MAX_STEP), killing training"
        pkill -f "$PATTERN"; sleep 3; pkill -9 -f "$PATTERN" 2>/dev/null
        break
    fi

    CKPT_TAG=$(printf "%07d" "$NEXT_CHECK")
    CKPT="$EXP_DIR/student/student_${CKPT_TAG}.pth"
    if [ -f "$CKPT" ]; then
        echo "[collapse-detect] evaluating step $NEXT_CHECK ..."
        cd "$REPO"
        bash scripts/auto_eval.sh "$EXP_NAME" "$NEXT_CHECK" "$EXTRA_OVERRIDES" > /tmp/collapse_detect_eval_${CKPT_TAG}.log 2>&1

        EVAL_LOG="$EXP_DIR/phc_eval_${CKPT_TAG}.log"
        if [ ! -f "$EVAL_LOG" ]; then
            echo "[collapse-detect] WARNING: eval log not found ($EVAL_LOG), skipping this check"
            NEXT_CHECK=$((NEXT_CHECK + EVAL_INTERVAL))
            save_state
            continue
        fi
        SUCCESS=$(grep "success_rate" "$EVAL_LOG" | head -1 | awk -F: '{print $2}' | tr -d ' ')
        echo "[collapse-detect] step $NEXT_CHECK success_rate=$SUCCESS (best so far: $BEST_SUCCESS @ step $BEST_STEP)"

        IMPROVED=$(python3 -c "print(1 if float('$SUCCESS') > float('$BEST_SUCCESS') else 0)")
        if [ "$IMPROVED" = "1" ]; then
            BEST_SUCCESS="$SUCCESS"
            BEST_STEP="$NEXT_CHECK"
            FAILS=0
        else
            FAILS=$((FAILS + 1))
            echo "[collapse-detect] no improvement over best ($FAILS/$PATIENCE)"
        fi
        save_state

        if [ "$FAILS" -ge "$PATIENCE" ]; then
            echo "[collapse-detect] patience exceeded — stopping training."
            echo "[collapse-detect] BEST CHECKPOINT: student_$(printf '%07d' $BEST_STEP).pth (success_rate=$BEST_SUCCESS)"
            pkill -f "$PATTERN"; sleep 3; pkill -9 -f "$PATTERN" 2>/dev/null
            break
        fi

        NEXT_CHECK=$((NEXT_CHECK + EVAL_INTERVAL))
        save_state
    fi
done

echo "[collapse-detect] DONE. Best checkpoint: student_$(printf '%07d' $BEST_STEP).pth (success_rate=$BEST_SUCCESS)"
