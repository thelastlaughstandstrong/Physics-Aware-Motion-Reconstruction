#!/bin/bash
# train_distill_quest_with_auto_recovery.sh — launches Phase C's quest_humanoid distillation
# training (distill_quest algo, DistillAgentQuest) with automatic crash recovery.
#
# Exists because a real ~15-hour training run (quest_distill_option2_scratch, 2026-07-17) OOM'd
# from a CUDA memory leak (gpu_alloc grew steadily from ~0.4GB to ~8.9GB over 302k steps).
# Isolated reproduction tests ruled out the FK/body-model module and the batched
# training-loop logic itself (both came back flat/clean on synthetic data, no simulator) --
# so the leak lives inside Isaac Gym's own environment-stepping internals, the same class of
# leak this project hit once before and only partially mitigated via torch.no_grad() around
# env_step(). Rather than chase an Isaac-Gym-internal leak further, this treats an eventual
# crash as expected and auto-resumes from the latest checkpoint, mirroring the strategy
# scripts/train_with_auto_recovery.sh already uses for the separate SMPL/Stage-R pipeline.
#
# Usage:
#   bash scripts/train_distill_quest_with_auto_recovery.sh <exp_name> <max_distill_step> \
#        [num_envs] [max_crash_retries] [gpu_free_mib_threshold] [extra_hydra_overrides]
#
# extra_hydra_overrides (new, quoted as one argument) is appended verbatim to the hydra
# command line -- e.g. to test the loss-reweighting experiment (action_loss dominant,
# matching SimXR's actual recipe -- see distill_agent_quest.py's lambda_kinematic):
#   "learning.params.config.lambda_kinematic=0.0 learning.params.config.lambda_action=1.0"
#
# Resume behavior is keyed off whether a checkpoint already exists in
# output/HumanoidIm/<exp_name>/student/ at the start of EACH attempt (not "attempt number == 1")
# -- so this is safe to (re)launch at any time: a genuinely new exp_name starts fresh (no
# student_checkpoint, matching the deliberate "from scratch, no checkpoint-chaining across
# algorithm changes" decision), while an exp_name with existing checkpoints (either because
# this wrapper crashed and was relaunched, or because training had already progressed before
# this wrapper was first used) resumes from the latest one automatically.
#
# Example (continuing the already-crashed quest_distill_option2_scratch run):
#   bash scripts/train_distill_quest_with_auto_recovery.sh quest_distill_option2_scratch 500000

set -uo pipefail

EXP_NAME=${1:?Usage: train_distill_quest_with_auto_recovery.sh <exp_name> <max_distill_step> [num_envs] [max_crash_retries] [gpu_free_mib_threshold]}
MAX_DISTILL_STEP=${2:?Usage: train_distill_quest_with_auto_recovery.sh <exp_name> <max_distill_step> [num_envs] [max_crash_retries] [gpu_free_mib_threshold]}
NUM_ENVS=${3:-128}
MAX_RETRIES=${4:-10}
GPU_FREE_MIB_THRESHOLD=${5:-10000}
EXTRA_OVERRIDES=${6:-}

REPO=/home/sia-irl/AvatarPoser
EXP_DIR="$REPO/phc/output/HumanoidIm/$EXP_NAME"
RECOVERY_DIR="$REPO/phc/output/HumanoidIm/.recovery_logs/$EXP_NAME"
RECOVERY_LOG="$RECOVERY_DIR/auto_recovery.log"
MOTION_FILE="$REPO/data/quest2_synthetic/train_seg_motion"

mkdir -p "$EXP_DIR" "$RECOVERY_DIR"
echo "[auto-recovery] starting supervised run for $EXP_NAME (max $MAX_RETRIES crash retries, cap=$MAX_DISTILL_STEP steps, num_envs=$NUM_ENVS)" | tee -a "$RECOVERY_LOG"

latest_checkpoint() {
    # student_%07d.pth is zero-padded, so plain lexicographic sort orders it numerically.
    ls -1 "$EXP_DIR/student/"student_*.pth 2>/dev/null | sort | tail -1
}

wait_for_gpu() {
    local free_mib
    while true; do
        free_mib=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | head -1)
        if [ -n "$free_mib" ] && [ "$free_mib" -ge "$GPU_FREE_MIB_THRESHOLD" ]; then
            echo "[auto-recovery] GPU has ${free_mib}MiB free (>= ${GPU_FREE_MIB_THRESHOLD}MiB threshold) — proceeding" | tee -a "$RECOVERY_LOG"
            return
        fi
        echo "[auto-recovery] waiting for GPU: ${free_mib:-unknown}MiB free, need ${GPU_FREE_MIB_THRESHOLD}MiB — rechecking in 60s" | tee -a "$RECOVERY_LOG"
        sleep 60
    done
}

ATTEMPT=0

while true; do
    ATTEMPT=$((ATTEMPT + 1))
    ATTEMPT_LOG="$RECOVERY_DIR/attempt_${ATTEMPT}.log"

    RESUME_CKPT=$(latest_checkpoint)
    RESUME_ARG=""
    if [ -z "$RESUME_CKPT" ]; then
        echo "[auto-recovery] attempt $ATTEMPT: no existing checkpoint found -- fresh start (no student_checkpoint)" | tee -a "$RECOVERY_LOG"
    else
        echo "[auto-recovery] attempt $ATTEMPT: resuming from $RESUME_CKPT" | tee -a "$RECOVERY_LOG"
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
        > "$ATTEMPT_LOG" 2>&1

    if grep -q "hit max_distill_step cap" "$ATTEMPT_LOG"; then
        echo "[auto-recovery] run completed normally (attempt $ATTEMPT)" | tee -a "$RECOVERY_LOG"
        tail -3 "$ATTEMPT_LOG" | tee -a "$RECOVERY_LOG"
        exit 0
    fi

    echo "[auto-recovery] attempt $ATTEMPT ended without a deliberate stop — treating as a crash" | tee -a "$RECOVERY_LOG"
    tail -15 "$ATTEMPT_LOG" | tee -a "$RECOVERY_LOG"

    if [ "$ATTEMPT" -gt "$MAX_RETRIES" ]; then
        echo "[auto-recovery] exceeded max_crash_retries ($MAX_RETRIES) — giving up. Last checkpoint: $(latest_checkpoint)" | tee -a "$RECOVERY_LOG"
        exit 1
    fi

    echo "[auto-recovery] pausing 30s before retry $((ATTEMPT + 1))..." | tee -a "$RECOVERY_LOG"
    sleep 30
done
