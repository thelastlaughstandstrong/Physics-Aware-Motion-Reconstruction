#!/bin/bash
# train_with_auto_recovery.sh — wraps train_stage_r_with_early_stop.sh with crash
# detection and automatic resume, so a run can be left unattended across crashes
# (e.g. GPU contention from an unrelated process, like the Gaussian-splatting job
# that once crashed a concurrent eval this session) instead of needing a human to
# notice a dead process and manually relaunch from the last checkpoint.
#
# Distinguishes a real crash from an intentional stop: train_stage_r_with_early_stop.sh
# always ends by either (a) printing "patience exceeded" or "hit hard max_step cap"
# (a deliberate decision — the run is genuinely done, don't restart), or (b) just
# disappearing without either message (the training process died unexpectedly —
# a crash). Only case (b) triggers a retry.
#
# Usage:
#   bash scripts/train_with_auto_recovery.sh <exp_name> <stage_e_step> <stage_r_step> \
#        [eval_interval] [patience] [max_step] [extra_hydra_overrides] [num_envs] \
#        [max_crash_retries] [gpu_free_mib_threshold]
#
# Waits for at least gpu_free_mib_threshold MiB free on the GPU (default 10000) before
# every launch attempt -- including the very first one -- polling every 60s. Handles
# an unrelated process hogging the GPU (observed this session: another user's job
# using ~16GB for 20+ minutes) without needing a human to notice and relaunch by hand.
#
# Example:
#   bash scripts/train_with_auto_recovery.sh ap_phc_distill_v7_scaled 100000 250000 \
#        10000 2 "" "" 100 5
#
# Always starts fresh (no manual resume_checkpoint argument — that's what crash
# recovery is for). To resume a specific existing checkpoint deliberately, use
# train_stage_r_with_early_stop.sh directly instead.

set -uo pipefail

EXP_NAME=${1:?Usage: train_with_auto_recovery.sh <exp_name> <stage_e_step> <stage_r_step> [eval_interval] [patience] [max_step] [extra_hydra_overrides] [num_envs] [max_crash_retries]}
STAGE_E_STEP=${2:?Usage: train_with_auto_recovery.sh <exp_name> <stage_e_step> <stage_r_step> [eval_interval] [patience] [max_step] [extra_hydra_overrides] [num_envs] [max_crash_retries]}
STAGE_R_STEP=${3:?Usage: train_with_auto_recovery.sh <exp_name> <stage_e_step> <stage_r_step> [eval_interval] [patience] [max_step] [extra_hydra_overrides] [num_envs] [max_crash_retries]}
EVAL_INTERVAL=${4:-10000}
PATIENCE=${5:-2}
MAX_STEP=${6:-$((STAGE_R_STEP + 200000))}
EXTRA_OVERRIDES=${7:-}
NUM_ENVS=${8:-16}
MAX_RETRIES=${9:-5}
GPU_FREE_MIB_THRESHOLD=${10:-10000}

REPO=/home/sia-irl/AvatarPoser
EXP_DIR="$REPO/phc/output/HumanoidIm/$EXP_NAME"
# Deliberately OUTSIDE EXP_DIR: a fresh (non-resume) attempt has
# train_stage_r_with_early_stop.sh do `rm -rf "$EXP_DIR"` internally, which would
# unlink these log files out from under our own open write handle if they lived
# inside it -- the path would vanish (though writes would silently keep landing in
# the orphaned inode), and our own grep/tail checks against that path would then
# fail with "No such file or directory", misreading a real completion as a crash.
RECOVERY_DIR="$REPO/phc/output/HumanoidIm/.recovery_logs/$EXP_NAME"
RECOVERY_LOG="$RECOVERY_DIR/auto_recovery.log"

mkdir -p "$EXP_DIR" "$RECOVERY_DIR"
echo "[auto-recovery] starting supervised run for $EXP_NAME (max $MAX_RETRIES crash retries)" | tee -a "$RECOVERY_LOG"

latest_checkpoint() {
    # Checkpoint filenames are zero-padded to a fixed width (student_%07d.pth), so a
    # plain lexicographic sort already orders them numerically -- avoids splitting on
    # "_" (which breaks since exp names themselves contain underscores).
    ls -1 "$EXP_DIR/student/"student_*.pth 2>/dev/null | sort | tail -1
}

wait_for_gpu() {
    # Waits until at least GPU_FREE_MIB_THRESHOLD MiB is free -- not just "no error",
    # actually checks nvidia-smi so an unrelated user's job (observed this session:
    # another user's Gaussian-splatting training using ~16GB for 20+ minutes) doesn't
    # cause a launch straight into a CUDA OOM.
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
RESUME_CKPT=""

while true; do
    ATTEMPT=$((ATTEMPT + 1))
    ATTEMPT_LOG="$RECOVERY_DIR/attempt_${ATTEMPT}.log"

    if [ "$ATTEMPT" -gt 1 ]; then
        RESUME_CKPT=$(latest_checkpoint)
        if [ -z "$RESUME_CKPT" ]; then
            # Crashed before the first checkpoint saved (e.g. transient GPU contention
            # from an unrelated process during startup) -- nothing to resume from, but
            # that doesn't mean the underlying cause is permanent, so retry fresh
            # instead of giving up outright.
            echo "[auto-recovery] attempt $ATTEMPT/$((MAX_RETRIES + 1)): crashed before any checkpoint saved, retrying fresh start" | tee -a "$RECOVERY_LOG"
        else
            echo "[auto-recovery] attempt $ATTEMPT/$((MAX_RETRIES + 1)): resuming from $RESUME_CKPT" | tee -a "$RECOVERY_LOG"
        fi
    else
        echo "[auto-recovery] attempt $ATTEMPT: fresh start" | tee -a "$RECOVERY_LOG"
    fi

    wait_for_gpu

    bash "$REPO/scripts/train_stage_r_with_early_stop.sh" \
        "$EXP_NAME" "$STAGE_E_STEP" "$STAGE_R_STEP" "$RESUME_CKPT" \
        "$EVAL_INTERVAL" "$PATIENCE" "$MAX_STEP" "$EXTRA_OVERRIDES" "$NUM_ENVS" \
        > "$ATTEMPT_LOG" 2>&1

    if grep -qE "patience exceeded|hit hard max_step cap" "$ATTEMPT_LOG"; then
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

    # Short pause before retrying -- the real wait for GPU contention specifically
    # happens via wait_for_gpu() at the top of the next iteration; this is just a
    # brief buffer so a non-GPU crash (e.g. an actual bug) doesn't tight-loop.
    echo "[auto-recovery] pausing 30s before retry $((ATTEMPT + 1))..." | tee -a "$RECOVERY_LOG"
    sleep 30
done
