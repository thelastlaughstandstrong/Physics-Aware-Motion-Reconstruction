#!/bin/bash
# run_stage_r_collapse_diagnostic.sh — 3-condition diagnostic for the residual Stage R collapse
# that survives critic warm-up (2026-08-13).
#
#   A  warm-up baseline        : kl_adaptive=True,  shared state_enc   (reproduces quest_stager_warmup)
#   B  fixed LR                : kl_adaptive=False, shared state_enc
#   C  fixed LR + detached critic: kl_adaptive=False, critic reads state_enc().detach()
#
# Everything else is held constant: same seed checkpoint, same warm-up procedure, same reward,
# entropy, kinematic anchor, PPO epochs/minibatches/rollout length, same actor architecture.
#
# Warm-up is PINNED to exactly 47 batches (min=max) rather than plateau-detected. Condition A's
# plateau chose 47; fixing it removes warm-up duration as a confound and guarantees all three
# conditions start PPO at the same step (381664) with the same 4416-step PPO budget, so the
# +480/+960/+2400/+4416 evaluation points line up exactly.
#
# Runs sequentially (one GPU), then sweeps each condition's checkpoints. Disk is at 100% on this
# box, so it aborts rather than filling the filesystem, and the sweep deletes each ~600MB
# position dump as soon as its metrics are read.

set -uo pipefail

REPO=/home/sia-irl/AvatarPoser
SEED="$REPO/phc/output/HumanoidIm/quest_distill_trackerr/student/student_0380160.pth"
WARMUP_BATCHES=47
START=380160
PPO_BUDGET=4416
WARM_END=$((START + WARMUP_BATCHES * 32))       # 381664
CAP=$((WARM_END + PPO_BUDGET))                  # 386080
EVAL_STEPS="$WARM_END $((WARM_END+480)) $((WARM_END+960)) $((WARM_END+2400)) $((WARM_END+PPO_BUDGET))"
LOG="$REPO/phc/output/HumanoidIm/stage_r_diagnostic.log"

say() { echo "[diag] $(date '+%F %T') $*" | tee -a "$LOG"; }

free_gb() { df -BG --output=avail / | tail -1 | tr -dc '0-9'; }

COMMON="learning.params.config.student_checkpoint=output/HumanoidIm/quest_distill_trackerr/student/student_0380160.pth \
learning.params.config.enable_stage_r=True \
learning.params.config.stage_r_critic_warmup=True \
learning.params.config.stage_r_critic_warmup_min_batches=$WARMUP_BATCHES \
learning.params.config.stage_r_critic_warmup_max_batches=$WARMUP_BATCHES \
learning.params.config.stage_r_lr=5e-6 \
learning.params.config.kl_threshold=0.03 \
learning.params.config.stage_r_entropy_coef=0.003 \
learning.params.config.stage_r_action_log_std_init=-3.6 \
learning.params.config.lambda_kinematic_stage_r=0.3 \
learning.params.config.max_distill_step=$CAP"

run_condition() {
    local NAME=$1; shift
    local EXTRA="$*"
    if [ "$(free_gb)" -lt 3 ]; then
        say "ABORT: only $(free_gb)GB free, refusing to start $NAME"; return 1
    fi
    say "=== $NAME === extra: ${EXTRA:-<none>}"
    cd "$REPO/phc"
    conda run -n rlgpu --no-capture-output python phc/run_hydra.py \
        learning=distill_quest exp_name="$NAME" \
        env=env_quest_im_basic robot=quest_humanoid \
        env.motion_file="$REPO/data/quest2_synthetic/train_seg_motion" \
        env.num_envs=128 env.cycle_motion=True env.min_length=5 \
        $COMMON $EXTRA \
        headless=True no_virtual_display=True no_log=True disable_multiprocessing=False \
        test=False im_eval=False \
        > "$REPO/${NAME}_train.log" 2>&1
    say "$NAME training done (warm-up end $WARM_END, cap $CAP)"
    sleep 60   # let Isaac Gym release the GPU before the next sim is created
}

run_condition quest_stager_condA
run_condition quest_stager_condB learning.params.config.kl_adaptive=False
run_condition quest_stager_condC learning.params.config.kl_adaptive=False \
                                  learning.params.config.stage_r_detach_critic_input=True

for C in quest_stager_condA quest_stager_condB quest_stager_condC; do
    say "sweeping $C at steps: $EVAL_STEPS"
    bash "$REPO/scripts/eval_stage_r_sweep.sh" "$C" $EVAL_STEPS > "$REPO/${C}_sweep.log" 2>&1
    say "$C sweep done"
    cat "$REPO/${C}_sweep.log" | tee -a "$LOG"
done

say "DIAGNOSTIC COMPLETE"
