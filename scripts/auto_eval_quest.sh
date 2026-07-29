#!/bin/bash
# auto_eval_quest.sh — evaluates a single quest_distill checkpoint on the full test_seg_motion
# set (physics rollout via run_hydra.py, then evaluate_phc_metrics_quest.py), mirroring
# auto_eval.sh's contract for the SMPL/Stage-R pipeline. Used both by
# train_distill_quest_with_periodic_eval.sh (live, during training) and
# quest_checkpoint_scan.sh (retroactive, on an already-finished run).
#
# Usage:
#   bash scripts/auto_eval_quest.sh <exp_name> <target_step> [extra_hydra_overrides] [num_envs]
#
# quest_distill checkpoints save every ~480 steps (not necessarily aligned to an arbitrary
# requested step), so this picks the checkpoint CLOSEST to <target_step> rather than requiring
# an exact match.
#
# num_envs (default 128): lower this when running concurrently with training on a GPU that's
# tight on memory -- it only trades eval parallelism/speed for a smaller footprint (still
# processes all 1072 test sequences, just in more/smaller batches), it does not change the
# resulting metrics. Added after a real OOM (2026-07-27): training's memory footprint grows
# over a long run, and a full-size concurrent eval can stop fitting alongside it.

set -uo pipefail

EXP_NAME=${1:?Usage: auto_eval_quest.sh <exp_name> <target_step> [extra_hydra_overrides] [num_envs]}
TARGET_STEP=${2:?Usage: auto_eval_quest.sh <exp_name> <target_step> [extra_hydra_overrides] [num_envs]}
EXTRA_OVERRIDES=${3:-}
NUM_ENVS=${4:-128}

REPO=/home/sia-irl/AvatarPoser
EXP_DIR=$REPO/phc/output/HumanoidIm/$EXP_NAME
CKPT_DIR=$EXP_DIR/student
TEST_MOTION=$REPO/data/quest2_synthetic/test_seg_motion

find_checkpoint() {
    local best_file="" best_diff="" f step diff
    for f in "$CKPT_DIR"/student_*.pth; do
        [ -e "$f" ] || continue
        step=$(basename "$f" | sed -E 's/student_0*([0-9]+)\.pth/\1/')
        diff=$(( step - TARGET_STEP ))
        if [ "$diff" -lt 0 ]; then diff=$(( -diff )); fi
        if [ -z "$best_diff" ] || [ "$diff" -lt "$best_diff" ]; then
            best_diff=$diff
            best_file=$f
        fi
    done
    echo "$best_file"
}

CKPT=$(find_checkpoint)
if [ -z "$CKPT" ]; then
    echo "[auto_eval_quest] ERROR: no checkpoints found in $CKPT_DIR"
    exit 1
fi
CKPT_STEP=$(basename "$CKPT" | sed -E 's/student_0*([0-9]+)\.pth/\1/')
echo "[auto_eval_quest] Evaluating $CKPT (step $CKPT_STEP, requested $TARGET_STEP)"

POS_DUMP="$EXP_DIR/phc_positions_for_eval.pkl"
# Remove any stale dump from a previous (possibly crashed) run first -- otherwise a run that
# OOMs/crashes before writing a fresh dump silently falls through to the "file exists" check
# below and re-scores the OLD dump under THIS run's step label, producing a mislabeled, wrong
# result instead of a loud failure. Bit us once already (2026-07-27, step 20160 silently
# re-reported step 10080's numbers this way).
rm -f "$POS_DUMP"

cd "$REPO/phc"
conda run -n rlgpu --no-capture-output python phc/run_hydra.py \
  learning=distill_quest exp_name=$EXP_NAME \
  env=env_quest_im_basic robot=quest_humanoid \
  env.motion_file=$TEST_MOTION \
  env.num_envs=$NUM_ENVS env.cycle_motion=False env.min_length=5 \
  learning.params.config.student_checkpoint=output/HumanoidIm/$EXP_NAME/student/$(basename "$CKPT") \
  test=True im_eval=True headless=True disable_multiprocessing=False \
  $EXTRA_OVERRIDES
if [ ! -f "$POS_DUMP" ]; then
    echo "[auto_eval_quest] ERROR: physics eval did not produce $POS_DUMP"
    exit 1
fi

STEP_TAG=$(printf "%07d" "$CKPT_STEP")
TAGGED_POS="$EXP_DIR/phc_positions_for_eval_${STEP_TAG}.pkl"
cp "$POS_DUMP" "$TAGGED_POS"

METRICS_LOG="$EXP_DIR/quest_eval_${STEP_TAG}.log"
conda run -n avatarposer --no-capture-output python "$REPO/scripts/evaluate_phc_metrics_quest.py" \
  --pos_dump "$TAGGED_POS" --out "$EXP_DIR/quest_eval_${STEP_TAG}.json" \
  | tee "$METRICS_LOG"

echo "[auto_eval_quest] METRICS_LOG=$METRICS_LOG"
echo "[auto_eval_quest] CKPT_STEP=$CKPT_STEP"
