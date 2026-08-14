#!/bin/bash
# eval_stage_r_sweep.sh — evaluate a list of Stage R checkpoints and print one metrics row each.
#
# Disk-safe by construction: each auto_eval_quest.sh run writes a ~600MB
# phc_positions_for_eval_<step>.pkl, and the box is currently at 100% (see the dataset note in
# results_comparison.md). This deletes each dump as soon as its metrics JSON has been read, so
# peak extra usage is one dump rather than one-per-checkpoint. The metrics JSON/log are tiny and
# are kept.
#
# Usage: bash scripts/eval_stage_r_sweep.sh <exp_name> <step> [<step> ...]

set -uo pipefail
EXP=${1:?Usage: eval_stage_r_sweep.sh <exp_name> <step> [<step> ...]}
shift

REPO=/home/sia-irl/AvatarPoser
DIR="$REPO/phc/output/HumanoidIm/$EXP"
OVR="learning.params.config.enable_stage_r=True \
learning.params.config.stage_r_lr=5e-6 \
learning.params.config.kl_threshold=0.03 \
learning.params.config.stage_r_entropy_coef=0.003 \
learning.params.config.stage_r_action_log_std_init=-3.6 \
learning.params.config.lambda_kinematic_stage_r=0.3"

printf "%-10s %-9s %9s %9s %9s %9s %9s %9s\n" \
       step ckpt success mpjpe_mm mpjre_deg mpjve_cms jitter footskate

for S in "$@"; do
    OUT="$DIR/sweep_eval_${S}.log"
    bash "$REPO/scripts/auto_eval_quest.sh" "$EXP" "$S" "$OVR" > "$OUT" 2>&1
    ACTUAL=$(grep -a "^\[auto_eval_quest\] CKPT_STEP=" "$OUT" | tail -1 | cut -d= -f2 | tr -d ' ')
    MJ=$(grep -a "^\[auto_eval_quest\] METRICS_LOG=" "$OUT" | tail -1 | cut -d= -f2-)
    JSON="${MJ%.log}.json"
    if [ -f "$JSON" ]; then
        python3 - "$JSON" "$S" "$ACTUAL" <<'PY'
import json, sys
d = json.load(open(sys.argv[1])); m = d.get("metrics", d)
g = lambda k: m.get(k, float("nan"))
print("%-10s %-9s %9.4f %9.2f %9.2f %9.2f %9.1f %9.2f" % (
    sys.argv[2], sys.argv[3], g("success_rate_phc_survival"), g("eg_mpjpe_mm"),
    g("rot_error_deg"), g("vel_error_cm_s"), g("jitter_m_s3"), g("foot_skate_cm_s")))
PY
    else
        printf "%-10s %-9s %s\n" "$S" "${ACTUAL:-?}" "EVAL FAILED (see $OUT)"
    fi
    # Free the dump before the next eval -- regenerable from the checkpoint, which is kept.
    rm -f "$DIR/phc_positions_for_eval_"*.pkl "$DIR/phc_positions_for_eval.pkl"
done
