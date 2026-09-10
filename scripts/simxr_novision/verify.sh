#!/bin/bash
# Check a completed eval log against the reference numbers, without rerunning anything.
set -uo pipefail
LOG=${1:-/home/sia-irl/SimXR/output/HumanoidIm/simxr_quest_novision_lrdecay/eval_peak_1500.log}
REF_SUCC="0.7173507463"; REF_MPJPE="112.527"
echo "checking: $LOG"
S=$(grep -m1 "Success Rate" "$LOG" 2>/dev/null | grep -oE "[0-9.]+")
M=$(grep -m1 "^All:" "$LOG" 2>/dev/null | grep -oE "mpjpe_g: [0-9.]+" | grep -oE "[0-9.]+")
printf "  success  : %-14s (reference %s)  %s\n" "${S:-MISSING}" "$REF_SUCC" \
  "$([ "${S:-}" = "$REF_SUCC" ] && echo MATCH || echo DIFFERS)"
printf "  mpjpe_g  : %-14s (reference %s)  %s\n" "${M:-MISSING}" "$REF_MPJPE" \
  "$([ "${M:-}" = "$REF_MPJPE" ] && echo MATCH || echo DIFFERS)"
