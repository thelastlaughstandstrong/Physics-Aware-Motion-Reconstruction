#!/bin/bash
# Characterization study: Step 5 Quest->SMPL retargeting for the 8 NEW clips.
# standing/walking already exist from Steps 5-6E and are reused unchanged.
# Same settings as the validated Step 5 run: native Z-up, neutral SMPL, fixed scale, w_root=0.25.
set -u
REPO=/home/sia-irl/AvatarPoser
OUT=$REPO/data/quest_smpl_retarget/charstudy
D=$REPO/data/quest2_synthetic/test_seg_motion

CLIPS="
gestures:SUBJECT070.gestures.pkl_0_423.pkl
whiteboard:SUBJECT069.whiteboard.pkl_0_424.pkl
rom:SUBJECT070.ROM.pkl_0_435.pkl
stretching:SUBJECT061.stretching.pkl_0_438.pkl
jogging:SUBJECT070.jogging.pkl_0_435.pkl
sports:SUBJECT070.sports.pkl_0_417.pkl
dancing:SUBJECT073.dancing.pkl_0_428.pkl
martialarts:SUBJECT059.martialarts.pkl_0_444.pkl
"
for entry in $CLIPS; do
  tag=${entry%%:*}; f=${entry##*:}
  if [ -f "$OUT/$tag.json" ]; then echo "[skip] $tag"; continue; fi
  echo "=== $tag ($f) ==="
  /home/sia-irl/anaconda3/envs/avatarposer/bin/python $REPO/scripts/retarget_quest_to_smpl_sequence.py \
      --motion_file "data/quest2_synthetic/test_seg_motion/$f" \
      --out-prefix "$OUT/$tag" 2>&1 | tail -2
done
echo "[charstudy] retargeting complete"
