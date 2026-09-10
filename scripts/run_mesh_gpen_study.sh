#!/bin/bash
# Mesh-GPen study: Step 1/2 diagnostics (GT retargeting floor on the known clips), then the
# Step 3 four-way comparison. Results cached per (clip, stream) so reruns are free.
set -u
REPO=/home/sia-irl/AvatarPoser
PY=/home/sia-irl/anaconda3/envs/avatarposer/bin/python
export LD_LIBRARY_PATH=/home/sia-irl/anaconda3/envs/avatarposer/lib
TR=$REPO/output/HumanoidIm/quest_distill_trackerr/phc_positions_for_eval.pkl
NV=$REPO/output/HumanoidIm/phc_prim_quest/phc_positions_for_eval.pkl
SX=/home/sia-irl/SimXR/output/HumanoidIm/simxr_quest_novision_lrdecay/phc_positions_for_eval.pkl

echo "=== STEP 1+2: diagnostic clips, GT retargeting floor ==="
$PY $REPO/scripts/mesh_gpen_quest.py \
  --clips SUBJECT070.sports.pkl_0_417 SUBJECT070.gestures.pkl_0_423 SUBJECT059.martialarts.pkl_0_444 \
  --streams gt=gt --out $REPO/scripts/mesh_gpen_diagnostics.json

echo "=== STEP 3: four-way on the approved clips ==="
$PY $REPO/scripts/mesh_gpen_quest.py \
  --clips SUBJECT072.gestures.pkl_427_853 SUBJECT086.walking.pkl_0_412 SUBJECT075.ROM.pkl_3820_4243 \
  --streams gt=gt "simxr=$SX" "finetuned_backbone=$TR" "naive=$NV" \
  --out $REPO/scripts/mesh_gpen_fourway.json
echo "[mesh-gpen] complete"
