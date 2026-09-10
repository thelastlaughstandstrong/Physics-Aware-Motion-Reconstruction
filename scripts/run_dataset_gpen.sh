#!/bin/bash
# Full-dataset mesh-GPen: all six stream-stages over the 1072-clip Quest2 test set.
# One chunk per stream (batched over ALL clips) since the IK is kernel-launch-bound.
set -u
REPO=/home/sia-irl/AvatarPoser
PY=/home/sia-irl/anaconda3/envs/avatarposer/bin/python
export LD_LIBRARY_PATH=/home/sia-irl/anaconda3/envs/avatarposer/lib
for s in gt phase_a fb_pre finetuned_backbone naive simxr; do
  if [ -f "$REPO/data/quest_smpl_retarget/dataset_gpen/$s.json" ]; then
    n=$($PY -c "import json;print(len(json.load(open('$REPO/data/quest_smpl_retarget/dataset_gpen/$s.json'))))" 2>/dev/null || echo 0)
    [ "$n" -ge 1000 ] && { echo "=== $s already complete ($n clips), skipping ==="; continue; }
  fi
  echo "=== STREAM $s ==="
  $PY $REPO/scripts/mesh_gpen_dataset.py --stream $s --chunk 1072 2>&1 \
    | grep -E "^stream=|loaded |frame (1|101|201|301|401)/|^\[|Error|Traceback"
done
echo "[dataset-gpen] ALL STREAMS COMPLETE"
