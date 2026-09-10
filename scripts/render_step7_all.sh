#!/bin/bash
# Step 7: render all three comparison clips, then encode PNG sequences to 30 fps H.264.
set -u
REPO=/home/sia-irl/AvatarPoser
EXP=$REPO/data/quest_smpl_retarget/blender_export
OUT=$REPO/data/quest_smpl_retarget/renders
mkdir -p "$OUT"
for tag in sports gestures martialarts; do
  echo "=== rendering $tag ==="
  blender --background --python $REPO/scripts/blender_render_comparison.py -- \
      --npz $EXP/mesh_${tag}.npz --quest $EXP/quest_gt_${tag}.npz \
      --tag $tag --outdir "$OUT" --stills 3 --samples 16 2>&1 \
    | grep -E "^\[render\]"
  /home/sia-irl/anaconda3/envs/avatarposer/bin/ffmpeg -y -loglevel error -framerate 30 -i "$OUT/_${tag}_frames/f_%04d.png" \
      -c:v libx264 -pix_fmt yuv420p -crf 18 "$OUT/${tag}_comparison.mp4" \
    && echo "[encode] $OUT/${tag}_comparison.mp4"
done
echo "[step7] all renders complete"
