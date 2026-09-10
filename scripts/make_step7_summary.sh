#!/bin/bash
# Combine the three comparison renders into one summary video, each preceded by a title card
# stating that clip's measured trade-off. All three renders are 1920x1080 @ 30 fps.
set -u
FF=/home/sia-irl/anaconda3/envs/avatarposer/bin/ffmpeg
FONT=/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf
OUT=/home/sia-irl/AvatarPoser/data/quest_smpl_retarget/renders
T=$OUT/_titles; mkdir -p "$T"

mk () {  # $1 name  $2 headline  $3 detail
  "$FF" -y -loglevel error -f lavfi -i "color=c=0x0d1117:s=1920x1080:d=2.5:r=30" \
    -vf "drawtext=fontfile=$FONT:text='$2':fontcolor=white:fontsize=72:x=(w-tw)/2:y=(h/2)-90,drawtext=fontfile=$FONT:text='$3':fontcolor=0x9aa4b2:fontsize=40:x=(w-tw)/2:y=(h/2)+40" \
    -c:v libx264 -pix_fmt yuv420p -crf 18 "$T/t_$1.mp4"
}
mk sports      "SPORTS - physics beneficial" "penetration 24.5 to 0 mm | skate 27.9 to 1.0 cm/s | MPJPE +42.3 mm"
mk gestures    "GESTURES - physics harmful"  "little to fix pre-physics | MPJPE +19.2 mm | jitter 2.6x"
mk martialarts "MARTIAL ARTS - mixed"        "penetration 29.9 to 0 mm | but MPJPE +48.2 mm"

: > "$T/list.txt"
for tag in sports gestures martialarts; do
  echo "file '$T/t_${tag}.mp4'"          >> "$T/list.txt"
  echo "file '$OUT/${tag}_comparison.mp4'" >> "$T/list.txt"
done
"$FF" -y -loglevel error -f concat -safe 0 -i "$T/list.txt" \
  -c:v libx264 -pix_fmt yuv420p -crf 18 -r 30 "$OUT/summary_all_three.mp4" \
  && echo "[summary] wrote $OUT/summary_all_three.mp4"
