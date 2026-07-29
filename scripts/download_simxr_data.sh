#!/bin/bash
# download_simxr_data.sh — downloads SimXR's Quest 2 synthetic train/test
# data from Google Drive into data/simxr/train and data/simxr/test.
#
# Uses gdown since these are Drive *folders*, not single files -- gdown
# --folder recursively mirrors a shared folder's contents without needing
# OAuth setup, as long as the folder is shared as "anyone with the link".
#
# Google Drive enforces a per-file anonymous-download quota -- a popular,
# widely-shared dataset like this one can have individual files temporarily
# blocked ("too many people have requested this today") independent of our
# own request pattern, and sustained automated retries can also trip a
# broader anonymous-access block covering the whole folder. gdown's
# --folder mode resolves each file's download link *before* checking
# whether it's already downloaded, and aborts the entire folder on the
# first file that hits this block -- so one persistently-blocked file (even
# one already fully downloaded) can stall every other file behind it
# indefinitely.
#
# To avoid that, this script gets the file listing cheaply via
# `--folder --json` (only needs the folder's HTML page, not each file's
# confirm-token resolution, so it's immune to the per-file quota), skips
# anything already present in the destination directory, and downloads the
# rest one file at a time -- so a single blocked file just gets retried on
# its own in the next pass instead of blocking every file after it.
#
# gdown's folder scraper only parses whatever Drive embeds in the initial
# HTML page load, which Drive caps at 50 items -- so gdown<6.0.0 silently
# lists only the first 50 files per folder no matter how many are actually
# shared. Fixed in gdown 6.0.0, which requires Python >=3.10 (the
# avatarposer conda env is 3.8), so this uses a separate `gdown_dl` conda
# env (python 3.12) created just to run modern gdown.
#
# Usage:
#   bash scripts/download_simxr_data.sh

set -uo pipefail

TRAIN_URL="https://drive.google.com/drive/folders/1jMld_d6JmyNkq0w1mBWH5nSts9tsO47b"
TEST_URL="https://drive.google.com/drive/folders/1RACtEleG5saxvjyt1KRe-p0jA4WmAGMS"

REPO=/home/sia-irl/AvatarPoser
OUT_DIR="$REPO/data/simxr"
GDOWN=/home/sia-irl/anaconda3/envs/gdown_dl/bin/gdown
PYTHON=/home/sia-irl/anaconda3/envs/gdown_dl/bin/python
RETRY_WAIT_SECS=${RETRY_WAIT_SECS:-300}
MAX_PASSES=${MAX_PASSES:-200}

mkdir -p "$OUT_DIR/train" "$OUT_DIR/test"

download_folder() {
    local url=$1 dest=$2 label=$3
    local listing="$dest/.file_listing.json"

    echo "[download] $label: fetching file listing"
    if ! "$GDOWN" --folder "$url" --json > "$listing" 2>"$dest/.listing_stderr.log"; then
        echo "[download] $label: failed to fetch file listing, see $dest/.listing_stderr.log"
        return 1
    fi

    local pass=1
    while [ "$pass" -le "$MAX_PASSES" ]; do
        local missing
        missing=$("$PYTHON" - "$listing" "$dest" <<'PYEOF'
import json, os, sys
listing_path, dest = sys.argv[1], sys.argv[2]
with open(listing_path) as f:
    entries = json.load(f)
missing = [e for e in entries if not os.path.isfile(os.path.join(dest, e["path"]))]
for e in missing:
    print(e["url"], e["path"], sep="\t")
PYEOF
        )

        if [ -z "$missing" ]; then
            echo "[download] $label: all files present"
            return 0
        fi

        local total_missing
        total_missing=$(echo "$missing" | wc -l)
        echo "[download] $label pass $pass/$MAX_PASSES: $total_missing file(s) still missing"

        local failed=0
        while IFS=$'\t' read -r file_url file_path; do
            [ -z "$file_url" ] && continue
            local out="$dest/$file_path"
            mkdir -p "$(dirname "$out")"
            if "$GDOWN" "$file_url" -O "$out" --continue -q; then
                echo "[download] $label: got $file_path"
            else
                echo "[download] $label: failed on $file_path (likely per-file/anonymous-access quota)"
                failed=$((failed + 1))
            fi
        done <<< "$missing"

        if [ "$failed" -eq 0 ]; then
            echo "[download] $label: all missing files downloaded"
            return 0
        fi

        echo "[download] $label pass $pass: $failed file(s) still failing -- waiting ${RETRY_WAIT_SECS}s before retry"
        sleep "$RETRY_WAIT_SECS"
        pass=$((pass + 1))
    done

    echo "[download] $label: gave up after $MAX_PASSES passes"
    return 1
}

download_folder "$TRAIN_URL" "$OUT_DIR/train" "train"
download_folder "$TEST_URL" "$OUT_DIR/test" "test"

echo "[download] done (or gave up) -- current counts:"
echo "  train: $(ls "$OUT_DIR/train"/*.pkl 2>/dev/null | wc -l) files"
echo "  test:  $(ls "$OUT_DIR/test"/*.pkl 2>/dev/null | wc -l) files"
