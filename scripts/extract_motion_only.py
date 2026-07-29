"""Rebuilds data/quest2_synthetic/{train,test}_seg_motion in place.

The existing *_seg_motion files already drop the two heaviest vision arrays
(segmentation_mono, heatmaps) but still carry j2ds (2D pixel-space joint
projections -- also camera-derived) and, more importantly, pickle every
per-segment tensor as a view into the *original* full-sequence storage
instead of a standalone copy. That means a 424-frame segment file embeds
the entire multi-thousand-frame original array on disk. The `scale` field
additionally was never sliced to the segment window at all.

This script re-derives each segment from the corresponding *_seg file
(which already has correctly-windowed T-indexed fields, since only `scale`
was left unsliced), drops all vision fields, slices `scale` using the
frame range encoded in the filename (`<orig_stem>_<start>_<end>.pkl`), and
clones every tensor/array so the saved file only contains its own segment's
bytes.
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import joblib
import numpy as np
import torch

VISION_KEYS = {"segmentation_mono", "heatmaps", "j2ds"}
SEG_NAME_RE = re.compile(r"^(?P<stem>.+\.pkl)_(?P<start>\d+)_(?P<end>\d+)\.pkl$")


def detach(value):
    if isinstance(value, torch.Tensor):
        return value.clone().contiguous()
    if isinstance(value, np.ndarray):
        return np.array(value, copy=True)
    return value


def process_one(seg_path: Path, out_path: Path) -> tuple[int, int]:
    m = SEG_NAME_RE.match(seg_path.name)
    if not m:
        raise ValueError(f"unrecognized segment filename: {seg_path.name}")
    start, end = int(m.group("start")), int(m.group("end"))

    before_size = seg_path.stat().st_size
    data = joblib.load(seg_path)
    (key, entry), = data.items()

    new_entry = {k: detach(v) for k, v in entry.items() if k not in VISION_KEYS}
    new_entry["scale"] = detach(entry["scale"][start : end + 1])

    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")
    joblib.dump({key: new_entry}, tmp_path)
    tmp_path.replace(out_path)
    after_size = out_path.stat().st_size
    return before_size, after_size


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/home/sia-irl/AvatarPoser/data/quest2_synthetic")
    ap.add_argument("--splits", nargs="+", default=["train", "test"])
    ap.add_argument("--limit", type=int, default=None, help="process only first N files per split (dry run)")
    args = ap.parse_args()

    root = Path(args.root)
    total_before = total_after = total_files = 0

    for split in args.splits:
        seg_dir = root / f"{split}_seg"
        out_dir = root / f"{split}_seg_motion"
        seg_files = sorted(seg_dir.glob("*.pkl"))
        if args.limit:
            seg_files = seg_files[: args.limit]

        print(f"[{split}] {len(seg_files)} segment files", file=sys.stderr)
        for i, seg_path in enumerate(seg_files):
            out_path = out_dir / seg_path.name
            try:
                before, after = process_one(seg_path, out_path)
            except Exception as e:
                print(f"[{split}] FAILED {seg_path.name}: {e}", file=sys.stderr)
                continue
            total_before += before
            total_after += after
            total_files += 1
            if (i + 1) % 200 == 0 or (i + 1) == len(seg_files):
                print(
                    f"[{split}] {i + 1}/{len(seg_files)} done "
                    f"(running total {total_before / 1e9:.2f}GB -> {total_after / 1e9:.2f}GB)",
                    file=sys.stderr,
                )

    print(
        f"[extract_motion_only] {total_files} files: "
        f"{total_before / 1e9:.2f}GB -> {total_after / 1e9:.2f}GB",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
