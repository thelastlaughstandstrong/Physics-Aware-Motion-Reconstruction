"""
Step 7 final sanity check on the rendered comparison videos.

Checks the things a render can silently get wrong: frame count, fps, duration, dropped/duplicated
frames, and that the rendered PNG count matches the motion's own frame count exactly. Geometry
(upright, left/right, offsets, ground contact) was already validated numerically upstream by
validate_addon_rig_report.py (54/54) -- this only confirms the render didn't alter what was
validated.
"""
import json
import os
import subprocess
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "scripts"))
import numpy as np                                                     # noqa: E402

RENDERS = os.path.join(REPO, "data/quest_smpl_retarget/renders")
EXPORT = os.path.join(REPO, "data/quest_smpl_retarget/blender_export")
CLIPS = ["sports", "gestures", "martialarts"]
FAILS, LINES = [], []


def PR(s=""):
    print(s)
    LINES.append(s)


def check(name, ok, detail=""):
    PR(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  — {detail}" if detail else ""))
    if not ok:
        FAILS.append(name)


def probe(path, entries):
    out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0",
                          "-show_entries", entries, "-of", "json", path],
                         capture_output=True, text=True).stdout
    return json.loads(out)


def main():
    PR("# Step 7 — render verification")
    PR("")
    out = {}
    for tag in CLIPS:
        mp4 = os.path.join(RENDERS, f"{tag}_comparison.mp4")
        seq = os.path.join(RENDERS, f"_{tag}_frames")
        mesh = np.load(os.path.join(EXPORT, f"mesh_{tag}.npz"), allow_pickle=True)
        T = len(mesh["pred_pos"])
        PR(f"## {tag}")
        PR("")
        if not os.path.isfile(mp4):
            check(f"{tag}: video exists", False, mp4)
            continue
        info = probe(mp4, "stream=nb_frames,r_frame_rate,width,height,duration")["streams"][0]
        nb = int(info.get("nb_frames", 0))
        num, den = info["r_frame_rate"].split("/")
        fps = float(num) / float(den)
        dur = float(info.get("duration", 0))
        pngs = len([f for f in os.listdir(seq) if f.endswith(".png")]) if os.path.isdir(seq) else 0
        PR(f"- {info['width']}x{info['height']}, **{nb} frames @ {fps:g} fps**, "
           f"duration **{dur:.3f} s**  (motion: {T} frames, {T/30:.3f} s)")
        PR(f"- PNG frames rendered: **{pngs}**")
        check(f"{tag}: frame count matches the motion exactly", nb == T, f"{nb} vs {T}")
        check(f"{tag}: no dropped/duplicated frames (PNG count == motion frames)",
              pngs == T, f"{pngs} vs {T}")
        check(f"{tag}: 30 fps", abs(fps - 30.0) < 1e-6, f"{fps:g}")
        check(f"{tag}: playback duration correct", abs(dur - T / 30.0) < 0.05,
              f"{dur:.3f} s vs {T/30.0:.3f} s")
        check(f"{tag}: presentation resolution", info["width"] >= 1920 and info["height"] >= 1080,
              f"{info['width']}x{info['height']}")
        stills = sorted(f for f in os.listdir(RENDERS) if f.startswith(f"{tag}_still"))
        check(f"{tag}: still frames produced", len(stills) >= 2, f"{len(stills)}: {stills}")
        PR("")
        out[tag] = dict(frames=nb, fps=fps, duration=dur, motion_frames=T, pngs=pngs,
                        resolution=f"{info['width']}x{info['height']}", stills=stills)
    PR("")
    PR(f"**{'PASSED' if not FAILS else 'FAILED'}**"
       + ("" if not FAILS else f" — {len(FAILS)}: {FAILS}"))
    json.dump({"clips": out, "passed": not FAILS, "failed": FAILS},
              open(os.path.join(REPO, "scripts/quest_smpl_step7_render_check.json"), "w"),
              indent=2, default=float)
    open(os.path.join(REPO, "scripts/quest_smpl_step7_render_check.md"), "w").write(
        "\n".join(LINES) + "\n")
    return 1 if FAILS else 0


if __name__ == "__main__":
    sys.exit(main())
