"""
Characterization study, stage 1: convert every retargeted clip into ONE PHC motion file.

Reuses convert_step5_to_phc.convert() unchanged -- the Step 6C recipe that was validated by the
6C round trip -- so no conversion logic is duplicated or re-derived here.

Inputs:  scripts/quest_smpl_sequence_fit{,_walking}.json      (Steps 5, reused as-is)
         data/quest_smpl_retarget/charstudy/<tag>.json         (this study's new retargets)
Output:  data/quest_smpl_retarget/charstudy/charstudy_all_phc.pkl
         data/quest_smpl_retarget/charstudy/manifest.json      (tag -> key, source clip, frames)

Run:
    LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH python scripts/charstudy_convert.py
"""
import json
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "scripts"))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "phc"))

import isaacgym  # noqa: F401,E402
import joblib  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

import convert_step5_to_phc as C  # noqa: E402

CHAR = os.path.join(REPO, "data/quest_smpl_retarget/charstudy")
OUT_PKL = os.path.join(CHAR, "charstudy_all_phc.pkl")
MANIFEST = os.path.join(CHAR, "manifest.json")

# tag -> (retarget json, coarse activity class). standing/walking are the Step 5/6E clips, reused
# without rerunning IK; the rest are this study's new retargets.
CLIPS = [
    ("standing",    os.path.join(REPO, "scripts/quest_smpl_sequence_fit.json"),         "static"),
    ("gestures",    os.path.join(CHAR, "gestures.json"),                                "upper-body"),
    ("whiteboard",  os.path.join(CHAR, "whiteboard.json"),                              "upper-body"),
    ("rom",         os.path.join(CHAR, "rom.json"),                                     "transition"),
    ("stretching",  os.path.join(CHAR, "stretching.json"),                              "transition"),
    ("walking",     os.path.join(REPO, "scripts/quest_smpl_sequence_fit_walking.json"), "locomotion"),
    ("jogging",     os.path.join(CHAR, "jogging.json"),                                 "locomotion"),
    ("sports",      os.path.join(CHAR, "sports.json"),                                  "dynamic"),
    ("dancing",     os.path.join(CHAR, "dancing.json"),                                 "dynamic"),
    ("martialarts", os.path.join(CHAR, "martialarts.json"),                             "dynamic"),
]


def main():
    from smpl_sim.smpllib.smpl_joint_names import SMPL_BONE_ORDER_NAMES as jn
    tree, tree_path = C.build_upright_tree()
    s2m = [jn.index(q) for q in C.MUJOCO_NAMES]
    print(f"skeleton: {tree_path}")

    merged, manifest, missing = {}, {}, []
    for tag, js, cls in CLIPS:
        if not os.path.isfile(js):
            missing.append(tag)
            continue
        # convert() writes its own per-clip pkl into C.OUT_DIR; point that at the study folder
        C.OUT_DIR = CHAR
        motion, pkl, key, src = C.convert(tag, js, tree, s2m)
        # give the merged entry a stable, tag-derived key so the rollout dump is unambiguous
        merged[f"cs_{tag}"] = motion
        manifest[f"cs_{tag}"] = {"tag": tag, "class": cls,
                                 "source_clip": json.load(open(js))["motion_file"],
                                 "retarget_json": os.path.relpath(js, REPO),
                                 "frames": int(src["T"]), "fps": float(src["fps"])}
        print(f"  {tag:12s} {cls:11s} {src['T']:4d} frames  <- "
              f"{os.path.basename(json.load(open(js))['motion_file'])}")

    if missing:
        print(f"\nMISSING retargets (skipped): {missing}")
    joblib.dump(merged, OUT_PKL)
    json.dump(manifest, open(MANIFEST, "w"), indent=2)
    print(f"\n{len(merged)} clips -> {os.path.relpath(OUT_PKL, REPO)}")
    print(f"manifest -> {os.path.relpath(MANIFEST, REPO)}")
    return 0 if not missing else 1


if __name__ == "__main__":
    sys.exit(main())
