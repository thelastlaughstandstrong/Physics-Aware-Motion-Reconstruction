"""
STEP 6D pre-flight: verify PHC's OWN motion loader ingests the Step 6C files correctly, before
committing a GPU rollout.

Runs the real code path -- MotionLibSMPL.load_motion_with_skeleton, the same static method the
env calls -- on CPU, including fix_trans_height with the real SMPL mesh parsers. Checks:
frame count, first-frame root position/orientation, upright orientation, joint ordering, and the
first-frame reference pose against the validated Step 5/6B motion.

flags.im_eval/flags.test are set True so the loader's random-heading augmentation is skipped,
exactly as in the eval recipe (learning=im_mcp ... test=True im_eval=True).

Run:
    LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH python scripts/preflight_phc_load.py
"""
import json
import os
import sys
from types import SimpleNamespace

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "scripts"))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "phc"))
sys.path.insert(0, os.path.join(REPO, "phc", "phc"))

import isaacgym  # noqa: F401,E402
import joblib  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from scipy.spatial.transform import Rotation as sRot  # noqa: E402
from poselib.poselib.skeleton.skeleton3d import SkeletonTree  # noqa: E402

from convert_step5_to_phc import (MUJOCO_NAMES, Q_UPRIGHT, build_upright_tree,  # noqa: E402
                                  reference_fk)

SEQS = [("standing", "data/quest_smpl_retarget/quest_smpl_standing_phc.pkl",
         "scripts/quest_smpl_sequence_fit.json"),
        ("walking", "data/quest_smpl_retarget/quest_smpl_walking_phc.pkl",
         "scripts/quest_smpl_sequence_fit_walking.json")]
PELVIS, HEAD = 0, 15
FAILS, LINES = [], []


def PR(s=""):
    print(s)
    LINES.append(s)


def check(name, ok, detail=""):
    PR(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  — {detail}" if detail else ""))
    if not ok:
        FAILS.append(name)


def main():
    from smpl_sim.smpllib.smpl_joint_names import SMPL_BONE_ORDER_NAMES as jn
    from smpl_sim.smpllib.smpl_parser import SMPL_Parser
    from phc.utils.flags import flags
    from phc.utils.motion_lib_base import FixHeightMode
    from phc.utils.motion_lib_smpl import MotionLibSMPL
    from retarget_quest_to_smpl import load_smpl_rest_offsets

    flags.im_eval, flags.test, flags.real_traj = True, True, False
    m2s = [MUJOCO_NAMES.index(q) for q in jn]
    corr = json.load(open(os.path.join(REPO, "scripts/quest_smpl_correspondence.json")))
    parents, soff = corr["smpl_parents"], load_smpl_rest_offsets(corr)

    tree, tree_path = build_upright_tree()
    smpl_dir = os.path.join(REPO, "phc/data/smpl")
    parsers = {i: SMPL_Parser(model_path=smpl_dir, gender=g)
               for i, g in enumerate(["neutral", "male", "female"])}
    cfg = SimpleNamespace(max_length=-1, fix_height=FixHeightMode.full_fix, multi_thread=False)

    PR("=" * 88)
    PR("STEP 6D PRE-FLIGHT — PHC's own loader on the Step 6C files")
    PR("=" * 88)
    PR(f"  loader   MotionLibSMPL.load_motion_with_skeleton (flags.im_eval=True, flags.test=True)")
    PR(f"  skeleton {tree_path}")
    PR(f"  fix_height {cfg.fix_height}   mesh parsers from phc/data/smpl")
    check("skeleton node order is the MUJOCO order", list(tree.node_names) == MUJOCO_NAMES)

    out = {}
    for tag, pkl, step5 in SEQS:
        entry = joblib.load(os.path.join(REPO, pkl))
        key = list(entry.keys())[0]
        data = entry[key]
        gender_beta = torch.zeros(17)                     # [gender, *betas16] -> neutral

        res_map = MotionLibSMPL.load_motion_with_skeleton(
            [0], [data], [tree], [gender_beta], parsers, cfg, None, 0)
        _, motion = res_map[0]

        d5 = json.load(open(os.path.join(REPO, step5)))
        src = dict(T=len(d5["smpl_trans"]), fps=float(d5["fps"]),
                   trans=np.array(d5["smpl_trans"], dtype=np.float64),
                   root_aa=np.array(d5["smpl_root_orient_aa"], dtype=np.float64),
                   body_aa=np.array(d5["smpl_body_pose_aa"], dtype=np.float64))
        P_ref = reference_fk(src, parents, soff)
        P = motion.global_translation.numpy().astype(np.float64)[:, m2s]

        PR("")
        PR("-" * 88)
        PR(f"{tag.upper()}  ({os.path.basename(pkl)}, key '{key}')")
        PR("-" * 88)

        # 1. frame count
        PR(f"  frames {len(P)} | fps {motion.fps} | duration {(len(P)-1)/motion.fps:.4f} s")
        check(f"{tag}: frame count as expected", len(P) == src["T"], f"{len(P)} == {src['T']}")
        check(f"{tag}: fps as expected", float(motion.fps) == src["fps"])

        # 2. first-frame root position / orientation
        # fix_trans_height shifts Z only. Measured on non-hand joints: the two hand leaves carry
        # PHC's own 40.9 mm hand bone (Step 6C), which is a skeleton difference, not a shift.
        non_hand = np.setdiff1d(np.arange(24), [22, 23])
        dz = float(np.median((P[:, non_hand, 2] - P_ref[:, non_hand, 2])))
        r0, r0_ref = P[0, PELVIS], P_ref[0, PELVIS]
        PR(f"  first-frame root XY  sim {np.round(r0[:2],4)}  ref {np.round(r0_ref[:2],4)}")
        PR(f"  fix_trans_height applied a uniform Z shift of {dz*1e3:+.1f} mm "
           f"(floor contact; expected and applied to AMASS too)")
        check(f"{tag}: first-frame root XY matches the reference",
              np.abs(r0[:2] - r0_ref[:2]).max() < 1e-4,
              f"max {np.abs(r0[:2]-r0_ref[:2]).max()*1e3:.3f} mm")
        zres = float(np.abs((P[:, non_hand, 2] - P_ref[:, non_hand, 2]) - dz).max())
        check(f"{tag}: the only vertical change is a single rigid Z shift (hand leaves aside)",
              zres < 5e-4, f"residual {zres*1e3:.3f} mm; hands differ by "
              f"{np.abs((P[:,[22,23],2]-P_ref[:,[22,23],2])-dz).max()*1e3:.1f} mm (known bone)")

        q0 = motion.global_rotation.numpy()[0, 0]
        aa_back = (sRot.from_quat(q0) * Q_UPRIGHT).as_rotvec()
        ang = np.degrees(np.linalg.norm(
            (sRot.from_rotvec(aa_back) * sRot.from_rotvec(src["root_aa"][0]).inv()).as_rotvec()))
        PR(f"  first-frame root orientation, recovered vs Step 5: {ang:.3e}°")
        check(f"{tag}: first-frame root orientation matches the reference", ang < 1e-3,
              f"{ang:.2e}°")

        # 3. upright
        up = P[:, HEAD, 2] - P[:, PELVIS, 2]
        PR(f"  head-above-pelvis in Z: min {up.min():.3f} m | lowest joint Z {P[...,2].min():+.4f} m")
        check(f"{tag}: upright, native Z-up", up.min() > 0.2, f"min {up.min():.3f} m")

        # 4. joint ordering + 5. first-frame reference pose
        e0 = np.linalg.norm((P[0] - P[0, PELVIS]) - (P_ref[0] - P_ref[0, PELVIS]), axis=-1)
        e = np.linalg.norm((P - P[:, PELVIS:PELVIS + 1])
                           - (P_ref - P_ref[:, PELVIS:PELVIS + 1]), axis=-1)
        PR(f"  first-frame pose vs Step 5 (root-relative): mean {e0.mean()*1e3:.3f} mm "
           f"max {e0.max()*1e3:.3f} mm  (worst joint {jn[int(e0.argmax())]})")
        PR(f"  whole sequence, excluding hand leaves: max {e[:,non_hand].max()*1e3:.3f} mm")
        check(f"{tag}: first-frame reference pose matches, hands aside",
              e0[non_hand].max() < 0.5e-3, f"max {e0[non_hand].max()*1e3:.3f} mm")
        check(f"{tag}: joint ordering correct (a permutation error would show cm-scale error)",
              e[:, non_hand].max() < 0.5e-3, f"max {e[:,non_hand].max()*1e3:.3f} mm")

        # velocities PHC derived
        v = motion.global_velocity.numpy()
        check(f"{tag}: PHC-derived velocities finite and plausible",
              np.isfinite(v).all() and np.linalg.norm(v, axis=-1).max() < 20,
              f"max |v| {np.linalg.norm(v,axis=-1).max():.2f} m/s")
        out[tag] = dict(frames=int(len(P)), fps=float(motion.fps),
                        z_shift_mm=float(dz * 1e3),
                        first_frame_max_mm=float(e0[non_hand].max() * 1e3),
                        seq_max_mm=float(e[:, non_hand].max() * 1e3),
                        root_orient_deg=float(ang),
                        max_vel=float(np.linalg.norm(v, axis=-1).max()),
                        lowest_joint_z=float(P[..., 2].min()))

    PR("")
    PR("=" * 88)
    out["passed"] = not FAILS
    out["failed_checks"] = FAILS
    json.dump(out, open(os.path.join(REPO, "scripts/quest_smpl_phc_preflight.json"), "w"),
              indent=2, default=float)
    if FAILS:
        PR(f"PRE-FLIGHT: FAIL — {len(FAILS)} check(s): {FAILS}")
        return 1
    PR("PRE-FLIGHT: PASS — PHC loads both files correctly; safe to roll out.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
