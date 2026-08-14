"""
Regression check for the SMPL-H R_Hand indexing fix in
phc/phc/learning/distill_agent.py::_load_smpl_rest_offsets.

Root cause it guards against: SMPL-H stores 52 joints (0-21 body, 22-36 LEFT hand, 37-51 RIGHT
hand). Taking `J_regressor[:24]` therefore returns the 22 body joints plus the first TWO LEFT-hand
joints, so SMPL-24 index 23 ("R_Hand") was populated from a left-hand joint -- placing the right
hand on the left side of the body with a bogus 1.5 m forearm bone.

Checks, on the neutral rest pose AND on real frames from the Step 5 Quest->SMPL exports:
  1. L_Hand attaches to L_Wrist and R_Hand to R_Wrist (correct chain, plausible bone length)
  2. no ~1.5 m offset remains anywhere
  3. left and right hands are on OPPOSITE sides (no swap, no both-on-one-side)
  4. every non-hand joint offset is bit-identical to the pre-fix values (fix is surgical)

Run:  python scripts/test_smpl_rest_offsets.py
Exit code 0 = all pass.
"""
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "scripts"))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "phc"))
sys.path.insert(0, os.path.join(REPO, "phc", "phc"))  # phc's internal `import learning.…`

import isaacgym  # noqa: F401  -- distill_agent pulls this in, and it MUST precede torch
import numpy as np
import torch

SMPLH_NPZ = os.path.join(REPO, "support_data/body_models/smplh/neutral/model.npz")
SMPL_PARENTS = [-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19, 20, 21]
L_WRIST, R_WRIST, L_HAND, R_HAND = 20, 21, 22, 23

FAILS = []


def check(name, ok, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  — {detail}" if detail else ""))
    if not ok:
        FAILS.append(name)


def buggy_offsets():
    """Reproduce the pre-fix behaviour for comparison only. Cast to float32 at the end so it
    matches what _load_smpl_rest_offsets stores -- otherwise float64-vs-float32 rounding shows
    up as ~1e-8 m 'changes' on every joint and masks what the fix actually touched."""
    bd = np.load(SMPLH_NPZ, allow_pickle=True)
    J = np.array(bd["J_regressor"])[:24] @ np.array(bd["v_template"])
    o = J.copy()
    for i in range(1, 24):
        o[i] = J[i] - J[SMPL_PARENTS[i]]
    return J, o.astype(np.float32).astype(np.float64)


# 1 micron: far below any meaningful skeleton change, far above float32 storage noise (~1e-8 m).
TOL = 1e-6


def fk(off, root_orient_aa, body_pose_aa, trans):
    """Minimal SMPL joint FK, matching distill_agent's own chain convention."""
    from scipy.spatial.transform import Rotation as Rot
    aa = np.concatenate([np.asarray(root_orient_aa)[None], np.asarray(body_pose_aa)], axis=0)
    Rl = Rot.from_rotvec(aa).as_matrix()
    Rg = [Rl[0]]
    pos = [np.asarray(trans, dtype=np.float64)]
    for i in range(1, 24):
        p = SMPL_PARENTS[i]
        Rg.append(Rg[p] @ Rl[i])
        pos.append(pos[p] + Rg[p] @ off[i])
    return np.stack(pos)


def main():
    from phc.learning.distill_agent import _load_smpl_rest_offsets

    fixed = _load_smpl_rest_offsets(SMPLH_NPZ, torch.device("cpu")).numpy().astype(np.float64)
    J_bug, off_bug = buggy_offsets()

    print("=" * 78)
    print("1. REST POSE")
    print("=" * 78)
    lb, rb = np.linalg.norm(off_bug[L_HAND]), np.linalg.norm(off_bug[R_HAND])
    lf, rf = np.linalg.norm(fixed[L_HAND]), np.linalg.norm(fixed[R_HAND])
    print(f"  L_Hand bone: buggy {lb:.4f} m -> fixed {lf:.4f} m")
    print(f"  R_Hand bone: buggy {rb:.4f} m -> fixed {rf:.4f} m")
    check("R_Hand bone length is anatomically plausible (< 0.25 m)", rf < 0.25, f"{rf:.4f} m")
    check("no ~1.5 m offset remains on ANY joint",
          float(np.linalg.norm(fixed, axis=-1).max()) < 0.5,
          f"max bone {np.linalg.norm(fixed, axis=-1).max():.4f} m")
    check("L/R hand bones are near-symmetric", abs(lf - rf) < 0.02, f"|{lf:.4f}-{rf:.4f}|")

    changed = [i for i in range(24) if np.linalg.norm(fixed[i] - off_bug[i]) > TOL]
    check("fix is surgical: only R_Hand's offset changed",
          changed == [R_HAND], f"changed indices {changed}")

    # rest-pose sides: SMPL rest has +X = left
    P_fix = fk(fixed, np.zeros(3), np.zeros((23, 3)), np.zeros(3))
    P_bug = fk(off_bug, np.zeros(3), np.zeros((23, 3)), np.zeros(3))
    print(f"  rest R_Hand x: buggy {P_bug[R_HAND][0]:+.4f} (WRONG SIDE) -> "
          f"fixed {P_fix[R_HAND][0]:+.4f}")
    check("rest: L_Hand and R_Hand on OPPOSITE sides",
          np.sign(P_fix[L_HAND][0]) != np.sign(P_fix[R_HAND][0]),
          f"L x={P_fix[L_HAND][0]:+.4f}, R x={P_fix[R_HAND][0]:+.4f}")
    check("rest: L_Hand on same side as L_Wrist",
          np.sign(P_fix[L_HAND][0]) == np.sign(P_fix[L_WRIST][0]))
    check("rest: R_Hand on same side as R_Wrist",
          np.sign(P_fix[R_HAND][0]) == np.sign(P_fix[R_WRIST][0]))

    # ---- posed frames from the Step 5 exports ----
    import json
    for tag, path in [("standing", os.path.join(REPO, "scripts/quest_smpl_sequence_fit.json")),
                      ("walking", os.path.join(REPO, "scripts/quest_smpl_sequence_fit_walking.json"))]:
        if not os.path.isfile(path):
            print(f"\n  (skipped {tag}: {os.path.basename(path)} not found)")
            continue
        d = json.load(open(path))
        t = len(d["smpl_body_pose_aa"]) // 2
        print("=" * 78)
        print(f"2. STEP 5 EXPORT — {tag}, frame {t}")
        print("=" * 78)
        Pf = fk(fixed, d["smpl_root_orient_aa"][t], d["smpl_body_pose_aa"][t], d["smpl_trans"][t])
        Pb = fk(off_bug, d["smpl_root_orient_aa"][t], d["smpl_body_pose_aa"][t], d["smpl_trans"][t])
        moved = np.linalg.norm(Pf - Pb, axis=-1)
        print(f"  R_Hand world: buggy {np.round(Pb[R_HAND],3)} -> fixed {np.round(Pf[R_HAND],3)}"
              f"   (moved {moved[R_HAND]:.3f} m)")
        # a hand must sit ~one bone length from ITS OWN wrist
        dl = np.linalg.norm(Pf[L_HAND] - Pf[L_WRIST])
        dr = np.linalg.norm(Pf[R_HAND] - Pf[R_WRIST])
        dr_bug = np.linalg.norm(Pb[R_HAND] - Pb[R_WRIST])
        print(f"  R_Hand-to-R_Wrist distance: buggy {dr_bug:.4f} m -> fixed {dr:.4f} m")
        check(f"{tag}: L_Hand attached to L_Wrist", dl < 0.25, f"{dl:.4f} m")
        check(f"{tag}: R_Hand attached to R_Wrist", dr < 0.25, f"{dr:.4f} m")
        check(f"{tag}: no left/right swap (R_Hand nearer R_Wrist than L_Wrist)",
              dr < np.linalg.norm(Pf[R_HAND] - Pf[L_WRIST]))
        check(f"{tag}: only R_Hand moved",
              set(np.where(moved > TOL)[0].tolist()) <= {R_HAND},
              f"moved joints {np.where(moved > TOL)[0].tolist()}, "
              f"max move elsewhere {np.delete(moved, R_HAND).max():.2e} m")

    print("=" * 78)
    if FAILS:
        print(f"FAILED: {len(FAILS)} check(s): {FAILS}")
        return 1
    print("ALL CHECKS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
