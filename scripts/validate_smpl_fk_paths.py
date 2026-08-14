"""
STEP 6B of the Quest->SMPL retargeting work: verify that the DOWNSTREAM SMPL FK path used by the
PHC pipeline reconstructs the Step 5 retargeted sequences identically to the validated retargeter.

    Step 5 exported SMPL parameters  (root trans, root orient AA, body_pose AA)
        |
        +--> path A: scripts/retarget_quest_to_smpl.SmplSkeletonFK      (validated retargeter)
        |            float64, LOCAL axis-angle in, composes the chain itself via aa_to_mat
        |
        +--> path B: phc/learning/distill_agent._fk_positions            (downstream / PHC)
                     float32, GLOBAL quaternions (xyzw) in, offsets from
                     _load_smpl_rest_offsets (the Step 6A-corrected loader)

The two FK routines take DIFFERENT inputs by design, so the local->global rotation composition is
done with scipy (a third implementation, independent of both) to avoid the test quietly validating
one FK against itself.

Path B is run three ways, which is what separates "precision" from "real mismatch":

  B-shipped   float32 offsets, float32 arithmetic   -- exactly what PHC runs
  B-widened   float32 offsets cast to float64       -- removes arithmetic round-off only.
              NOTE _load_smpl_rest_offsets returns a float32 tensor, so widening it AFTERWARDS
              does not give a float64 pipeline: the offsets stay quantized at ~1.4e-8 m, and
              that quantization (not the FK) is what this run's residual measures.
  B-exact     retargeter's own float64 offsets      -- same _fk_positions code, so any residual
              here can ONLY come from joint order / rotation convention / root transform /
              coordinate frame. This is the decisive test; it must hit machine epsilon.

Does NOT rerun IK, alter the Step 5 exports, or touch the Step 6A fix.

Run:
    LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH python scripts/validate_smpl_fk_paths.py
"""
import json
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "scripts"))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "phc"))
sys.path.insert(0, os.path.join(REPO, "phc", "phc"))  # phc's internal `import learning.…`

import isaacgym  # noqa: F401,E402  -- distill_agent pulls this in, and it MUST precede torch
import joblib  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import warnings  # noqa: E402
import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
# The discrepancy panels are log-scaled over ~1e-5..1e-3 mm; matplotlib's minor-tick locator
# warns and dumps its whole tick array, which buries the actual report in megabytes of numbers.
warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")
from scipy.spatial.transform import Rotation as Rot  # noqa: E402

CORR_JSON = os.path.join(REPO, "scripts/quest_smpl_correspondence.json")
SMPLH_NPZ = os.path.join(REPO, "support_data/body_models/smplh/neutral/model.npz")
SEQS = [("standing", os.path.join(REPO, "scripts/quest_smpl_sequence_fit.json")),
        ("walking", os.path.join(REPO, "scripts/quest_smpl_sequence_fit_walking.json"))]
OUT_PNG = os.path.join(REPO, "scripts/quest_smpl_fk_path_validation.png")
OUT_JSON = os.path.join(REPO, "scripts/quest_smpl_fk_path_validation.json")

# float32 storage of the offsets gives ~1e-7 m per bone; a 5-deep chain accumulates under 1e-5 m.
TOL_F32_M = 1e-5
# Offsets quantized to float32 (~1.4e-8 m) then widened -- the floor here is that quantization.
TOL_WIDE_M = 1e-7
# Identical offsets in float64: nothing left but the FK itself, so this must be machine epsilon.
# A convention mismatch (joint order, rotation convention, root transform, frame) would show up
# here at centimetre scale, ~10 orders of magnitude above this bar.
TOL_EXACT_M = 1e-12

PELVIS, L_HIP, R_HIP, HEAD, L_WRIST, R_WRIST, L_HAND, R_HAND = 0, 1, 2, 15, 20, 21, 22, 23
# Structural pairs are rigidly attached to pelvis/chest and CANNOT cross in any real pose, so a
# left/right swap is guaranteed to show up here on every frame. Distal pairs legitimately cross
# during gait (a narrow stride swings one foot past the other), so for those the sequence MEAN
# sign is the swap detector and per-frame crossings are reported as a motion characteristic.
LR_STRUCTURAL = [(1, 2, "hip"), (16, 17, "shoulder")]
LR_DISTAL = [(4, 5, "knee"), (7, 8, "ankle"), (10, 11, "foot"),
             (18, 19, "elbow"), (20, 21, "wrist"), (22, 23, "hand")]

FAILS = []


def check(name, ok, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  — {detail}" if detail else ""))
    if not ok:
        FAILS.append(name)


def fk_path_a(off64, parents, root_aa, body_aa, trans):
    """Validated Step 5 retargeter FK, frame by frame (its signature is single-frame)."""
    from retarget_quest_to_smpl import SmplSkeletonFK
    fk = SmplSkeletonFK(off64, parents, torch.device("cpu"))
    out = np.empty((len(trans), 24, 3))
    for t in range(len(trans)):
        p, _ = fk(torch.tensor(root_aa[t], dtype=torch.float64),
                  torch.tensor(body_aa[t], dtype=torch.float64),
                  torch.tensor(trans[t], dtype=torch.float64))
        out[t] = p.numpy()
    return out


def global_quats_xyzw(root_aa, body_aa, parents):
    """LOCAL axis-angle -> GLOBAL quaternions (xyzw), composed with scipy: a third implementation,
    independent of both FK paths under test."""
    aa = np.concatenate([root_aa[:, None, :], body_aa], axis=1)      # (T,24,3)
    T = aa.shape[0]
    Rl = Rot.from_rotvec(aa.reshape(-1, 3)).as_matrix().reshape(T, 24, 3, 3)
    Rg = np.empty_like(Rl)
    Rg[:, 0] = Rl[:, 0]
    for i in range(1, 24):
        Rg[:, i] = Rg[:, parents[i]] @ Rl[:, i]
    return Rot.from_matrix(Rg.reshape(-1, 3, 3)).as_quat().reshape(T, 24, 4)


def fk_path_b(trans, quat_xyzw, dtype, off64=None):
    """Downstream PHC FK: distill_agent._fk_positions. Offsets come from the shipped
    _load_smpl_rest_offsets unless off64 is given (the B-exact variant, which substitutes the
    retargeter's float64 offsets so only the FK algorithm itself is under test)."""
    from phc.learning.distill_agent import _fk_positions, _load_smpl_rest_offsets
    if off64 is None:
        off = _load_smpl_rest_offsets(SMPLH_NPZ, torch.device("cpu")).to(dtype)
    else:
        off = torch.tensor(off64, dtype=dtype)
    return _fk_positions(torch.tensor(trans, dtype=dtype),
                         torch.tensor(quat_xyzw, dtype=dtype),
                         off, list(range(24))).numpy().astype(np.float64)


def lateral_signs(P):
    """Per-frame lateral coordinate of each joint in the BODY's own frame (+ = anatomical left),
    so the left/right check survives arbitrary root yaw."""
    lat = P[:, L_HIP] - P[:, R_HIP]
    lat /= np.linalg.norm(lat, axis=-1, keepdims=True)
    return np.einsum("tjc,tc->tj", P - P[:, PELVIS:PELVIS + 1], lat)


def analyse(tag, path, names, parents, off64):
    d = json.load(open(path))
    trans = np.array(d["smpl_trans"], dtype=np.float64)
    root_aa = np.array(d["smpl_root_orient_aa"], dtype=np.float64)
    body_aa = np.array(d["smpl_body_pose_aa"], dtype=np.float64)
    T, fps = len(trans), d.get("fps", 30.0)

    # Cross-check the export length against the source clip. The `_{start}_{end}` suffix in these
    # filenames is INCLUSIVE (verified across the dataset), so `_0_426` is 427 frames, not 426 --
    # reading it as exclusive is what makes standing (_0_425 -> 426) and walking (_0_426 -> 427)
    # look like they should match when they genuinely differ by one source frame.
    src = joblib.load(os.path.join(REPO, d["motion_file"]))
    T_src = len(src[list(src.keys())[0]]["pose_quat"])
    lo, hi = (int(v) for v in os.path.basename(d["motion_file"])[:-4].rsplit("_", 2)[-2:])

    print("=" * 84)
    print(f"{tag.upper()}  —  {T} frames @ {fps:g} fps   ({os.path.basename(path)})")
    print("=" * 84)
    print(f"  source clip {os.path.basename(d['motion_file'])}")
    print(f"    name suffix _{lo}_{hi} (inclusive) => {hi - lo + 1} frames; "
          f"source pose_quat = {T_src}; Step 5 export = {T}")
    check(f"{tag}: export length matches source clip — no frame inserted or dropped",
          T == T_src == hi - lo + 1, f"{T} == {T_src} == {hi - lo + 1}")
    res_frames = {"source_frames": T_src, "export_frames": T,
                  "suffix_inclusive_frames": hi - lo + 1}

    A = fk_path_a(off64, parents, root_aa, body_aa, trans)
    q = global_quats_xyzw(root_aa, body_aa, parents)
    B32 = fk_path_b(trans, q, torch.float32)
    Bwide = fk_path_b(trans, q, torch.float64)
    Bexact = fk_path_b(trans, q, torch.float64, off64=off64)

    e32 = np.linalg.norm(A - B32, axis=-1)                     # (T,24) metres
    ewide = np.linalg.norm(A - Bwide, axis=-1)
    eexact = np.linalg.norm(A - Bexact, axis=-1)
    res = {"frames": T, "fps": fps, **res_frames}

    print(f"  A vs B-shipped (f32 offsets, f32 math): mean {e32.mean()*1e3:.3e} mm   "
          f"max {e32.max()*1e3:.3e} mm")
    print(f"  A vs B-widened (f32 offsets, f64 math): mean {ewide.mean()*1e3:.3e} mm   "
          f"max {ewide.max()*1e3:.3e} mm   <- floored by float32 offset quantization")
    print(f"  A vs B-exact   (f64 offsets, f64 math): mean {eexact.mean()*1e3:.3e} mm   "
          f"max {eexact.max()*1e3:.3e} mm   <- FK algorithm alone")
    res.update(mean_mm_f32=e32.mean() * 1e3, max_mm_f32=e32.max() * 1e3,
               mean_mm_widened=ewide.mean() * 1e3, max_mm_widened=ewide.max() * 1e3,
               mean_mm_exact=eexact.mean() * 1e3, max_mm_exact=eexact.max() * 1e3)

    check(f"{tag}: as-shipped discrepancy within float32 storage tolerance",
          e32.max() < TOL_F32_M, f"max {e32.max()*1e3:.3e} mm < {TOL_F32_M*1e3:g} mm")
    check(f"{tag}: widened discrepancy tracks float32 offset quantization",
          ewide.max() < TOL_WIDE_M, f"max {ewide.max()*1e3:.3e} mm < {TOL_WIDE_M*1e3:g} mm")
    check(f"{tag}: DECISIVE — identical offsets give machine epsilon, so no joint-order / "
          f"rotation-convention / root-transform / coordinate-frame mismatch",
          eexact.max() < TOL_EXACT_M, f"max {eexact.max():.3e} m < {TOL_EXACT_M:g} m")

    print("  per-joint MAX discrepancy over all frames (B-shipped / B-exact), mm:")
    pj32, pj64 = e32.max(0), eexact.max(0)
    for i in range(0, 24, 3):
        print("    " + "   ".join(f"{i+k:2d} {names[i+k]:<12s}{pj32[i+k]*1e3:9.2e}/{pj64[i+k]*1e3:8.2e}"
                                  for k in range(3) if i + k < 24))
    res["per_joint_max_mm_shipped"] = {names[i]: pj32[i] * 1e3 for i in range(24)}
    res["per_joint_max_mm_exact"] = {names[i]: pj64[i] * 1e3 for i in range(24)}

    print("  joints called out by the Step 6B spec (max over frames, B-shipped / B-exact), mm:")
    for i, lbl in [(PELVIS, "root/pelvis"), (HEAD, "head"), (L_WRIST, "L_Wrist"),
                   (R_WRIST, "R_Wrist"), (L_HAND, "L_Hand"), (R_HAND, "R_Hand")]:
        print(f"    {lbl:<12s} {pj32[i]*1e3:9.3e} / {pj64[i]*1e3:8.3e}")
    check(f"{tag}: root/pelvis identical (it is passed through, never FK'd)",
          pj64[PELVIS] == 0.0, f"{pj64[PELVIS]:.1e} m")
    check(f"{tag}: L_Hand and R_Hand agree between paths",
          max(pj32[L_HAND], pj32[R_HAND]) < TOL_F32_M,
          f"L {pj32[L_HAND]*1e3:.2e} mm, R {pj32[R_HAND]*1e3:.2e} mm")

    # ---- geometry sanity, on the downstream path's own output ----
    check(f"{tag}: all downstream values finite", np.isfinite(B32).all())

    up = B32[:, HEAD, 2] - B32[:, PELVIS, 2]
    extent = B32.reshape(-1, 3).ptp(0)
    print(f"  world extent (X,Y,Z) = {np.round(extent,3)} m; "
          f"head-above-pelvis in Z: min {up.min():.3f} m, mean {up.mean():.3f} m")
    check(f"{tag}: native Z-up — Z carries the vertical (head above pelvis every frame)",
          up.min() > 0.2, f"min {up.min():.3f} m")
    check(f"{tag}: upright — Z spans a human stature, not a floor-plane sequence",
          0.9 < extent[2] < 2.6, f"Z extent {extent[2]:.3f} m")

    latA, latB = lateral_signs(A), lateral_signs(B32)
    print("  left/right separation in the body frame (+ = L correctly left of R), mm:")
    struct_bad, cross = [], {}
    for l, r, nm in LR_STRUCTURAL + LR_DISTAL:
        sep = latB[:, l] - latB[:, r]
        n_neg = int((sep < 0).sum())
        kind = "structural" if (l, r, nm) in LR_STRUCTURAL else "distal"
        print(f"    {nm:<9s} {kind:<10s} mean {sep.mean()*1e3:+8.1f}  min {sep.min()*1e3:+8.1f}"
              f"  crossing frames {n_neg:3d}/{T}")
        if kind == "structural" and n_neg:
            struct_bad.append((nm, n_neg))
        if n_neg:
            cross[nm] = {"frames": n_neg, "deepest_mm": float(sep.min() * 1e3)}
        if sep.mean() <= 0:
            struct_bad.append((nm, "mean sign inverted"))
    check(f"{tag}: no left/right swap — structural pairs never cross, all pairs mean-correct",
          not struct_bad, "violations: " + (str(struct_bad) if struct_bad else "none"))
    check(f"{tag}: left/right ordering identical in both paths (crossings are the motion, "
          f"not the FK)",
          np.array_equal(np.sign(latA - latA[:, PELVIS:PELVIS + 1]),
                         np.sign(latB - latB[:, PELVIS:PELVIS + 1])))
    res["distal_crossings"] = cross

    # ---- temporal continuity ----
    stepA = np.linalg.norm(np.diff(A, axis=0), axis=-1).max(1)
    stepB = np.linalg.norm(np.diff(B32, axis=0), axis=-1).max(1)
    med = float(np.median(stepB))
    jumps = np.where(stepB > max(10 * med, 0.05))[0]
    print(f"  per-frame max joint step: A {stepA.max()*1e3:.1f} mm, B {stepB.max()*1e3:.1f} mm "
          f"(median {med*1e3:.1f} mm) -> {stepB.max()*fps:.2f} m/s peak")
    check(f"{tag}: no frame-to-frame discontinuities introduced downstream",
          len(jumps) == 0, f"{len(jumps)} jump(s)" + (f" at {jumps[:5].tolist()}" if len(jumps) else ""))
    check(f"{tag}: discrepancy is stationary in time (no drift/blow-up)",
          e32.max(1).std() < TOL_F32_M, f"std over frames {e32.max(1).std()*1e3:.2e} mm")
    res.update(max_step_mm_A=stepA.max() * 1e3, max_step_mm_B=stepB.max() * 1e3,
               n_discontinuities=int(len(jumps)),
               head_above_pelvis_min_m=float(up.min()), z_extent_m=float(extent[2]))
    return A, B32, e32, res


def overlay(fig, gs, row, tag, A, B, e32, names, parents):
    T = len(A)
    for col, (t, lbl) in enumerate([(0, "start"), (T // 2, "middle"), (T - 1, "end")]):
        ax = fig.add_subplot(gs[row, col], projection="3d")
        for P, c, ls, mk, nm in [(A, "#1f77b4", "-", None, "A: retargeter"),
                                 (B, "#d62728", "--", "o", "B: distill_agent")]:
            for i in range(1, 24):
                p = parents[i]
                ax.plot(*zip(P[t, p], P[t, i]), color=c, ls=ls, lw=1.6, alpha=.85)
            ax.scatter(*P[t].T, color=c, s=9 if mk else 5, marker=mk or "o",
                       depthshade=False, label=nm)
        c = A[t].mean(0)
        ax.set_xlim(c[0] - .9, c[0] + .9); ax.set_ylim(c[1] - .9, c[1] + .9)
        ax.set_zlim(c[2] - .9, c[2] + .9)
        ax.set_title(f"{tag} — {lbl} (frame {t})\nmax err {e32[t].max()*1e3:.2e} mm", fontsize=8)
        ax.tick_params(labelsize=5)
        ax.set_xlabel("X", fontsize=6); ax.set_ylabel("Y", fontsize=6); ax.set_zlabel("Z (up)", fontsize=6)
        if col == 0:
            ax.legend(fontsize=6, loc="upper left")

    ax = fig.add_subplot(gs[row, 3])
    ax.plot(e32.max(1) * 1e3, lw=.9, color="#d62728", label="max over joints")
    ax.plot(e32.mean(1) * 1e3, lw=.9, color="#1f77b4", label="mean over joints")
    ax.axhline(TOL_F32_M * 1e3, color="k", ls=":", lw=.9, label="float32 tolerance")
    ax.set_yscale("log"); ax.set_xlabel("frame", fontsize=7)
    ax.set_ylabel("|A − B| (mm)", fontsize=7)
    ax.set_title(f"{tag} — path discrepancy vs frame", fontsize=8)
    ax.tick_params(labelsize=6); ax.legend(fontsize=6)


def main():
    from smpl_sim.smpllib.smpl_joint_names import SMPL_BONE_ORDER_NAMES as names
    from retarget_quest_to_smpl import load_smpl_rest_offsets

    corr = json.load(open(CORR_JSON))
    parents = corr["smpl_parents"]
    off64 = load_smpl_rest_offsets(corr)

    # The two loaders must already agree; Step 6A is what made that true.
    from phc.learning.distill_agent import _load_smpl_rest_offsets
    off_dn = _load_smpl_rest_offsets(SMPLH_NPZ, torch.device("cpu")).numpy().astype(np.float64)
    print("=" * 84)
    print("0. REST OFFSETS — retargeter vs downstream loader")
    print("=" * 84)
    doff = np.linalg.norm(off64[1:] - off_dn[1:], axis=-1)
    print(f"  row map: retargeter {corr['smplh_row_for_smpl24'][-2:]}  "
          f"downstream {[22, 37]}   (indices 22,23)")
    print(f"  max |Δoffset| over joints 1..23: {doff.max()*1e3:.3e} mm")
    check("rest offsets agree between the two pipelines (joints 1..23)", doff.max() < TOL_F32_M,
          f"{doff.max()*1e3:.2e} mm")
    check("offsets[0] differs harmlessly (retargeter zeroes it; _fk_positions never reads it)",
          np.allclose(off64[0], 0.0) and not np.allclose(off_dn[0], 0.0),
          f"A {np.round(off64[0],3)} vs B {np.round(off_dn[0],3)}")

    fig = plt.figure(figsize=(17, 8.4))
    gs = fig.add_gridspec(2, 4, width_ratios=[1, 1, 1, 1.15], wspace=.22, hspace=.28)
    out = {}
    for row, (tag, path) in enumerate(SEQS):
        A, B, e32, res = analyse(tag, path, names, parents, off64)
        overlay(fig, gs, row, tag, A, B, e32, names, parents)
        out[tag] = res

    fig.suptitle("Step 6B — Step 5 SMPL exports through both FK paths "
                 "(A: validated retargeter, float64 · B: distill_agent._fk_positions, float32)",
                 fontsize=11)
    fig.savefig(OUT_PNG, dpi=135, bbox_inches="tight")
    out["passed"] = not FAILS
    out["failed_checks"] = FAILS
    json.dump(out, open(OUT_JSON, "w"), indent=2, default=float)

    print("=" * 84)
    print(f"  overlay -> {os.path.relpath(OUT_PNG, REPO)}")
    print(f"  metrics -> {os.path.relpath(OUT_JSON, REPO)}")
    if FAILS:
        print(f"STEP 6B: FAIL — {len(FAILS)} check(s): {FAILS}")
        return 1
    print("STEP 6B: PASS — both FK paths reproduce the same SMPL motion.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
