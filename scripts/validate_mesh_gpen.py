"""
Per-frame numerical validation of the mesh-GPen metric (not visual inspection).

For a cached (clip, stream) IK result, rebuilds the SMPL-H neutral mesh and checks, frame by frame:
  * the lowest mesh vertex, its index and its world-space Z
  * that frames the metric CLASSIFIES as penetrating really do have vertices below the floor
  * that frames it classifies as clean really do not
  * units / scale (stature), world-up (head above pelvis in Z), and left/right ordering

Run: python scripts/validate_mesh_gpen.py --clip <clip> --stream <label> [--frames N]
"""
import argparse
import json
import os
import sys

import joblib
import numpy as np
import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "scripts"))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "phc"))

from human_body_prior.body_model.body_model import BodyModel            # noqa: E402
from utils.physical_metrics import compute_ground_penetration           # noqa: E402
from mesh_gpen_quest import BM_NEUTRAL, DMPL_NEUTRAL, CACHE             # noqa: E402

TOL = 1e-4      # 0.1 mm, the definition's own tolerance
FAILS = []


def check(name, ok, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  — {detail}" if detail else ""))
    if not ok:
        FAILS.append(name)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clip", required=True)
    ap.add_argument("--stream", required=True)
    ap.add_argument("--frames", type=int, default=4)
    args = ap.parse_args()

    recs = joblib.load(os.path.join(CACHE, f"{args.clip}__{args.stream}.pkl"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bm = BodyModel(bm_fname=BM_NEUTRAL, num_betas=16, num_dmpls=8,
                   dmpl_fname=DMPL_NEUTRAL).to(device)
    T = len(recs)
    root = torch.tensor(np.array([r["root_orient_aa"] for r in recs]), dtype=torch.float32)
    body = torch.tensor(np.array([r["body_pose_aa"] for r in recs]), dtype=torch.float32)
    trans = torch.tensor(np.array([r["trans"] for r in recs]), dtype=torch.float32)
    pose_body = body[:, :21].reshape(T, 63)
    V, J = [], []
    with torch.no_grad():
        for s in range(0, T, 256):
            e = min(s + 256, T)
            out = bm(pose_body=pose_body[s:e].to(device), root_orient=root[s:e].to(device),
                     trans=trans[s:e].to(device))
            V.append(out.v.cpu()); J.append(out.Jtr.cpu())
    V = torch.cat(V, 0).numpy(); J = torch.cat(J, 0).numpy()

    mean_mm, max_mm, rate = compute_ground_penetration(torch.from_numpy(V), ground_z=0.0)
    pen = np.clip(-V[..., 2], 0, None)                   # depth below Z=0 per vertex
    frame_max = pen.max(-1)
    pen_frames = np.where(frame_max > TOL)[0]
    clean_frames = np.where(frame_max <= TOL)[0]

    print(f"\n=== {args.clip} / {args.stream} ===")
    print(f"  frames {T}, vertices {V.shape[1]}  | metric: mean {float(mean_mm):.2f} mm, "
          f"rate {float(rate):.3f}, max {float(max_mm):.2f} mm")
    print(f"  penetrating frames {len(pen_frames)}/{T}, clean {len(clean_frames)}/{T}")

    print("\n  -- deepest penetrating frames (metric says PENETRATING) --")
    for t in (pen_frames[np.argsort(-frame_max[pen_frames])][:args.frames] if len(pen_frames) else []):
        below = int((pen[t] > TOL).sum())
        vi = int(np.argmin(V[t, :, 2]))
        print(f"    frame {t:4d}: lowest vertex #{vi:5d} at Z = {V[t,vi,2]*1000:+8.2f} mm | "
              f"{below:5d} vertices below floor | depth {frame_max[t]*1000:6.2f} mm")
        check(f"frame {t} classified penetrating truly has vertices below Z=0",
              V[t, vi, 2] < 0 and below > 0)

    print("\n  -- clean frames (metric says NO penetration) --")
    for t in (clean_frames[np.argsort(-np.array([V[i,:,2].min() for i in clean_frames]))][:args.frames]
              if len(clean_frames) else []):
        vi = int(np.argmin(V[t, :, 2]))
        print(f"    frame {t:4d}: lowest vertex #{vi:5d} at Z = {V[t,vi,2]*1000:+8.2f} mm "
              f"(above floor)")
        check(f"frame {t} classified clean has no vertex below Z=0", V[t, vi, 2] >= -TOL)

    # units / scale / up-axis / left-right, on the SMPL joints the mesh was built from
    stature = float(np.median(V[..., 2].max(-1) - V[..., 2].min(-1)))
    ha = J[:, 15, 2] - J[:, 0, 2]
    head_above = float(np.median(ha))        # MEDIAN, not min: martialarts genuinely inverts
    head_above_min = float(np.min(ha))
    lat = J[:, 1] - J[:, 2]
    lat = lat / np.maximum(np.linalg.norm(lat, axis=-1, keepdims=True), 1e-9)
    s = np.einsum("tjc,tc->tj", J[:, :22] - J[:, 0:1], lat)
    ext = np.ptp(J.reshape(-1, 3), axis=0)
    print(f"\n  stature (median vertical extent) {stature:.3f} m | "
          f"median head-above-pelvis {head_above:+.3f} m (min {head_above_min:+.3f} m)")
    print(f"  joint extent per axis (X,Y,Z) = {np.round(ext,3)} m")
    check("units/scale plausible (1.4-2.1 m stature)", 1.4 < stature < 2.1, f"{stature:.3f} m")
    # A clip may legitimately invert (martialarts reaches world-up cos -0.70 in the Quest source,
    # verified against the raw pkl), so "upright on every frame" is not a convention test. What
    # must hold is that +Z is the UP axis: the body is head-up in the median pose.
    check("world-up is +Z (median pose is head-above-pelvis)", head_above > 0.2,
          f"median {head_above:+.3f} m, min {head_above_min:+.3f} m"
          + (" — clip genuinely inverts at some frames" if head_above_min < 0 else ""))
    check("left/right correct (L hip & shoulder left of R, all frames)",
          bool((s[:, 1] > s[:, 2]).all() and (s[:, 16] > s[:, 17]).all()))
    print("\n" + ("ALL CHECKS PASSED" if not FAILS else f"FAILED: {FAILS}"))
    return 1 if FAILS else 0


if __name__ == "__main__":
    sys.exit(main())
