"""
STEP 5 of the Quest->SMPL retargeting work: extend the validated single-frame IK to a full motion
sequence using PREVIOUS-FRAME WARM STARTING ONLY.

Deliberately NO temporal smoothness loss. The question this step answers is whether warm-starting
alone already yields temporally stable motion -- adding a smoothing term first would mask that.

    Quest sequence -> native Z-up targets -> frame 0 IK (cold)
                                          -> frame t IK warm-started from t-1
                                          -> SMPL {trans, root_orient, body_pose} per frame

Every numerical primitive is imported from scripts/retarget_quest_to_smpl.py -- the FK, the
Rodrigues sinc implementation, the correspondence/calibration loaders and, critically, the fit
itself. There is exactly ONE optimizer implementation in the project; this file only supplies the
sequential control flow and the temporal diagnostics.

Usage:
    python scripts/retarget_quest_to_smpl_sequence.py --motion_file <clip.pkl>
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "scripts"))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "phc"))

import joblib                                            # noqa: E402
import matplotlib                                        # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt                          # noqa: E402
from scipy.spatial.transform import Rotation as Rot      # noqa: E402
from poselib.poselib.skeleton.skeleton3d import SkeletonTree  # noqa: E402

import retarget_quest_to_smpl as M                       # noqa: E402

FPS = 30.0


def relative_angle_deg(aa_a, aa_b):
    """Per-joint geodesic angle between two axis-angle poses [J,3] -> [J] degrees."""
    Ra = Rot.from_rotvec(aa_a)
    Rb = Rot.from_rotvec(aa_b)
    return np.degrees((Ra.inv() * Rb).magnitude())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--motion_file", required=True)
    ap.add_argument("--max-frames", type=int, default=0, help="0 = whole clip")
    ap.add_argument("--iters-first", type=int, default=3000, help="cold start, frame 0")
    ap.add_argument("--iters-warm", type=int, default=600, help="cap for warm-started frames")
    ap.add_argument("--tol", type=float, default=1e-6, help="early-stop relative tolerance")
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--w-group-a", type=float, default=1.0)
    ap.add_argument("--w-group-b", type=float, default=0.3)
    ap.add_argument("--w-reg", type=float, default=1e-3)
    ap.add_argument("--w-root", type=float, default=0.25)
    ap.add_argument("--cold-every", type=int, default=25,
                    help="Additionally solve every Nth frame COLD (independent, Step-4 style) as "
                         "a drift control: warm-started results are compared against these.")
    ap.add_argument("--out-prefix", default=os.path.join(REPO, "scripts/quest_smpl_sequence_fit"))
    args = ap.parse_args()
    args.arm_init = False
    args.iters = args.iters_first
    dev = torch.device("cpu")

    corr = json.load(open(M.CORR_JSON))
    cal = json.load(open(M.CAL_JSON))
    R = np.array(cal["R_quest_to_smpl"])
    direct = {int(k): v for k, v in corr["direct"].items()}
    groups = corr["validation_groups"]
    qn, smpl_par = corr["quest_joint_names"], corr["smpl_parents"]
    soff = M.load_smpl_rest_offsets(corr)

    tree = SkeletonTree.from_mjcf(M.MJCF)
    qp = tree.parent_indices.tolist()
    d = joblib.load(args.motion_file)
    e = d[list(d.keys())[0]]
    pq = np.asarray(e["pose_quat"], dtype=np.float32)
    rt = np.asarray(e["root_trans_offset"], dtype=np.float32)
    T = pq.shape[0] if args.max_frames <= 0 else min(args.max_frames, pq.shape[0])

    # ---- targets: native Z-up, no anatomical R on world coords (Step 4b decision) ----
    Qt = np.stack([M.build_targets(M.quest_fk_frame(tree, pq, rt, t), R, "native")
                   for t in range(T)], axis=0)                       # [T,25,3]

    gA, gB = groups["high_confidence"], groups["structural_mismatch"]
    recs, prev, cold_ref = [], None, {}
    t_start = time.time()
    for t in range(T):
        warm = prev is not None
        r = M.fit_single_frame(
            Qt[t], soff, smpl_par, direct, groups, qn, args, dev,
            init=prev,
            max_iters=(args.iters_warm if warm else args.iters_first),
            tol=args.tol)
        prev = {"root_orient": r["root_orient_aa"], "body_pose": r["body_pose_aa"],
                "trans": r["trans"]}
        recs.append(r)

        if args.cold_every and t % args.cold_every == 0:
            c = M.fit_single_frame(Qt[t], soff, smpl_par, direct, groups, qn, args, dev,
                                   init=None, max_iters=args.iters_first, tol=args.tol)
            cold_ref[t] = c
        if (t + 1) % 50 == 0:
            print(f"  frame {t+1}/{T}  all={r['all_mean']:.1f}mm  iters={r['iters_used']}")
    wall = time.time() - t_start

    # ---- aggregate ----
    A = np.array([r["group_a_mean"] for r in recs])
    Amax = np.array([r["group_a_max"] for r in recs])
    B = np.array([r["group_b_mean"] for r in recs])
    Bmax = np.array([r["group_b_max"] for r in recs])
    ALL = np.array([r["all_mean"] for r in recs])
    ALLmax = np.array([r["all_max"] for r in recs])
    root = np.array([r["root_trans_err_mm"] for r in recs])
    iters = np.array([r["iters_used"] for r in recs])
    maxang = np.array([r["max_joint_angle_deg"] for r in recs])
    upc = np.array([r["upright_cos"] for r in recs])
    latc = np.array([r["lateral_cos"] for r in recs])
    det = np.array([r["handedness_det"] for r in recs])
    wup = np.array([r["world_up_cos"] for r in recs])

    pose = np.stack([np.asarray(r["body_pose_aa"]) for r in recs])    # [T,23,3]
    rootaa = np.stack([np.asarray(r["root_orient_aa"]) for r in recs])
    P = np.stack([r["positions"] for r in recs])                      # [T,24,3]

    # frame-to-frame rotation change (deg/frame)
    d_joint = np.stack([relative_angle_deg(pose[t - 1], pose[t]) for t in range(1, T)])  # [T-1,23]
    d_root = np.array([relative_angle_deg(rootaa[t - 1][None], rootaa[t][None])[0]
                       for t in range(1, T)])

    def kin(X):
        v = np.linalg.norm(np.diff(X, axis=0), axis=-1) * FPS                  # m/s
        a = np.linalg.norm(np.diff(X, n=2, axis=0), axis=-1) * FPS ** 2        # m/s^2
        j = np.linalg.norm(X[3:] - 3 * X[2:-1] + 3 * X[1:-2] - X[:-3], axis=-1) * FPS ** 3
        return v, a, j

    v_f, a_f, j_f = kin(P)
    v_t, a_t, j_t = kin(Qt[:, [direct[q] if False else q for q in range(25)]][:, :, :]) \
        if False else kin(Qt)

    # discontinuity detection: frames whose joint rotation jump is a gross outlier
    dj_max = d_joint.max(axis=1)
    thr = np.median(dj_max) + 6 * (np.percentile(dj_max, 75) - np.percentile(dj_max, 25) + 1e-9)
    flips = np.where(dj_max > max(thr, 20.0))[0] + 1

    # warm vs cold drift control
    drift = {t: float(abs(recs[t]["all_mean"] - c["all_mean"])) for t, c in cold_ref.items()}
    cold_all = np.array([c["all_mean"] for c in cold_ref.values()])
    warm_at_cold = np.array([recs[t]["all_mean"] for t in cold_ref])

    out = []

    def PR(s=""):
        print(s)
        out.append(s)

    PR("# Quest → SMPL Full-Sequence Retarget (Step 5)")
    PR("")
    PR(f"- clip: `{os.path.basename(args.motion_file)}` — **{T} frames** @ {FPS:.0f} fps")
    PR(f"- world frame **native Z-up**, neutral SMPL, **scale fixed**, `lambda_root={args.w_root}`")
    PR(f"- warm start from frame t−1; **no temporal smoothness loss** (deliberately)")
    PR(f"- frame 0 cold ≤{args.iters_first} iters; warm frames ≤{args.iters_warm}, "
       f"early-stop tol {args.tol}")
    PR(f"- wall clock **{wall:.1f}s** ({1000*wall/T:.0f} ms/frame)")
    PR("")
    PR("## Spatial fit over the sequence")
    PR("")
    PR("| metric | mean | max | Step 4 single-frame reference |")
    PR("|---|---|---|---|")
    PR(f"| Group A (mm) | **{A.mean():.1f}** | {Amax.max():.1f} | 29.0 |")
    PR(f"| Group B (mm) | **{B.mean():.1f}** | {Bmax.max():.1f} | 45.8 |")
    PR(f"| all mapped (mm) | **{ALL.mean():.1f}** | {ALLmax.max():.1f} | 34.3 |")
    PR(f"| root translation (mm) | **{root.mean():.1f}** | {root.max():.1f} | 45.0 |")
    PR("")
    PR(f"Per-frame all-mapped residual: min {ALL.min():.1f}, median {np.median(ALL):.1f}, "
       f"max {ALL.max():.1f} mm (std {ALL.std():.1f}).")
    PR("")
    PR("## Warm-start effectiveness")
    PR("")
    PR(f"| | frame 0 (cold) | warm frames |")
    PR("|---|---|---|")
    PR(f"| iterations | {iters[0]} | mean **{iters[1:].mean():.0f}**, "
       f"median {np.median(iters[1:]):.0f}, max {iters[1:].max()} |")
    PR("")
    PR(f"Warm-started frames converge in **{iters[1:].mean():.0f}** iterations vs "
       f"**{iters[0]}** for the cold start — a **{iters[0]/max(iters[1:].mean(),1e-9):.1f}×** "
       f"reduction at the same tolerance.")
    PR("")
    PR(f"**Drift control** — every {args.cold_every}th frame was ALSO solved cold (independent, "
       f"Step-4 style) and compared against the warm-started result:")
    PR("")
    PR(f"- warm all-mapped at those frames: **{warm_at_cold.mean():.2f} mm**")
    PR(f"- cold all-mapped at those frames: **{cold_all.mean():.2f} mm**")
    PR(f"- max |warm − cold| over {len(drift)} checks: **{max(drift.values()):.3f} mm**")
    PR("")
    PR("A small max difference means warm-starting is not accumulating error down the sequence — "
       "it reaches the same optimum as an independent solve, just faster.")
    PR("")
    PR("## Temporal quality (no smoothing applied)")
    PR("")
    PR("| quantity | mean | p99 | max |")
    PR("|---|---|---|---|")
    PR(f"| joint rotation change (°/frame) | {d_joint.mean():.2f} | "
       f"{np.percentile(d_joint,99):.2f} | {d_joint.max():.2f} |")
    PR(f"| root rotation change (°/frame) | {d_root.mean():.2f} | "
       f"{np.percentile(d_root,99):.2f} | {d_root.max():.2f} |")
    PR(f"| joint velocity (m/s) | {v_f.mean():.3f} | {np.percentile(v_f,99):.3f} | {v_f.max():.3f} |")
    PR(f"| joint acceleration (m/s²) | {a_f.mean():.2f} | {np.percentile(a_f,99):.2f} | {a_f.max():.2f} |")
    PR(f"| jitter / jerk (m/s³) | {j_f.mean():.1f} | {np.percentile(j_f,99):.1f} | {j_f.max():.1f} |")
    PR("")
    PR("Same quantities on the **Quest target motion** (the source, i.e. the floor any faithful "
       "retarget should approach — not zero):")
    PR("")
    PR("| quantity | fitted SMPL | Quest target | ratio |")
    PR("|---|---|---|---|")
    PR(f"| velocity (m/s) | {v_f.mean():.3f} | {v_t.mean():.3f} | {v_f.mean()/max(v_t.mean(),1e-9):.2f}× |")
    PR(f"| acceleration (m/s²) | {a_f.mean():.2f} | {a_t.mean():.2f} | {a_f.mean()/max(a_t.mean(),1e-9):.2f}× |")
    PR(f"| jerk (m/s³) | {j_f.mean():.1f} | {j_t.mean():.1f} | {j_f.mean()/max(j_t.mean(),1e-9):.2f}× |")
    PR("")
    PR(f"**Discontinuities / rotational flips**: {len(flips)} frame(s) exceeded the outlier "
       f"threshold ({max(thr,20.0):.1f}°/frame)"
       + (f" at {flips.tolist()[:12]}" if len(flips) else " — **none detected**") + ".")
    PR("")
    PR("## Sanity over all frames")
    PR("")
    PR("| check | range | verdict |")
    PR("|---|---|---|")
    PR(f"| upright cos (vs target) | {upc.min():.4f} … {upc.max():.4f} | "
       f"{'PASS' if upc.min() > 0.98 else 'CHECK'} |")
    PR(f"| **world-up cos (absolute)** | {wup.min():+.4f} … {wup.max():+.4f} | "
       f"{'PASS — upright in native Z-up' if wup.min() > 0.98 else 'CHECK'} |")
    PR(f"| left/right cos | {latc.min():.4f} … {latc.max():.4f} | "
       f"{'PASS' if latc.min() > 0.9 else 'FAIL — mirrored'} |")
    PR(f"| handedness det | {det.min():+.3f} … {det.max():+.3f} | "
       f"{'PASS' if det.min() > 0 else 'FAIL — reflection'} |")
    PR(f"| max joint angle | {maxang.min():.1f}° … {maxang.max():.1f}° | "
       f"{'PASS' if maxang.max() < 150 else 'CHECK — extreme'} |")
    PR(f"| all finite | {all(r['finite'] for r in recs)} | "
       f"{'PASS' if all(r['finite'] for r in recs) else 'FAIL'} |")
    PR("")

    # ---- figures ----
    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    ax = axes[0, 0]
    ax.plot(A, label="Group A", lw=1.2); ax.plot(B, label="Group B", lw=1.2)
    ax.plot(ALL, label="all mapped", lw=1.0, c="k", alpha=.6)
    ax.plot(root, label="root", lw=1.0, ls="--", c="tab:purple")
    if cold_ref:
        ax.scatter(list(cold_ref), [c["all_mean"] for c in cold_ref.values()],
                   c="tab:red", s=18, zorder=5, label="cold-solve control")
    ax.axhline(34.3, c="0.5", ls=":", lw=1)
    ax.set_xlabel("frame"); ax.set_ylabel("mm"); ax.set_title("Spatial residual per frame")
    ax.legend(fontsize=8)
    ax = axes[0, 1]
    ax.plot(dj_max, lw=.9, label="max joint Δ")
    ax.plot(d_root, lw=.9, label="root Δ", alpha=.8)
    ax.set_xlabel("frame"); ax.set_ylabel("°/frame")
    ax.set_title("Frame-to-frame rotation change"); ax.legend(fontsize=8)
    ax = axes[1, 0]
    ax.plot(iters, lw=1.0)
    ax.set_xlabel("frame"); ax.set_ylabel("iterations")
    ax.set_title(f"Iterations to converge (cold frame 0 = {iters[0]})")
    ax = axes[1, 1]
    ax.plot(j_f.mean(axis=1), lw=.9, label="fitted SMPL")
    ax.plot(j_t.mean(axis=1), lw=.9, label="Quest target", alpha=.75)
    ax.set_xlabel("frame"); ax.set_ylabel("jerk (m/s³)")
    ax.set_title("Jitter vs source motion"); ax.legend(fontsize=8)
    fig.suptitle(f"Step 5 — {os.path.basename(args.motion_file)} ({T} frames), warm-start only, "
                 f"no temporal loss", fontsize=11)
    fig.tight_layout()
    fig.savefig(args.out_prefix + "_diagnostics.png", dpi=140)

    # pose strip in RAW native Z-up (no corrective rotation)
    idx = np.linspace(0, T - 1, 6).astype(int)
    fig2 = plt.figure(figsize=(17, 4.2))
    for k, t in enumerate(idx):
        ax = fig2.add_subplot(1, 6, k + 1, projection="3d")
        for i in range(1, len(qp)):
            ax.plot(*zip(Qt[t][i], Qt[t][qp[i]]), c="tab:red", lw=1.4, alpha=.85)
        for i in range(1, 24):
            ax.plot(*zip(P[t][i], P[t][smpl_par[i]]), c="tab:blue", lw=1.4, alpha=.85)
        c = Qt[t].mean(0); lim = 1.0
        ax.set_xlim(c[0]-lim, c[0]+lim); ax.set_ylim(c[1]-lim, c[1]+lim); ax.set_zlim(0, 2.0)
        ax.set_title(f"frame {t}", fontsize=9)
        ax.set_xlabel("X", fontsize=7); ax.set_ylabel("Y", fontsize=7); ax.set_zlabel("Z (up)", fontsize=7)
        ax.tick_params(labelsize=6)
        ax.view_init(elev=12, azim=-60)
    fig2.suptitle("Raw native Z-up world — no display-only corrective rotation "
                  "(red = Quest target, blue = retargeted SMPL)", fontsize=11)
    fig2.tight_layout()
    fig2.savefig(args.out_prefix + "_strip.png", dpi=140)

    PR("## Visualization")
    PR("")
    PR(f"![diagnostics]({os.path.basename(args.out_prefix)}_diagnostics.png)")
    PR("")
    PR(f"![strip]({os.path.basename(args.out_prefix)}_strip.png)")
    PR("")

    json.dump({
        "motion_file": args.motion_file, "frames": T, "fps": FPS,
        "world_frame": "native", "w_root": args.w_root, "temporal_loss": None,
        "wall_seconds": wall,
        "group_a_mean_mm": float(A.mean()), "group_a_max_mm": float(Amax.max()),
        "group_b_mean_mm": float(B.mean()), "group_b_max_mm": float(Bmax.max()),
        "all_mean_mm": float(ALL.mean()), "all_max_mm": float(ALLmax.max()),
        "root_mean_mm": float(root.mean()), "root_max_mm": float(root.max()),
        "iters_cold_frame0": int(iters[0]), "iters_warm_mean": float(iters[1:].mean()),
        "joint_rot_change_deg_mean": float(d_joint.mean()),
        "joint_rot_change_deg_max": float(d_joint.max()),
        "root_rot_change_deg_max": float(d_root.max()),
        "vel_mean_fit": float(v_f.mean()), "vel_mean_target": float(v_t.mean()),
        "acc_mean_fit": float(a_f.mean()), "acc_mean_target": float(a_t.mean()),
        "jerk_mean_fit": float(j_f.mean()), "jerk_mean_target": float(j_t.mean()),
        "n_discontinuities": int(len(flips)), "discontinuity_frames": flips.tolist(),
        "warm_vs_cold_max_diff_mm": float(max(drift.values())) if drift else None,
        "max_joint_angle_deg": float(maxang.max()),
        "world_up_cos_min": float(wup.min()),
        "smpl_trans": [r["trans"] for r in recs],
        "smpl_root_orient_aa": rootaa.tolist(),
        "smpl_body_pose_aa": pose.tolist(),
    }, open(args.out_prefix + ".json", "w"))
    with open(args.out_prefix + ".md", "w") as f:
        f.write("\n".join(out) + "\n")
    print(f"\n[json] {args.out_prefix}.json\n[md]   {args.out_prefix}.md")


if __name__ == "__main__":
    main()
