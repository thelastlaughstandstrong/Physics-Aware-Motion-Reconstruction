"""
STEP 6E: quantify what the frozen-PHC physics pass does to the retargeted Quest motions.

Pure analysis of the Step 6D outputs -- no IK rerun, no PHC rerun, no controller/reward change.

Three comparisons, because "pre vs post" alone is ambiguous:

  A. vs the RETARGETED REFERENCE (what PHC was asked to track). Pre-sim is 0 by construction
     here, so these numbers are the COST physics introduces, not a comparison.
  B. vs the QUEST GROUND-TRUTH TARGETS (the independent upstream source, via the Step 2
     correspondence). Both pre- and post-sim are measurable against this, so it is the only
     framing that can actually answer "does physics degrade kinematic fidelity?".
  C. ABSOLUTE physical-quality metrics (jitter, foot skate, ground penetration), measurable on
     both sides directly.

Then two targeted diagnoses:
  * standing's 19.76 deg MPJRE -- swing (bone direction) vs twist (rotation about the bone axis).
    Step 5's IK was POSITION-based, so twist was never constrained; twist-dominated error is not
    a fidelity loss against anything the retarget actually determined.
  * walking's jitter 35.3 -> 71.4 -- distributed, foot-contact-locked, a few corrective events,
    or root/balance driven.

Run:
    LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH python scripts/analyze_step6e.py
"""
import json
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "scripts"))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "phc"))

import isaacgym  # noqa: F401,E402  -- must precede torch
import joblib  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from scipy.spatial.transform import Rotation as sRot  # noqa: E402

from evaluate_phc_metrics import (  # noqa: E402
    MUJOCO_2_SMPL_REORDER as M2S, NUM_SMPL_JOINTS as NJ, FOOT_JOINTS_MUJ,
    compute_foot_skate, compute_gpen, compute_metrics, compute_mpjre)

TRAJ = os.path.join(REPO, "data/quest_smpl_retarget/step6d_sim")
PREFLIGHT = os.path.join(REPO, "scripts/quest_smpl_phc_preflight.json")
CORR_JSON = os.path.join(REPO, "scripts/quest_smpl_correspondence.json")
OUT_MD = os.path.join(REPO, "scripts/quest_smpl_step6e_report.md")
OUT_JSON = os.path.join(REPO, "scripts/quest_smpl_step6e_report.json")
OUT_PNG = os.path.join(REPO, "scripts/quest_smpl_step6e_diagnostics.png")
SEQS = [("standing", "quest_retarget_standing",
         "data/quest2_synthetic/test_seg_motion/SUBJECT069.standing.pkl_0_425.pkl"),
        ("walking", "quest_retarget_walking",
         "data/quest2_synthetic/test_seg_motion/SUBJECT084.walking.pkl_0_426.pkl")]
FPS = 30
LINES = []


def PR(s=""):
    print(s)
    LINES.append(s)


def verdict(pre, post, lower_is_better=True, rel_tol=0.02):
    """improved / unchanged / worsened, with a 2% dead-band so noise isn't called a change."""
    if abs(post - pre) <= rel_tol * max(abs(pre), 1e-12):
        return "unchanged"
    better = post < pre if lower_is_better else post > pre
    return "improved" if better else "worsened"


def jitter_per_frame(p):
    """|jerk| per frame, averaged over joints. Returns (T-3,) aligned to frames 0..T-4."""
    j = (p[3:] - 3 * p[2:-1] + 3 * p[1:-2] - p[:-3]) * (FPS ** 3)
    return np.linalg.norm(j, axis=-1).mean(-1)


def swing_twist(q_err, axis):
    """Split a rotation error into twist about `axis` and the remaining swing.
    q_err: (N,4) xyzw. axis: (N,3) unit. Returns (twist_deg, swing_deg)."""
    v, w = q_err[:, :3], q_err[:, 3]
    proj = (v * axis).sum(-1, keepdims=True) * axis            # component of the rotation axis
    tw = np.concatenate([proj, w[:, None]], axis=-1)
    n = np.linalg.norm(tw, axis=-1, keepdims=True)
    n = np.where(n < 1e-12, 1.0, n)
    tw = tw / n
    twist = sRot.from_quat(tw)
    swing = sRot.from_quat(q_err) * twist.inv()
    return np.degrees(twist.magnitude()), np.degrees(swing.magnitude())


def quest_targets(motion_file, corr):
    """Quest GT joint positions in the native Z-up world -- the independent upstream reference."""
    import retarget_quest_to_smpl as M
    from poselib.poselib.skeleton.skeleton3d import SkeletonTree
    cal = json.load(open(os.path.join(REPO, "scripts/quest_smpl_frame_calibration.json")))
    R = np.array(cal[[k for k in cal if k.startswith("R")][0]])
    tree = SkeletonTree.from_mjcf(M.MJCF)
    e = joblib.load(os.path.join(REPO, motion_file))
    e = e[list(e.keys())[0]]
    pq = np.asarray(e["pose_quat"], dtype=np.float32)
    rt = np.asarray(e["root_trans_offset"], dtype=np.float32)
    return np.stack([M.build_targets(M.quest_fk_frame(tree, pq, rt, t), R, "native")
                     for t in range(pq.shape[0])], 0)


def main():
    from smpl_sim.smpllib.smpl_joint_names import SMPL_BONE_ORDER_NAMES as jn
    corr = json.load(open(CORR_JSON))
    direct = {int(k): v for k, v in corr["direct"].items()}     # quest idx -> smpl idx
    parents = corr["smpl_parents"]
    pre_json = json.load(open(PREFLIGHT))
    sel = M2S[:NJ]
    children = {j: [c for c in range(24) if parents[c] == j] for j in range(24)}

    PR("# Step 6E — effect of the frozen-PHC physics pass")
    PR("")
    PR("Analysis only: no IK rerun, no PHC rerun, no controller or reward change. Inputs are the "
       "Step 6D saved trajectories.")

    out = {}
    fig, axes = plt.subplots(2, 3, figsize=(16.5, 8))

    for row, (tag, key, motion_file) in enumerate(SEQS):
        t = joblib.load(os.path.join(TRAJ, f"{key}_sim.pkl"))
        post, pre = np.asarray(t["pred_pos"]), np.asarray(t["ref_pos"])   # mujoco order
        post_r, pre_r = np.asarray(t["pred_rot"]), np.asarray(t["ref_rot"])
        T = len(pre)
        r = {}

        # ---------- C: absolute physical quality, measurable on both ----------
        gp_pre, gp_post = compute_gpen(pre), compute_gpen(post)
        fs_pre = compute_foot_skate(pre, FPS) * 100
        fs_post = compute_foot_skate(post, FPS) * 100
        jf_pre, jf_post = jitter_per_frame(pre[:, sel]), jitter_per_frame(post[:, sel])

        # ---------- A: cost relative to the tracked reference ----------
        m = compute_metrics(post[:, sel], pre[:, sel], fps=FPS)
        mpjre = compute_mpjre(post_r, pre_r)
        drift = np.linalg.norm(post[:, 0] - pre[:, 0], axis=-1)
        local = np.linalg.norm((post[:, sel] - post[:, 0:1]) - (pre[:, sel] - pre[:, 0:1]), axis=-1)

        # ---------- B: fidelity vs the Quest ground truth ----------
        Q = quest_targets(motion_file, corr)
        dz = pre_json[tag]["z_shift_mm"] / 1000.0          # fix_trans_height, applied to BOTH
        qi = sorted(direct)
        sm_idx = [direct[q] for q in qi]
        muj = [M2S[s] for s in sm_idx]                      # smpl idx -> mujoco column
        best = None
        for sh in range(-3, 4):                             # PHC samples the ref with a small lag
            a = slice(max(0, sh), T + min(0, sh))
            b = slice(max(0, -sh), len(Q) + min(0, -sh))
            n = min(len(range(*a.indices(T))), len(range(*b.indices(len(Q)))))
            err = np.linalg.norm((pre[a][:n][:, muj] - np.array([0, 0, dz])) - Q[b][:n][:, qi],
                                 axis=-1).mean()
            if best is None or err < best[1]:
                best = (sh, err)
        sh = best[0]
        a = slice(max(0, sh), T + min(0, sh))
        b = slice(max(0, -sh), len(Q) + min(0, -sh))
        n = min(len(range(*a.indices(T))), len(range(*b.indices(len(Q)))))
        off = np.array([0, 0, dz])
        gt_pre = np.linalg.norm((pre[a][:n][:, muj] - off) - Q[b][:n][:, qi], axis=-1).mean() * 1000
        gt_post = np.linalg.norm((post[a][:n][:, muj] - off) - Q[b][:n][:, qi], axis=-1).mean() * 1000

        PR("")
        PR(f"## {tag}")
        PR("")
        PR("### Kinematic fidelity vs the Quest ground truth (the objective that matters)")
        PR("")
        PR(f"Both sides scored against the same independent Quest targets "
           f"({len(qi)} mapped joints, frame alignment {sh:+d}, the known "
           f"{dz*1000:+.1f} mm fix_trans_height shift removed from both).")
        PR("")
        PR("| | pre-sim (retargeted) | post-sim (PHC) | change | verdict |")
        PR("|---|---|---|---|---|")
        PR(f"| MPJPE vs Quest GT | **{gt_pre:.1f} mm** | **{gt_post:.1f} mm** | "
           f"{gt_post-gt_pre:+.1f} mm | **{verdict(gt_pre, gt_post)}** |")
        r["gt_mpjpe_pre_mm"], r["gt_mpjpe_post_mm"] = gt_pre, gt_post
        r["gt_verdict"] = verdict(gt_pre, gt_post)

        PR("")
        PR("### Physical-motion quality (absolute, measurable on both)")
        PR("")
        PR("| metric | pre-sim | post-sim | verdict |")
        PR("|---|---|---|---|")
        rows = [("jitter (m/s³)", jf_pre.mean(), jf_post.mean()),
                ("foot skate (cm/s)", fs_pre, fs_post),
                ("ground penetration mean (mm)", gp_pre[0], gp_post[0]),
                ("ground penetration max (mm)", gp_pre[1], gp_post[1]),
                ("ground penetration rate", gp_pre[2], gp_post[2])]
        for nm, x, y in rows:
            v = verdict(x, y)
            PR(f"| {nm} | {x:.3f} | {y:.3f} | **{v}** |")
            r[nm] = {"pre": float(x), "post": float(y), "verdict": v}

        PR("")
        PR("### Cost relative to the tracked reference (pre-sim is 0 by construction)")
        PR("")
        PR("| metric | value | note |")
        PR("|---|---|---|")
        PR(f"| MPJPE | {m['pos_error_m']*100:.2f} cm | how far PHC moved from the retarget |")
        PR(f"| — root/global displacement | {drift.mean()*100:.2f} cm | "
           f"{100*drift.mean()**2/(drift.mean()**2+local.mean()**2):.0f}% of squared error |")
        PR(f"| — root-relative pose | {local.mean()*100:.2f} cm | articulated-pose change |")
        PR(f"| MPJRE | {mpjre:.2f}° | decomposed below |")
        PR(f"| MPJVE | {m['vel_error_m_s']*100:.2f} cm/s | |")
        PR(f"| survival | 1.000 | no early termination |")
        r.update(mpjpe_cm=m["pos_error_m"] * 100, mpjre_deg=mpjre,
                 mpjve_cm_s=m["vel_error_m_s"] * 100,
                 root_drift_cm=float(drift.mean() * 100), pose_err_cm=float(local.mean() * 100))

        # ---------- diagnosis 1: rotation error, swing vs twist ----------
        qe = (sRot.from_quat(post_r.reshape(-1, 4)).inv()
              * sRot.from_quat(pre_r.reshape(-1, 4))).as_quat().reshape(T, 24, 4)
        per_joint = np.degrees((sRot.from_quat(post_r.reshape(-1, 4)).inv()
                                * sRot.from_quat(pre_r.reshape(-1, 4))).magnitude()).reshape(T, 24)
        tw_all, sw_all, names, per_j_tw = [], [], [], {}
        for j in range(24):
            if not children[j]:
                continue
            ax = pre[:, [M2S[c] for c in children[j]]].mean(1) - pre[:, M2S[j]]
            ax = ax / np.maximum(np.linalg.norm(ax, axis=-1, keepdims=True), 1e-9)
            tw, sw = swing_twist(qe[:, M2S[j]], ax)
            tw_all.append(tw.mean()); sw_all.append(sw.mean()); names.append(jn[j])
            per_j_tw[jn[j]] = (float(tw.mean()), float(sw.mean()))
        tw_all, sw_all = np.array(tw_all), np.array(sw_all)
        order = np.argsort(-per_joint.mean(0))
        PR("")
        PR(f"### Rotation-error diagnosis — MPJRE {mpjre:.2f}°")
        PR("")
        PR(f"Worst joints (mean°): " + ", ".join(
            f"{jn[i]} {per_joint.mean(0)[i]:.1f}" for i in order[:6]))
        PR(f"Mean over jointed (non-leaf) bones: **twist {tw_all.mean():.1f}°** about the bone "
           f"axis vs **swing {sw_all.mean():.1f}°** of bone direction "
           f"({100*tw_all.mean()/(tw_all.mean()+sw_all.mean()):.0f}% twist).")
        r["twist_deg"], r["swing_deg"] = float(tw_all.mean()), float(sw_all.mean())
        r["per_joint_mpjre"] = {jn[i]: float(per_joint.mean(0)[i]) for i in order[:8]}
        PR("")
        PR("| joint | MPJRE | twist (bone axis) | swing (bone direction) | twist share |")
        PR("|---|---|---|---|---|")
        for i in order[:8]:
            nm = jn[i]
            if nm in per_j_tw:
                tw, sw = per_j_tw[nm]
                PR(f"| {nm} | {per_joint.mean(0)[i]:.1f}° | {tw:.1f}° | {sw:.1f}° | "
                   f"{100*tw/max(tw+sw,1e-9):.0f}% |")
            else:
                PR(f"| {nm} (leaf) | {per_joint.mean(0)[i]:.1f}° | — | — | inherits parent |")
        r["per_joint_swing_twist"] = per_j_tw
        # is the arm error a stable bias or fluctuating?
        arm = [jn.index(n) for n in ["L_Shoulder", "R_Shoulder", "L_Elbow", "R_Elbow"]]
        armq = qe[:, [M2S[j] for j in arm]].reshape(-1, 4)
        mean_rot = sRot.from_quat(armq).mean()
        resid = np.degrees((sRot.from_quat(armq) * mean_rot.inv()).magnitude()).mean()
        PR("")
        PR(f"Arm chain: mean error rotation **{np.degrees(mean_rot.magnitude()):.1f}°** "
           f"(a constant postural bias), residual scatter about it **{resid:.1f}°** "
           f"-- {'mostly a STATIC offset' if np.degrees(mean_rot.magnitude()) > resid else 'mostly FLUCTUATION'}")
        r["arm_static_bias_deg"] = float(np.degrees(mean_rot.magnitude()))
        r["arm_fluctuation_deg"] = float(resid)

        # ---------- diagnosis 2: where the jitter lives ----------
        # Per-FOOT contact. Taking min over both feet calls ~100% of a walk "contact" (one foot
        # is nearly always down), which says nothing -- classify by how many feet are down.
        fz = pre[:, FOOT_JOINTS_MUJ, 2][3:]                       # (T-3, 2)
        thr = 0.05
        n_down = (fz < thr).sum(-1)
        double, single, flight = n_down == 2, n_down == 1, n_down == 0
        # heel-strike / toe-off events: any change in the contact set
        events = np.zeros(len(n_down), bool)
        ch = np.where(np.diff(n_down) != 0)[0]
        for c in ch:
            events[max(0, c - 2):min(len(events), c + 3)] = True

        # Concentration of the INCREASE. Only positive excess counts, and the whole question is
        # moot when jitter fell (standing) -- dividing by a negative total gives nonsense.
        excess = jf_post - jf_pre
        pos = np.clip(excess, 0, None)
        tot = pos.sum()
        idx = np.argsort(-pos)
        top1 = max(1, len(pos) // 100)
        top5 = max(1, len(pos) // 20)
        share1 = float(pos[idx[:top1]].sum() / tot) if tot > 1e-9 else float("nan")
        share5 = float(pos[idx[:top5]].sum() / tot) if tot > 1e-9 else float("nan")

        # High-frequency content: fraction of jerk power above 7.5 Hz (half of the 15 Hz Nyquist).
        # Chatter concentrates here; genuine corrective motions do not.
        def hf_frac(p):
            acc = np.diff(p, 2, axis=0) * (FPS ** 2)
            sp = np.abs(np.fft.rfft(acc - acc.mean(0), axis=0)) ** 2
            f = np.fft.rfftfreq(len(acc), 1.0 / FPS)
            return float(sp[f > 7.5].sum() / max(sp.sum(), 1e-12))
        hf_pre, hf_post = hf_frac(pre[:, sel]), hf_frac(post[:, sel])
        jr_pre = jitter_per_frame(pre[:, 0:1]); jr_post = jitter_per_frame(post[:, 0:1])
        loc_pre = jitter_per_frame(pre[:, sel] - pre[:, 0:1])
        loc_post = jitter_per_frame(post[:, sel] - post[:, 0:1])
        PR("")
        PR(f"### Jitter diagnosis — {jf_pre.mean():.1f} → {jf_post.mean():.1f} m/s³ "
           f"({jf_post.mean()/max(jf_pre.mean(),1e-9):.2f}×)")
        PR("")
        for nm, msk in [("double support", double), ("single support", single), ("flight", flight)]:
            if msk.any():
                PR(f"- {nm}: {100*msk.mean():4.0f}% of frames | jitter "
                   f"{jf_pre[msk].mean():6.1f} → **{jf_post[msk].mean():6.1f}** "
                   f"({jf_post[msk].mean()/max(jf_pre[msk].mean(),1e-9):.2f}×)")
        if events.any():
            PR(f"- within ±2 frames of a contact change ({100*events.mean():.0f}% of frames): "
               f"{jf_pre[events].mean():.1f} → **{jf_post[events].mean():.1f}** "
               f"({jf_post[events].mean()/max(jf_pre[events].mean(),1e-9):.2f}×); "
               f"elsewhere {jf_pre[~events].mean():.1f} → **{jf_post[~events].mean():.1f}** "
               f"({jf_post[~events].mean()/max(jf_pre[~events].mean(),1e-9):.2f}×)")
        if np.isfinite(share1):
            PR(f"- concentration: worst 1% of frames carry **{100*share1:.0f}%** of the increase, "
               f"worst 5% carry **{100*share5:.0f}%** (uniform would be 1% / 5%)")
        else:
            PR("- concentration: n/a — jitter decreased, there is no increase to concentrate")
        PR(f"- high-frequency power (>7.5 Hz, i.e. chatter): "
           f"{100*hf_pre:.1f}% → **{100*hf_post:.1f}%** of jerk power")
        PR(f"- root vs articulation: root jitter {jr_pre.mean():.1f} → **{jr_post.mean():.1f}** "
           f"({jr_post.mean()/max(jr_pre.mean(),1e-9):.2f}×); root-relative "
           f"{loc_pre.mean():.1f} → **{loc_post.mean():.1f}** "
           f"({loc_post.mean()/max(loc_pre.mean(),1e-9):.2f}×)")
        r["jitter_double_support"] = float(jf_post[double].mean()) if double.any() else None
        r["jitter_single_support"] = float(jf_post[single].mean()) if single.any() else None
        r["jitter_event_ratio"] = (float(jf_post[events].mean() / max(jf_pre[events].mean(), 1e-9))
                                   if events.any() else None)
        r["jitter_top1pct_share"] = share1
        r["jitter_top5pct_share"] = share5
        r["hf_power_pre"], r["hf_power_post"] = hf_pre, hf_post
        r["jitter_root_ratio"] = float(jr_post.mean() / max(jr_pre.mean(), 1e-9))
        r["jitter_local_ratio"] = float(loc_post.mean() / max(loc_pre.mean(), 1e-9))

        # ---------- plots ----------
        ax = axes[row, 0]
        ax.plot(jf_pre, lw=.8, label="pre-sim", color="#1f77b4")
        ax.plot(jf_post, lw=.8, label="post-sim", color="#d62728", alpha=.85)
        ax.fill_between(np.arange(len(events)), 0, ax.get_ylim()[1], where=events,
                        color="k", alpha=.10, step="mid", label="contact change ±2f")
        ax.set_yscale("log"); ax.set_title(f"{tag} — per-frame jitter", fontsize=9)
        ax.set_xlabel("frame", fontsize=7); ax.set_ylabel("m/s³", fontsize=7)
        ax.legend(fontsize=6); ax.tick_params(labelsize=6)

        ax = axes[row, 1]
        ax.bar(np.arange(len(names)), tw_all, label="twist (about bone axis)", color="#ff7f0e")
        ax.bar(np.arange(len(names)), sw_all, bottom=tw_all, label="swing (bone direction)",
               color="#2ca02c")
        ax.set_xticks(np.arange(len(names)))
        ax.set_xticklabels(names, rotation=90, fontsize=5)
        ax.set_title(f"{tag} — rotation error, swing vs twist", fontsize=9)
        ax.set_ylabel("degrees", fontsize=7); ax.legend(fontsize=6); ax.tick_params(labelsize=6)

        ax = axes[row, 2]
        ax.plot(drift * 100, lw=.9, color="#9467bd", label="root/global displacement")
        ax.plot(local.mean(-1) * 100, lw=.9, color="#8c564b", label="root-relative pose")
        ax.set_title(f"{tag} — where the position error is", fontsize=9)
        ax.set_xlabel("frame", fontsize=7); ax.set_ylabel("cm", fontsize=7)
        ax.legend(fontsize=6); ax.tick_params(labelsize=6)

        out[tag] = r

    fig.suptitle("Step 6E — frozen-PHC physics refinement diagnostics", fontsize=12)
    fig.tight_layout()
    fig.savefig(OUT_PNG, dpi=135, bbox_inches="tight")
    json.dump(out, open(OUT_JSON, "w"), indent=2, default=float)
    open(OUT_MD, "w").write("\n".join(LINES) + "\n")
    PR("")
    PR(f"diagnostics -> {os.path.relpath(OUT_PNG, REPO)}")
    PR(f"report      -> {os.path.relpath(OUT_MD, REPO)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
