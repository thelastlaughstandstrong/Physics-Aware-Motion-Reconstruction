"""
STEP 6D report: metrics for the frozen-PHC rollout on the two Quest->SMPL motions.

Reads output/HumanoidIm/quest_smpl_frozen_phc/phc_positions_for_eval.pkl and scores PHC's
SIMULATED trajectory against the REFERENCE motion PHC itself loaded (both are in the dump), so no
external ground truth or key parsing is needed.

Metric definitions are imported from scripts/evaluate_phc_metrics.py -- the same functions and
constants that produced the SMPL naive baseline in results_comparison.md -- so these numbers are
directly comparable to that table.

Also decomposes MPJPE into root drift (whole body in the wrong place) vs root-relative pose error
(limbs in the wrong configuration), and saves each sequence's simulated trajectory separately for
later pre-vs-post comparison and Blender export.

Run:
    LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH python scripts/report_step6d.py
"""
import json
import os
import sys

import joblib
import numpy as np
from scipy.spatial.transform import Rotation as sRot

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "scripts"))
sys.path.insert(0, REPO)

from evaluate_phc_metrics import (  # noqa: E402  -- the project's own SMPL definitions
    FOOT_JOINTS_MUJ, MUJOCO_2_SMPL_REORDER, NUM_SMPL_JOINTS, compute_foot_skate,
    compute_gpen, compute_metrics, compute_mpjre)

EXP_DIR = os.path.join(REPO, "output/HumanoidIm/quest_smpl_frozen_phc")
DUMP = os.path.join(EXP_DIR, "phc_positions_for_eval.pkl")
TRAJ_DIR = os.path.join(REPO, "data/quest_smpl_retarget/step6d_sim")
OUT_MD = os.path.join(REPO, "scripts/quest_smpl_step6d_report.md")
OUT_JSON = os.path.join(REPO, "scripts/quest_smpl_step6d_report.json")
FPS = 30
LINES = []


def PR(s=""):
    print(s)
    LINES.append(s)


def main():
    if not os.path.isfile(DUMP):
        print(f"No rollout dump at {DUMP} — run scripts/run_step6d_frozen_phc.sh first.")
        return 2
    d = joblib.load(DUMP)
    keys = [str(k) for k in d["motion_keys"]]
    succ = set()
    p = os.path.join(EXP_DIR, "long_succ.pkl")
    if os.path.isfile(p):
        succ = {os.path.basename(str(x)) for x in joblib.load(p)}
    term = d.get("terminate_hist", [])
    os.makedirs(TRAJ_DIR, exist_ok=True)

    PR("# Step 6D — frozen PHC tracking the retargeted Quest motions")
    PR("")
    PR(f"- checkpoint: `output/HumanoidIm/phc_kp_pnn_iccv/Humanoid.pth` (frozen, not fine-tuned)")
    PR(f"- policy/config: `learning=im_mcp env=env_im_getup_mcp robot=smpl_humanoid "
       f"env.obs_v=7 test=True im_eval=True epoch=-1`")
    PR(f"- motions: `data/quest_smpl_retarget/quest_smpl_both_phc.pkl`")
    PR(f"- overall success_rate reported by PHC: **{d.get('success_rate', float('nan')):.3f}**")
    PR("")

    out = {}
    for i, key in enumerate(keys):
        pred = np.asarray(d["pred_pos_all"][i])            # [T,24,3] mujoco order
        ref = np.asarray(d["ref_pos_all"][i])
        pr = np.asarray(d["pred_rot_all"][i])              # [T,24,4] xyzw
        gr = np.asarray(d["gt_rot_all"][i])
        T_sim, T_ref = len(pred), len(ref)
        n = min(T_sim, T_ref)
        pred, ref, pr, gr = pred[:n], ref[:n], pr[:n], gr[:n]

        survived = os.path.basename(key) in succ
        # terminate_hist is a per-motion BOOLEAN (did this motion terminate early), not a frame
        # index -- reading it as a frame turns "False" into a spurious "failure at frame 0".
        terminated = bool(term[i]) if i < len(term) else (not survived)
        fail_frame = (T_sim - 1) if terminated else None

        # positional metrics on the 22 AvatarPoser joints, SMPL order (drops the hand leaves)
        sel = MUJOCO_2_SMPL_REORDER[:NUM_SMPL_JOINTS]
        m = compute_metrics(pred[:, sel], ref[:, sel], fps=FPS)
        mpjre = compute_mpjre(pr, gr)
        skate = compute_foot_skate(pred, fps=FPS)
        gp_mean, gp_max, gp_rate = compute_gpen(pred)
        root_drift = np.linalg.norm(pred[:, 0] - ref[:, 0], axis=-1)
        local = np.linalg.norm((pred[:, sel] - pred[:, 0:1]) - (ref[:, sel] - ref[:, 0:1]), axis=-1)

        PR("")
        PR(f"## {key}")
        PR("")
        PR(f"- reference frames **{T_ref}** (PHC drops the source's final frame: "
           f"{T_ref + 1} -> {T_ref}), simulated frames **{T_sim}**, "
           f"survival ratio **{T_sim/T_ref:.3f}**")
        PR(f"- survived: **{'YES' if survived else 'NO'}**, early termination: "
           f"**{'YES' if terminated else 'NO'}**"
           + (f", first failure at frame **{fail_frame}** "
              f"({fail_frame/FPS:.2f}s of {(T_ref-1)/FPS:.2f}s)" if fail_frame is not None
              else " — tracked to the end of the clip"))
        PR("")
        PR("| metric | value |")
        PR("|---|---|")
        PR(f"| MPJPE (22 joints) | **{m['pos_error_m']*100:.2f} cm** |")
        PR(f"| — root drift | {root_drift.mean()*100:.2f} cm |")
        PR(f"| — root-relative pose error | {local.mean()*100:.2f} cm |")
        PR(f"| MPJRE (global, 24 joints) | **{mpjre:.2f}°** |")
        PR(f"| MPJVE | {m['vel_error_m_s']*100:.2f} cm/s |")
        PR(f"| jitter | {m['jitter_m_s3']:.1f} m/s³ |")
        PR(f"| foot skate | {skate*100:.2f} cm/s |")
        PR(f"| ground penetration (mean / max / rate) | {gp_mean:.2f} mm / {gp_max:.2f} mm "
           f"/ {gp_rate:.3f} |")
        PR(f"| root trajectory error (mean / max) | {root_drift.mean()*100:.2f} cm / "
           f"{root_drift.max()*100:.2f} cm |")

        traj = os.path.join(TRAJ_DIR, f"{os.path.basename(key)}_sim.pkl")
        joblib.dump({"pred_pos": pred, "ref_pos": ref, "pred_rot": pr, "ref_rot": gr,
                     "fps": FPS, "key": key, "survived": survived,
                     "fail_frame": fail_frame, "terminated": terminated,
                     "joint_order": "mujoco"}, traj)
        PR("")
        PR(f"Simulated trajectory saved: `{os.path.relpath(traj, REPO)}`")

        out[key] = dict(ref_frames=T_ref, sim_frames=T_sim, survival_ratio=T_sim / T_ref,
                        survived=bool(survived), terminated=terminated, fail_frame=fail_frame,
                        mpjpe_cm=m["pos_error_m"] * 100, mpjre_deg=mpjre,
                        mpjve_cm_s=m["vel_error_m_s"] * 100, jitter_m_s3=m["jitter_m_s3"],
                        foot_skate_cm_s=skate * 100, gpen_mean_mm=gp_mean,
                        gpen_max_mm=gp_max, gpen_rate=gp_rate,
                        root_drift_cm=float(root_drift.mean() * 100),
                        root_drift_max_cm=float(root_drift.max() * 100),
                        pose_err_cm=float(local.mean() * 100),
                        trajectory=os.path.relpath(traj, REPO))

    # --- refinement check: does simulation improve physical quality without wrecking the pose? ---
    PR("")
    PR("## Refinement check — pre-sim (retargeted reference) vs post-sim (frozen PHC)")
    PR("")
    PR("| sequence | metric | pre-sim | post-sim | |")
    PR("|---|---|---|---|---|")
    for i, key in enumerate(keys):
        tag = key.replace("quest_retarget_", "")
        pred = np.asarray(d["pred_pos_all"][i])
        ref = np.asarray(d["ref_pos_all"][i])
        sel = MUJOCO_2_SMPL_REORDER[:NUM_SMPL_JOINTS]

        def jit(p):
            j = (p[3:] - 3 * p[2:-1] + 3 * p[1:-2] - p[:-3]) * (FPS ** 3)
            return float(np.linalg.norm(j, axis=-1).mean())

        a, b = compute_gpen(ref), compute_gpen(pred)
        rows = [("ground penetration mean (mm)", a[0], b[0]),
                ("ground penetration max (mm)", a[1], b[1]),
                ("ground penetration rate", a[2], b[2]),
                ("foot skate (cm/s)", compute_foot_skate(ref, FPS) * 100,
                 compute_foot_skate(pred, FPS) * 100),
                ("jitter (m/s^3)", jit(ref[:, sel]), jit(pred[:, sel]))]
        for nm, x, y in rows:
            verdict = "improved" if y < x - 1e-9 else ("unchanged" if abs(y - x) < 1e-9 else "WORSE")
            PR(f"| {tag} | {nm} | {x:.3f} | {y:.3f} | {verdict} |")
            out.setdefault(key, {}).setdefault("pre_vs_post", {})[nm] = {
                "pre": float(x), "post": float(y), "verdict": verdict}

    out["success_rate"] = float(d.get("success_rate", float("nan")))
    json.dump(out, open(OUT_JSON, "w"), indent=2, default=float)
    open(OUT_MD, "w").write("\n".join(LINES) + "\n")
    PR("")
    PR(f"report -> {os.path.relpath(OUT_MD, REPO)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
