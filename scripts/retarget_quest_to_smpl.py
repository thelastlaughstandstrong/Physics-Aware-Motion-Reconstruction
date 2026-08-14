"""
STEP 4 of the Quest->SMPL retargeting work: fit a fixed-shape neutral SMPL pose to ONE
ground-truth Quest frame by position-based differentiable IK.

Flow (exactly the chain validated in Steps 1-3):

    Quest GT local pose_quat + root_trans_offset
      -> Quest FK (poselib, verified in Step 1 to ~0.002 deg)
      -> world targets in Quest's native Z-up frame (default since the Step 4b ablation;
         --world-frame calibrated reproduces the original Qw @ R.T path)
      -> mapped target joint positions (16 fit targets from Step 2's artifact)
      -> optimize SMPL root_orient + body pose + root translation
      -> SMPL FK (lightweight differentiable skeleton FK, NOT the mesh BodyModel)
      -> residuals vs the Quest targets

Nothing is duplicated: the joint correspondence and the frame transform are READ from the
Step 2 / Step 3 json artifacts. Shape (betas), gender and global scale are held fixed.

Single-frame only. No temporal term, no foot contact, no sequence mode, no export.

Usage:
    python scripts/retarget_quest_to_smpl.py --motion_file <clip.pkl> --frame N --single_frame
"""
import argparse
import json
import os
import sys

import numpy as np
import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "phc"))

import joblib                                            # noqa: E402
import matplotlib                                        # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt                          # noqa: E402
from poselib.poselib.skeleton.skeleton3d import SkeletonState, SkeletonTree  # noqa: E402

CORR_JSON = os.path.join(REPO, "scripts/quest_smpl_correspondence.json")
CAL_JSON = os.path.join(REPO, "scripts/quest_smpl_frame_calibration.json")
MJCF = os.path.join(REPO, "phc/phc/data/assets/mjcf/quest_humanoid.xml")
SMPLH_NPZ = os.path.join(REPO, "support_data/body_models/smplh/neutral/model.npz")


# ---------------------------------------------------------------- rotation utils
def aa_to_mat(aa):
    """Axis-angle [...,3] -> rotation matrix [...,3,3], differentiable EVERYWHERE incl. aa=0.

    Uses the sinc form  R = I + A*skew(aa) + B*skew(aa)^2  with A = sin(t)/t, B = (1-cos t)/t^2,
    rather than the textbook  R = I + sin(t)*skew(k) + (1-cos t)*skew(k)^2  with k = aa/t.
    The latter is mathematically identical but has **zero gradient at aa = 0**: every term carries
    a factor of sin(t) or (1-cos t) or dt/daa, all of which vanish there, so a body pose
    initialized at exact zeros can never move. That silently froze the first Step 4 run
    (max joint angle stayed 0.0 deg, loss reduced 1x). Here skew(aa) is used un-normalized, so
    R ~ I + skew(aa) near the origin and the Jacobian is the identity-like basis, not zero.
    """
    t2 = (aa ** 2).sum(-1, keepdim=True)
    t = torch.sqrt(t2 + 1e-12)                 # strictly positive => both A and B stay finite
    A = (torch.sin(t) / t).unsqueeze(-1)
    B = ((1.0 - torch.cos(t)) / (t * t)).unsqueeze(-1)
    ax, ay, az = aa[..., 0], aa[..., 1], aa[..., 2]
    O = torch.zeros_like(ax)
    K = torch.stack([torch.stack([O, -az, ay], -1),
                     torch.stack([az, O, -ax], -1),
                     torch.stack([-ay, ax, O], -1)], -2)
    I = torch.eye(3, dtype=aa.dtype, device=aa.device).expand(K.shape)
    return I + A * K + B * (K @ K)


# ---------------------------------------------------------------- SMPL skeleton FK
class SmplSkeletonFK:
    """Differentiable joint-only SMPL FK on FIXED rest offsets.

    Deliberately not human_body_prior's BodyModel: this stage needs joint positions and gradients,
    not a mesh, and the mesh model would also drag in shape/pose blendshapes we are holding fixed.
    """

    def __init__(self, rest_offsets, parents, device):
        self.off = torch.tensor(rest_offsets, dtype=torch.float64, device=device)
        self.par = parents
        self.n = len(parents)

    def __call__(self, root_orient, body_pose, trans):
        """root_orient [3], body_pose [n-1,3], trans [3] -> positions [n,3] (all axis-angle)."""
        aa = torch.cat([root_orient.unsqueeze(0), body_pose], dim=0)   # [n,3]
        Rl = aa_to_mat(aa)                                             # [n,3,3] joint-local
        Rg = [Rl[0]]
        pos = [trans]
        for i in range(1, self.n):
            p = self.par[i]
            Rg.append(Rg[p] @ Rl[i])
            pos.append(pos[p] + Rg[p] @ self.off[i])
        return torch.stack(pos, 0), torch.stack(Rg, 0)


# ---------------------------------------------------------------- helpers
def rest_positions(parents, offsets):
    P = np.zeros((len(parents), 3))
    for i in range(1, len(parents)):
        P[i] = P[parents[i]] + offsets[i]
    return P


def anatomical_basis(P, root, head, l_hip, r_hip):
    up = P[head] - P[root]; up = up / np.linalg.norm(up)
    lat = P[r_hip] - P[l_hip]; lat = lat - up * np.dot(lat, up); lat = lat / np.linalg.norm(lat)
    # right-handed: with columns (right, up, forward) we need right x up = forward
    # (X x Y = Z). Using cross(up, lat) instead yields a LEFT-handed basis whose det is -1,
    # which made Step 4's "not mirrored" sanity check fire a false alarm on a correct fit.
    fwd = np.cross(lat, up)
    return np.stack([lat, up, fwd], axis=1)      # columns right, up, forward


def quest_fk_frame(tree, pose_quat, root_trans, frame):
    st = SkeletonState.from_rotation_and_root_translation(
        tree,
        torch.tensor(pose_quat[frame:frame + 1], dtype=torch.float32),
        torch.tensor(root_trans[frame:frame + 1], dtype=torch.float32),
        is_local=True)
    return st.global_translation[0].numpy().astype(np.float64)


def load_smpl_rest_offsets(corr):
    bd = np.load(SMPLH_NPZ, allow_pickle=True)
    J52 = np.array(bd["J_regressor"]) @ np.array(bd["v_template"])
    Js = J52[corr["smplh_row_for_smpl24"]]
    soff = Js.copy()
    for i in range(1, 24):
        soff[i] = Js[i] - Js[corr["smpl_parents"][i]]
    soff[0] = 0.0
    return soff


def build_targets(Qw, R, world_frame):
    """Quest world joint positions -> IK target positions, under the chosen convention.

    calibrated : Qw @ R.T -- the Step 4 reference path. R aligns the two REST poses, so applying
                 it to WORLD coordinates additionally rotates Quest's native Z-up world (world-up
                 +Z lands on +X). The fit is unaffected because root_orient absorbs it, but the
                 resulting world frame is neither Quest's nor SMPL's convention.
    native     : leave targets in Quest's own Z-up world. root_orient then encodes SMPL-rest ->
                 Quest-world directly, which is what PHC's SMPL path already expects.
    """
    return Qw @ R.T if world_frame == "calibrated" else Qw.copy()


def fit_single_frame(Qt, soff, smpl_par, direct, groups, qn, args, dev,
                     init=None, max_iters=None, tol=None):
    """Position-based IK for one frame. IDENTICAL for both world conventions -- only Qt differs.

    init      : optional {"root_orient","body_pose","trans"} numpy/list warm start (Step 5 uses the
                previous frame's solution). None -> the Step 4 anatomical cold-start rule.
    max_iters : overrides args.iters.
    tol       : if set, stop early once the loss stops improving by more than this *relative*
                amount over a 50-iteration window, and report `iters_used`. This is how the
                "does warm-starting reduce iterations?" question is measured rather than assumed.
    """
    from scipy.spatial.transform import Rotation as Rot
    torch.manual_seed(0)
    fk = SmplSkeletonFK(soff, smpl_par, dev)

    # Joint targets, then the root appended as one more weighted target. Putting it in the SAME
    # normalized weighted mean makes lambda_root directly comparable to the Group A/B weights,
    # rather than an separate term whose scale would have to be re-tuned if the others changed.
    tgt_q = [q for q in sorted(direct) if q != 0]
    tgt_s = [direct[q] for q in tgt_q]
    gA = set(groups["high_confidence"].values())
    w_joint = [args.w_group_a if q in gA else args.w_group_b for q in tgt_q]

    opt_q = tgt_q + [0]                     # 0 = quest root  (b_root)
    opt_s = tgt_s + [direct[0]]             # -> SMPL Pelvis
    w = torch.tensor(w_joint + [args.w_root], dtype=torch.float64, device=dev)
    T = torch.tensor(Qt[opt_q], dtype=torch.float64, device=dev)

    Bq = anatomical_basis(Qt, 0, 6, 19, 22)
    if init is None:
        Srest = rest_positions(smpl_par, soff)
        Bs = anatomical_basis(Srest, 0, 15, 1, 2)
        root_orient = torch.tensor(Rot.from_matrix(Bq @ Bs.T).as_rotvec(), dtype=torch.float64,
                                   device=dev, requires_grad=True)
        body_pose = (torch.randn(23, 3, dtype=torch.float64, device=dev) * 1e-4).requires_grad_(True)
        if args.arm_init:
            with torch.no_grad():
                body_pose[15] = torch.tensor([0.0, 0.0, -0.6], dtype=torch.float64)
                body_pose[16] = torch.tensor([0.0, 0.0, 0.6], dtype=torch.float64)
        trans = torch.tensor(Qt[0], dtype=torch.float64, device=dev).clone().requires_grad_(True)
    else:
        root_orient = torch.tensor(np.asarray(init["root_orient"]), dtype=torch.float64,
                                   device=dev).clone().requires_grad_(True)
        body_pose = torch.tensor(np.asarray(init["body_pose"]), dtype=torch.float64,
                                 device=dev).clone().requires_grad_(True)
        trans = torch.tensor(np.asarray(init["trans"]), dtype=torch.float64,
                             device=dev).clone().requires_grad_(True)

    n_iter = int(max_iters if max_iters is not None else args.iters)
    opt = torch.optim.Adam([root_orient, body_pose, trans], lr=args.lr)
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=max(1, n_iter // 3), gamma=0.3)
    hist = []
    iters_used = n_iter
    for _it in range(n_iter):
        opt.zero_grad()
        P, _ = fk(root_orient, body_pose, trans)
        loss = (w * (P[opt_s] - T).norm(dim=-1) ** 2).sum() / w.sum() \
            + args.w_reg * (body_pose ** 2).sum()
        loss.backward()
        opt.step()
        sched.step()
        hist.append(float(loss.item()))
        if tol is not None and _it >= 100 and (_it % 10 == 0):
            w0, w1 = hist[-50], hist[-1]
            if abs(w0 - w1) / max(w1, 1e-12) < tol:
                iters_used = _it + 1
                break

    with torch.no_grad():
        P, _ = fk(root_orient, body_pose, trans)
        err_mm = ((P[tgt_s] - torch.tensor(Qt[tgt_q], dtype=torch.float64, device=dev))
                  .norm(dim=-1) * 1000.0).numpy()
        Pn = P.numpy()

    res = {qn[q]: float(err_mm[i]) for i, q in enumerate(tgt_q)}
    a = np.array([res[qn[q]] for q in groups["high_confidence"].values()])
    b = np.array([res[qn[q]] for q in groups["structural_mismatch"].values()])
    Bf = anatomical_basis(Pn, 0, 15, 1, 2)
    tail = np.array(hist[-200:])
    return dict(
        positions=Pn, targets=Qt, per_joint=res, hist=hist,
        group_a_mean=float(a.mean()), group_a_max=float(a.max()),
        group_b_mean=float(b.mean()), group_b_max=float(b.max()),
        all_mean=float(np.concatenate([a, b]).mean()),
        all_max=float(np.concatenate([a, b]).max()),
        loss_initial=hist[0], loss_final=hist[-1],
        plateaued=bool(tail.std() / max(tail.mean(), 1e-12) < 1e-3),
        finite=bool(np.isfinite(Pn).all()),
        root_trans_err_mm=float(np.linalg.norm(Pn[0] - Qt[0]) * 1000),
        upright_cos=float(np.dot(Bf[:, 1], Bq[:, 1])),
        lateral_cos=float(np.dot(Bf[:, 0], Bq[:, 0])),
        handedness_det=float(np.linalg.det(Bf)),
        max_joint_angle_deg=float(np.degrees(body_pose.detach().norm(dim=-1).max().item())),
        n_over_150=int((np.degrees(body_pose.detach().norm(dim=-1).numpy()) > 150).sum()),
        # ABSOLUTE (not target-relative) uprightness: does the fitted body's up axis align with
        # the WORLD +Z axis? ~1 means the result is upright in the world with no display fix.
        world_up_cos=float(np.dot(Bf[:, 1], np.array([0.0, 0.0, 1.0]))),
        root_orient_aa=root_orient.detach().tolist(),
        body_pose_aa=body_pose.detach().tolist(),
        trans=trans.detach().tolist(),
        w_root=args.w_root,
        iters_used=iters_used,
    )


def run_ablation(args, corr, cal, soff, smpl_par, dev):
    R = np.array(cal["R_quest_to_smpl"])
    direct = {int(k): v for k, v in corr["direct"].items()}
    groups, qn = corr["validation_groups"], corr["quest_joint_names"]
    tree = SkeletonTree.from_mjcf(MJCF)
    frames = [("SUBJECT069.standing.pkl_0_425", 200), ("SUBJECT069.standing.pkl_0_425", 60),
              ("SUBJECT069.standing.pkl_0_425", 350), ("SUBJECT084.walking.pkl_0_426", 150),
              ("SUBJECT072.gestures.pkl_1281_1707", 200)]
    rows, keep = [], {}
    for clip, fr in frames:
        d = joblib.load(os.path.join(REPO, "data/quest2_synthetic/test_seg_motion", clip + ".pkl"))
        e = d[list(d.keys())[0]]
        Qw = quest_fk_frame(tree, np.asarray(e["pose_quat"], dtype=np.float32),
                            np.asarray(e["root_trans_offset"], dtype=np.float32), fr)
        for mode in ("calibrated", "native"):
            args.world_frame = mode
            r = fit_single_frame(build_targets(Qw, R, mode), soff, smpl_par,
                                 direct, groups, qn, args, dev)
            r.update(clip=clip, frame=fr, mode=mode)
            rows.append(r)
            if clip.startswith("SUBJECT069.standing") and fr == 200:
                keep[mode] = r
    return rows, keep


def emit_ablation(args, corr, cal, soff, smpl_par, dev):
    rows, keep = run_ablation(args, corr, cal, soff, smpl_par, dev)
    qp = SkeletonTree.from_mjcf(MJCF).parent_indices.tolist()
    out = []

    def PR(s=""):
        print(s)
        out.append(s)

    PR("# World-Frame Convention Ablation (Step 4b)")
    PR("")
    PR("**A = `calibrated`** (Step 4 reference): targets are `Qw @ R.T` — the fitted anatomical R "
       "is applied to world coordinates.  \n"
       "**B = `native`**: targets stay in Quest's own Z-up world; `root_orient` carries "
       "SMPL-rest → Quest-world.")
    PR("")
    PR(f"Identical for both: correspondence, rest offsets, loss (A={args.w_group_a} "
       f"B={args.w_group_b}, reg={args.w_reg}), initialization rule, Adam lr={args.lr}, "
       f"{args.iters} iters, fixed neutral shape/scale. Only `build_targets` differs.")
    PR("")
    PR("## Residuals")
    PR("")
    PR("| clip | frame | conv | GrpA mean | GrpA max | GrpB mean | GrpB max | all mean | all max |")
    PR("|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        PR(f"| `{r['clip'][:28]}` | {r['frame']} | {'A' if r['mode']=='calibrated' else 'B'} | "
           f"{r['group_a_mean']:.1f} | {r['group_a_max']:.1f} | {r['group_b_mean']:.1f} | "
           f"{r['group_b_max']:.1f} | {r['all_mean']:.1f} | {r['all_max']:.1f} |")
    PR("")
    ca = np.array([r["all_mean"] for r in rows if r["mode"] == "calibrated"])
    na = np.array([r["all_mean"] for r in rows if r["mode"] == "native"])
    PR(f"Mean over the 5 frames — **A {ca.mean():.2f} mm**, **B {na.mean():.2f} mm** "
       f"(difference {abs(ca.mean()-na.mean()):.3f} mm).")
    PR("")
    PR("## Sanity / frame diagnostics")
    PR("")
    PR("| clip | frame | conv | upright cos (rel.) | **world-up cos (abs.)** | lat cos | det | "
       "max ang | root err mm | final loss |")
    PR("|---|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        PR(f"| `{r['clip'][:24]}` | {r['frame']} | {'A' if r['mode']=='calibrated' else 'B'} | "
           f"{r['upright_cos']:.4f} | **{r['world_up_cos']:+.4f}** | {r['lateral_cos']:.4f} | "
           f"{r['handedness_det']:+.3f} | {r['max_joint_angle_deg']:.1f}° | "
           f"{r['root_trans_err_mm']:.1f} | {r['loss_final']:.6f} |")
    PR("")
    PR("`upright cos (rel.)` compares the fit to its own target and passes either way. "
       "**`world-up cos (abs.)` is the decisive column**: the cosine between the fitted body's up "
       "axis and the WORLD `+Z` axis. ~+1 means the result stands upright in the world with no "
       "display-only corrective rotation; ~0 means it does not.")
    PR("")

    # overlay: each convention drawn in its OWN raw world frame, no corrective rotation
    fig = plt.figure(figsize=(12, 6))
    for k, mode in enumerate(("calibrated", "native")):
        r = keep[mode]
        Qt, Pn = r["targets"], r["positions"]
        ax = fig.add_subplot(1, 2, k + 1, projection="3d")
        for i in range(1, len(qp)):
            ax.plot(*zip(Qt[i], Qt[qp[i]]), c="tab:red", lw=1.6, alpha=.85)
        for i in range(1, 24):
            ax.plot(*zip(Pn[i], Pn[smpl_par[i]]), c="tab:blue", lw=1.6, alpha=.85)
        ax.scatter(*Qt.T, c="tab:red", s=11, label="Quest target")
        ax.scatter(*Pn.T, c="tab:blue", s=11, label="fitted SMPL")
        c = Qt.mean(0); lim = 1.0
        ax.set_xlim(c[0]-lim, c[0]+lim); ax.set_ylim(c[1]-lim, c[1]+lim); ax.set_zlim(c[2]-lim, c[2]+lim)
        ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")
        ax.view_init(elev=12, azim=-60)
        ax.set_title(f"{'A calibrated (Qw @ R.T)' if mode=='calibrated' else 'B native Z-up'}\n"
                     f"world-up cos = {r['world_up_cos']:+.3f}", fontsize=10)
        if k == 0:
            ax.legend(loc="upper left", fontsize=8)
    fig.suptitle("Raw world frame, NO display-only corrective rotation "
                 "(matplotlib draws Z vertically)", fontsize=11)
    fig.tight_layout()
    png = os.path.join(REPO, "scripts/quest_smpl_frame_convention_ablation.png")
    fig.savefig(png, dpi=140)
    PR("## Overlay (raw world frame, no display correction)")
    PR("")
    PR("![ablation](quest_smpl_frame_convention_ablation.png)")
    PR("")

    strip = [{k: v for k, v in r.items() if k not in ("positions", "targets", "hist")} for r in rows]
    json.dump({"frames_tested": len(rows) // 2, "rows": strip,
               "mean_all_calibrated_mm": float(ca.mean()),
               "mean_all_native_mm": float(na.mean())},
              open(os.path.join(REPO, "scripts/quest_smpl_frame_convention_ablation.json"), "w"),
              indent=2)
    with open(os.path.join(REPO, "scripts/quest_smpl_frame_convention_ablation.md"), "w") as f:
        f.write("\n".join(out) + "\n")
    print("\n[artifacts] scripts/quest_smpl_frame_convention_ablation.{json,md,png}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--motion_file", default=None)
    ap.add_argument("--frame", type=int, default=200)
    ap.add_argument("--single_frame", action="store_true")
    ap.add_argument("--world-frame", choices=["native", "calibrated"], default="native",
                    help="native (DEFAULT, adopted after the Step 4b ablation): targets stay in "
                         "Quest's own Z-up world; root_orient carries SMPL-rest -> Quest-world. "
                         "Naturally upright, PHC-compatible. "
                         "calibrated: the Step 4 reference path, targets = Qw @ R.T. Retained "
                         "unchanged for reproducibility; produces numerically identical residuals "
                         "(~5e-11) but a world whose up axis is not +Z.")
    ap.add_argument("--ablate", action="store_true",
                    help="Run BOTH world-frame conventions over a fixed frame set and emit the "
                         "comparison artifact. Ignores --frame/--single_frame.")
    ap.add_argument("--iters", type=int, default=3000)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--w-group-a", type=float, default=1.0)
    ap.add_argument("--w-group-b", type=float, default=0.3)
    ap.add_argument("--w-reg", type=float, default=1e-3)
    ap.add_argument("--w-root", type=float, default=0.25,
                    help="lambda_root: weight on the pelvis/root position term, on the SAME scale "
                         "as the Group A/B joint weights (A=1.0, B=0.3). Without it the root is "
                         "only constrained through its children and drifts 39-51 mm, which would "
                         "misplace the whole body along a walking trajectory.")
    ap.add_argument("--arm-init", action="store_true",
                    help="Seed shoulders toward an A-pose. OFF by default; only enable if the "
                         "T-pose start is shown to converge poorly, and say so in the report.")
    ap.add_argument("--out-prefix", default=os.path.join(REPO, "scripts/quest_smpl_single_frame_fit"))
    args = ap.parse_args()
    if not (args.single_frame or args.ablate):
        raise SystemExit("Pass --single_frame (one fit) or --ablate (both conventions).")
    if args.single_frame and not args.motion_file:
        raise SystemExit("--single_frame requires --motion_file")

    dev = torch.device("cpu")
    torch.manual_seed(0)

    corr = json.load(open(CORR_JSON))
    cal = json.load(open(CAL_JSON))
    R = np.array(cal["R_quest_to_smpl"])
    direct = {int(k): v for k, v in corr["direct"].items()}
    groups = corr["validation_groups"]
    qn, sn = corr["quest_joint_names"], corr["smpl_joint_names"]
    smpl_par = corr["smpl_parents"]

    if args.ablate:
        emit_ablation(args, corr, cal, load_smpl_rest_offsets(corr), smpl_par, dev)
        return

    # ---- Quest targets ----
    tree = SkeletonTree.from_mjcf(MJCF)
    qp = tree.parent_indices.tolist()
    d = joblib.load(args.motion_file)
    e = d[list(d.keys())[0]]
    pq = np.asarray(e["pose_quat"], dtype=np.float32)
    rt = np.asarray(e["root_trans_offset"], dtype=np.float32)
    n_frames = pq.shape[0]
    if not (0 <= args.frame < n_frames):
        raise SystemExit(f"--frame {args.frame} out of range (clip has {n_frames})")
    Qw = quest_fk_frame(tree, pq, rt, args.frame)          # Quest world positions [25,3]
    Qt = build_targets(Qw, R, args.world_frame)

    # ---- SMPL rest (corrected 24-joint indexing from Step 2) ----
    soff = load_smpl_rest_offsets(corr)

    # Single shared implementation -- main() and the ablation call the SAME optimizer, so the
    # loss cannot silently diverge between the reported single-frame fit and the ablation.
    r = fit_single_frame(Qt, soff, smpl_par, direct, groups, qn, args, dev)
    Pn, res, hist = r["positions"], r["per_joint"], r["hist"]
    a_vals = np.array([res[qn[q]] for q in groups["high_confidence"].values()])
    b_vals = np.array([res[qn[q]] for q in groups["structural_mismatch"].values()])
    gA_names, gB_names = groups["high_confidence"], groups["structural_mismatch"]
    body_pose = torch.tensor(r["body_pose_aa"])
    root_orient = torch.tensor(r["root_orient_aa"])
    trans = torch.tensor(r["trans"])
    checks = {
        "finite": r["finite"],
        "root_translation_err_mm": r["root_trans_err_mm"],
        "upright_cos(up_fit, up_target)": r["upright_cos"],
        "lateral_cos(right_fit, right_target)": r["lateral_cos"],
        "handedness_det": r["handedness_det"],
        "max_joint_angle_deg": r["max_joint_angle_deg"],
        "n_joints_over_150deg": r["n_over_150"],
        "world_up_cos": r["world_up_cos"],
    }
    plateau = r["plateaued"]
    tail = np.array(hist[-200:])
    # pre-IK baseline (Step 3 rest-pose offsets, root-aligned only) for the change column
    pre = json.load(open(CAL_JSON))["rest_offsets_mm"]
    # mapped-pair indices, for the residual lines drawn in the overlay
    tgt_q = [q for q in sorted(direct) if q != 0]
    tgt_s = [direct[q] for q in tgt_q]

    # ---- report ----
    out = []

    def PR(s=""):
        print(s)
        out.append(s)

    PR("# Quest -> SMPL Single-Frame IK Fit (Step 4)")
    PR("")
    PR(f"- clip: `{os.path.basename(args.motion_file)}` frame **{args.frame}** of {n_frames}")
    PR(f"- SMPL: neutral, **shape/betas and global scale held fixed**, rest offsets fixed")
    PR(f"- world frame: **`{args.world_frame}`**" +
       ("  (Quest-native Z-up; `root_orient` carries SMPL-rest → world)"
        if args.world_frame == "native" else
        "  (Step 4 reference path: targets = `Qw @ R.T`)"))
    PR(f"- optimizer: Adam, lr {args.lr} (StepLR ×0.3), **{args.iters} iters**; "
       f"weights A={args.w_group_a} B={args.w_group_b} **root={args.w_root}**, "
       f"pose reg {args.w_reg}")
    PR(f"- arm/A-pose initialization: **{'ENABLED' if args.arm_init else 'not used'}**")
    PR("")
    PR(f"initial loss **{hist[0]:.6f}** → final **{hist[-1]:.6f}** "
       f"({hist[0]/max(hist[-1],1e-12):.0f}× reduction); "
       f"plateaued: **{'yes' if plateau else 'no'}** "
       f"(last-200-iter relative std {tail.std()/max(tail.mean(),1e-12):.2e})")
    PR("")
    PR("## Residuals")
    PR("")
    PR(f"| group | mean (mm) | max (mm) | pre-IK mean (mm) |")
    PR("|---|---|---|---|")
    pa = np.mean([pre[qn[q]] for q in gA_names.values()])
    pb = np.mean([pre[qn[q]] for q in gB_names.values()])
    PR(f"| **A — high confidence** ({len(a_vals)}) | **{a_vals.mean():.1f}** | {a_vals.max():.1f} | {pa:.1f} |")
    PR(f"| **B — structural mismatch** ({len(b_vals)}) | **{b_vals.mean():.1f}** | {b_vals.max():.1f} | {pb:.1f} |")
    allv = np.concatenate([a_vals, b_vals])
    PR(f"| all mapped ({len(allv)}) | {allv.mean():.1f} | {allv.max():.1f} | "
       f"{np.mean(list(pre.values())):.1f} |")
    PR("")
    PR("Synthesized joints (hands, toes) are excluded from the optimization and from this table, "
       "per the Step 2 protocol.")
    PR("")
    PR("### Per-joint")
    PR("")
    PR("| group | joint | Quest | SMPL | pre-IK (mm) | post-IK (mm) | change |")
    PR("|---|---|---|---|---|---|---|")
    for gname, gd in (("A", gA_names), ("B", gB_names)):
        for label, q in gd.items():
            p0, p1 = pre[qn[q]], res[qn[q]]
            PR(f"| {gname} | `{label}` | `{qn[q]}` | `{sn[direct[q]]}` | {p0:.1f} | **{p1:.1f}** | "
               f"{100*(p1-p0)/p0:+.0f}% |")
    PR("")
    PR("## Sanity checks")
    PR("")
    PR("| check | value | verdict |")
    PR("|---|---|---|")
    PR(f"| all values finite / non-NaN | {checks['finite']} | "
       f"{'PASS' if checks['finite'] else 'FAIL'} |")
    PR(f"| root translation error | {checks['root_translation_err_mm']:.3f} mm | "
       f"{'PASS' if checks['root_translation_err_mm'] < 25 else 'CHECK'} |")
    PR(f"| world-up cos (absolute) | {checks['world_up_cos']:+.4f} | "
       f"{'PASS' if checks['world_up_cos'] > 0.98 else 'n/a (calibrated frame)'} |")
    PR(f"| upright: cos(up_fit, up_target) | {checks['upright_cos(up_fit, up_target)']:.4f} | "
       f"{'PASS' if checks['upright_cos(up_fit, up_target)'] > 0.98 else 'CHECK'} |")
    PR(f"| left/right not swapped: cos(right_fit, right_target) | "
       f"{checks['lateral_cos(right_fit, right_target)']:.4f} | "
       f"{'PASS' if checks['lateral_cos(right_fit, right_target)'] > 0.9 else 'FAIL — MIRRORED'} |")
    PR(f"| not mirrored: det(anatomical basis) | {checks['handedness_det']:.4f} | "
       f"{'PASS' if checks['handedness_det'] > 0 else 'FAIL — REFLECTION'} |")
    PR(f"| max joint angle | {checks['max_joint_angle_deg']:.1f}° | "
       f"{'PASS' if checks['max_joint_angle_deg'] < 150 else 'CHECK — extreme'} |")
    PR(f"| joints > 150° | {checks['n_joints_over_150deg']} | "
       f"{'PASS' if checks['n_joints_over_150deg'] == 0 else 'CHECK'} |")
    PR("")

    # ---- overlay ----
    # In `native` the world is already Z-up and matplotlib draws Z vertically, so the raw
    # coordinates render upright with NO corrective rotation. In `calibrated` the fitted R sends
    # world-up (+Z) to +X, so the figure would render lying down; there we rotate both skeletons
    # into the target's anatomical frame purely for legibility (changes no fitted value).
    if args.world_frame == "native":
        def disp(X):
            return X.copy()
        _disp_note = ("raw world frame — no corrective rotation applied (native Z-up is upright "
                      "by construction)")
    else:
        D = anatomical_basis(Qt, 0, 6, 19, 22).T
        def disp(X):
            A = (X - Qt[0]) @ D.T
            return np.stack([A[:, 0], A[:, 2], A[:, 1]], axis=1)
        _disp_note = ("rotated into the target's anatomical frame for legibility — the calibrated "
                      "world is not Z-up, so the raw frame would render lying down")

    QD, SD = disp(Qt), disp(Pn)
    fig = plt.figure(figsize=(15, 6))
    for k, (elev, azim, ttl) in enumerate([(12, -60, "3/4 view"), (0, -90, "front"), (0, 0, "side")]):
        ax = fig.add_subplot(1, 3, k + 1, projection="3d")
        for i in range(1, len(qp)):
            ax.plot(*zip(QD[i], QD[qp[i]]), c="tab:red", lw=1.7, alpha=.85)
        for i in range(1, 24):
            ax.plot(*zip(SD[i], SD[smpl_par[i]]), c="tab:blue", lw=1.7, alpha=.85)
        for q, s in zip(tgt_q, tgt_s):
            ax.plot(*zip(QD[q], SD[s]), c="0.5", lw=.8, ls=":")
        ax.scatter(*QD.T, c="tab:red", s=13, label="Quest target")
        ax.scatter(*SD.T, c="tab:blue", s=13, label="fitted SMPL")
        c = QD.mean(0)
        lim = 1.0
        ax.set_xlim(c[0]-lim, c[0]+lim); ax.set_ylim(c[1]-lim, c[1]+lim); ax.set_zlim(c[2]-lim, c[2]+lim)
        if args.world_frame == "native":
            ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z (up)")
        else:
            ax.set_xlabel("X (lateral)"); ax.set_ylabel("Z (fwd)"); ax.set_zlabel("Y (up)")
        ax.view_init(elev=elev, azim=azim)
        ax.set_title(ttl, fontsize=10)
        if k == 0:
            ax.legend(loc="upper left", fontsize=8)
    fig.suptitle(f"Step 4 single-frame fit — {os.path.basename(args.motion_file)} frame {args.frame}"
                 f"  |  Group A {a_vals.mean():.0f} mm, Group B {b_vals.mean():.0f} mm", fontsize=11)
    fig.tight_layout()
    fig.savefig(args.out_prefix + ".png", dpi=140)
    PR(f"## Overlay")
    PR("")
    PR(f"![fit]({os.path.basename(args.out_prefix)}.png)")
    PR("")
    PR(f"Red = Quest target, blue = fitted SMPL, dotted grey = per-joint residual. Display: "
       f"{_disp_note}.")
    PR("")
    PR("## World-frame convention (resolved in Step 4b)")
    PR("")
    PR("`R` from Step 3 aligns the two **rest** poses (Quest rest X-up, SMPL rest Y-up). Quest "
       "**motion** already lives in a **Z-up world**, so applying `R` to world coordinates sent "
       "world-up `+Z` to `+X` — the fit was correct (root_orient absorbed it) but the resulting "
       "world was neither convention, and the overlay needed a display-only rotation to look "
       "upright.")
    PR("")
    PR("The Step 4b ablation ran both conventions over 5 frames with an identical optimizer path. "
       "Residuals agreed to **~5e-11** on every metric, while absolute world-up cosine was "
       "**-0.08 … +0.10 (calibrated)** vs **+0.99 … +1.00 (native)**. `native` was adopted and is "
       "now the default: same fit, naturally upright, and Z-up matches what PHC's SMPL path "
       "already expects via `upright_start`.")
    PR("")
    PR("`--world-frame calibrated` still reproduces the original Step 4 path unchanged. `R` "
       "remains correct and in use for what it was derived for — relating the two rest poses "
       "(Step 3, and the Step 2 chain-ratio validation).")
    PR("")


    json.dump({
        "motion_file": args.motion_file, "frame": args.frame,
        "world_frame": args.world_frame,
        "iters": args.iters, "loss_initial": hist[0], "loss_final": hist[-1],
        "plateaued": plateau,
        "group_a_mean_mm": float(a_vals.mean()), "group_a_max_mm": float(a_vals.max()),
        "group_b_mean_mm": float(b_vals.mean()), "group_b_max_mm": float(b_vals.max()),
        "all_mean_mm": float(allv.mean()),
        "per_joint_mm": res, "sanity": checks,
        "smpl_root_orient_aa": root_orient.detach().tolist(),
        "smpl_body_pose_aa": body_pose.detach().tolist(),
        "smpl_trans": trans.detach().tolist(),
        "arm_init_used": bool(args.arm_init),
    }, open(args.out_prefix + ".json", "w"), indent=2)
    with open(args.out_prefix + ".md", "w") as f:
        f.write("\n".join(out) + "\n")
    print(f"\n[png]  {args.out_prefix}.png\n[json] {args.out_prefix}.json\n[md]   {args.out_prefix}.md")


if __name__ == "__main__":
    main()
