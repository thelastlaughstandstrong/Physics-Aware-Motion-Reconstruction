"""
Full-dataset mesh-level GPen over the 1072-clip Quest2 test set.

The per-frame IK in mesh_gpen_quest.py costs ~343 ms/frame sequentially -- 296 hours for the
7 stream-stages x 1072 clips asked for here. This module keeps the SAME objective, optimizer,
schedule and warm-start semantics but moves the BATCH DIMENSION FROM FRAMES TO CLIPS: frame t is
solved for every clip simultaneously on the GPU, marching t forward so each frame still warm-starts
from the previous frame's solution exactly as the validated Step 5 pipeline does.

Because each clip's parameters are separate tensor rows and the batch loss is a SUM over clips,
every clip's gradient is identical to what it would be if fitted alone -- batching changes speed,
not the optimization problem.

Objective (verbatim from retarget_quest_to_smpl.fit_single_frame):
    loss_b = sum_j w_j * ||P_b[opt_s_j] - T_b[j]||^2 / sum_j w_j  +  w_reg * ||body_pose_b||^2
    Adam(lr=0.05), StepLR(step_size=n_iter//3, gamma=0.3)
    cold init: root_orient from the anatomical basis, body_pose ~ 1e-4 noise, trans = Qt[root]

GPen is then utils.physical_metrics.compute_ground_penetration, unchanged: mean depth over
PENETRATING (vertex,frame) pairs, frame-based rate, tolerance 1e-4 m, floor Z=0, SMPL-H neutral,
betas=0, 6890 vertices.

Must reproduce the three validated clips before its aggregates are trusted (--validate).
"""
import argparse
import json
import os
import sys
import time

import joblib
import numpy as np
import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "scripts"))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "phc"))

import retarget_quest_to_smpl as M                                      # noqa: E402
from poselib.poselib.skeleton.skeleton3d import SkeletonTree            # noqa: E402
from human_body_prior.body_model.body_model import BodyModel            # noqa: E402
from utils.physical_metrics import compute_ground_penetration           # noqa: E402
from mesh_gpen_quest import BM_NEUTRAL, DMPL_NEUTRAL, IKArgs  # noqa: E402
from poselib.poselib.skeleton.skeleton3d import SkeletonState as _SkState  # noqa: E402


def quest_fk_all(tree, pose_quat, root_trans):
    """Whole-clip Quest FK in ONE poselib call. mesh_gpen_quest calls SkeletonState per frame,
    which dominates load time at dataset scale; this is the identical computation batched."""
    st = _SkState.from_rotation_and_root_translation(
        tree, torch.tensor(pose_quat, dtype=torch.float32),
        torch.tensor(root_trans, dtype=torch.float32), is_local=True)
    return st.global_translation.numpy().astype(np.float64)


def gt_positions(clip, tree):
    """GT 25-joint positions under PHC's own loader vertical convention (see mesh_gpen_quest)."""
    d = joblib.load(os.path.join(REPO, f"data/quest2_synthetic/test_seg_motion/{clip}.pkl"))
    e = d[list(d.keys())[0]]
    P = quest_fk_all(tree, np.asarray(e["pose_quat"], dtype=np.float32),
                     np.asarray(e["root_trans_offset"], dtype=np.float32))
    P[..., 2] -= float(P[:30, :, 2].min()) - 0.0871
    return P

CORR = os.path.join(REPO, "scripts/quest_smpl_correspondence.json")
OUTDIR = os.path.join(REPO, "data/quest_smpl_retarget/dataset_gpen")


def aa_to_mat_b(aa):
    """Batched sinc-form Rodrigues -- same formulation as retarget_quest_to_smpl.aa_to_mat."""
    t2 = (aa ** 2).sum(-1, keepdim=True)
    t = torch.sqrt(t2 + 1e-12)
    A = (torch.sin(t) / t).unsqueeze(-1)
    B = ((1.0 - torch.cos(t)) / (t * t)).unsqueeze(-1)
    ax, ay, az = aa[..., 0], aa[..., 1], aa[..., 2]
    O = torch.zeros_like(ax)
    K = torch.stack([torch.stack([O, -az, ay], -1),
                     torch.stack([az, O, -ax], -1),
                     torch.stack([-ay, ax, O], -1)], -2)
    I = torch.eye(3, dtype=aa.dtype, device=aa.device).expand(K.shape)
    return I + A * K + B * (K @ K)


def fk_batch(root_aa, body_aa, trans, off, parents):
    """(B,3),(B,23,3),(B,3) -> (B,24,3) world joint positions."""
    B = root_aa.shape[0]
    aa = torch.cat([root_aa.unsqueeze(1), body_aa], dim=1)          # (B,24,3)
    Rl = aa_to_mat_b(aa)
    Rg = [Rl[:, 0]]
    pos = [trans]
    for i in range(1, 24):
        p = parents[i]
        Rg.append(Rg[p] @ Rl[:, i])
        pos.append(pos[p] + (Rg[p] @ off[i].unsqueeze(-1)).squeeze(-1))
    return torch.stack(pos, 1)


def batched_ik(Qt_list, soff, corr, dev, iters_first=3000, iters_warm=600, lr=0.05, w_reg=1e-3,
               w_ga=1.0, w_gb=0.3, w_root=0.25, log=None):
    """Qt_list: list of (T_i,25,3) target arrays. Returns list of dicts of fitted params."""
    parents = corr["smpl_parents"]
    direct = {int(k): v for k, v in corr["direct"].items()}
    groups = corr["validation_groups"]
    gA = set(groups["high_confidence"].values())
    tgt_q = [q for q in sorted(direct) if q != 0]
    opt_q = tgt_q + [0]
    opt_s = [direct[q] for q in tgt_q] + [direct[0]]
    w = torch.tensor([(w_ga if q in gA else w_gb) for q in tgt_q] + [w_root],
                     dtype=torch.float64, device=dev)
    off = torch.tensor(soff, dtype=torch.float64, device=dev)
    B = len(Qt_list)
    lens = np.array([len(q) for q in Qt_list])
    Tmax = int(lens.max())

    # pad targets to a dense (B, Tmax, 25, 3) tensor; masked rows simply stop being updated
    Q = np.zeros((B, Tmax, 25, 3))
    for i, q in enumerate(Qt_list):
        Q[i, :len(q)] = q
        Q[i, len(q):] = q[-1]
    Q = torch.tensor(Q, dtype=torch.float64, device=dev)

    # cold init at frame 0 (anatomical basis per clip), matching fit_single_frame's rule
    Srest = M.rest_positions(parents, soff)
    Bs = M.anatomical_basis(Srest, 0, 15, 1, 2)
    from scipy.spatial.transform import Rotation as Rot
    r0 = np.stack([Rot.from_matrix(M.anatomical_basis(Qt_list[i][0], 0, 6, 19, 22) @ Bs.T).as_rotvec()
                   for i in range(B)])
    torch.manual_seed(0)
    root = torch.tensor(r0, dtype=torch.float64, device=dev).requires_grad_(True)
    body = (torch.randn(B, 23, 3, dtype=torch.float64, device=dev) * 1e-4).requires_grad_(True)
    tr = Q[:, 0, 0, :].clone().requires_grad_(True)

    out = [dict(root=np.zeros((lens[i], 3)), body=np.zeros((lens[i], 23, 3)),
                trans=np.zeros((lens[i], 3)), err=np.zeros(lens[i])) for i in range(B)]
    t_start = time.time()
    for t in range(Tmax):
        n_iter = iters_first if t == 0 else iters_warm
        opt = torch.optim.Adam([root, body, tr], lr=lr)
        sched = torch.optim.lr_scheduler.StepLR(opt, step_size=max(1, n_iter // 3), gamma=0.3)
        Tt = Q[:, t][:, opt_q]                                   # (B,17,3)
        for _ in range(n_iter):
            opt.zero_grad()
            P = fk_batch(root, body, tr, off, parents)
            d2 = ((P[:, opt_s] - Tt) ** 2).sum(-1)               # (B,17)
            loss = ((w * d2).sum(-1) / w.sum()).sum() + w_reg * (body ** 2).sum()
            loss.backward()
            opt.step()
            sched.step()
        with torch.no_grad():
            P = fk_batch(root, body, tr, off, parents)
            err = ((P[:, [direct[q] for q in tgt_q]] - Q[:, t][:, tgt_q]).norm(dim=-1)
                   * 1000.0).mean(-1).cpu().numpy()
            rn, bn, tn = root.cpu().numpy(), body.cpu().numpy(), tr.cpu().numpy()
        for i in range(B):
            if t < lens[i]:
                out[i]["root"][t], out[i]["body"][t] = rn[i], bn[i]
                out[i]["trans"][t], out[i]["err"][t] = tn[i], err[i]
        if log and (t % 25 == 0 or t == Tmax - 1):
            el = time.time() - t_start
            print(f"    frame {t+1}/{Tmax}  {el:.0f}s elapsed, "
                  f"eta {el/(t+1)*(Tmax-t-1):.0f}s", flush=True)
    return out


def mesh_gpen_from_params(p, bm, device, batch=256):
    T = len(p["root"])
    root = torch.tensor(p["root"], dtype=torch.float32)
    body = torch.tensor(p["body"], dtype=torch.float32)
    trans = torch.tensor(p["trans"], dtype=torch.float32)
    pose_body = body[:, :21].reshape(T, 63)
    mm_all, fmax_all = [], []
    with torch.no_grad():
        for s in range(0, T, batch):
            e = min(s + batch, T)
            v = bm(pose_body=pose_body[s:e].to(device), root_orient=root[s:e].to(device),
                   trans=trans[s:e].to(device)).v
            pen = torch.clamp(-v[..., 2], min=0.0)
            fmax_all.append(pen.amax(-1).cpu())
            mm_all.append(pen[pen > 1e-4].cpu())
    fmax = torch.cat(fmax_all)
    pens = torch.cat(mm_all) if sum(x.numel() for x in mm_all) else torch.tensor([])
    return (float(pens.mean() * 1000) if pens.numel() else 0.0,
            float(fmax.max() * 1000), float((fmax > 1e-4).float().mean()))


STREAMS = {
    "gt":                 ("pre",  "gt"),
    # Phase A = the standalone AMASS-finetuned AvatarPoser, i.e. the kinematic backbone the naive
    # baseline tracks -- so this row doubles as Naive PHC's pre-sim reference.
    "phase_a":            ("pre",  "backbone:options/train_avatarposer_quest2_finetune.json:"
                                   "results/AvatarPoseEstimationQuest2Finetune/models/15001_G.pth"),
    "fb_pre":             ("pre",  "backbone:options/train_avatarposer_quest2.json:"
                                   "model_zoo/quest_backbone_trackerr_extracted.pth"),
    "finetuned_backbone": ("post", f"{REPO}/output/HumanoidIm/quest_distill_trackerr/phc_positions_for_eval.pkl"),
    "naive":              ("post", f"{REPO}/output/HumanoidIm/phc_prim_quest/phc_positions_for_eval.pkl"),
    "simxr":              ("post", "/home/sia-irl/SimXR/output/HumanoidIm/simxr_quest_novision_lrdecay/phc_positions_for_eval.pkl"),
}


def backbone_positions(opt_path, ckpt, clips, tree):
    """Open-loop backbone q_base -> 25-joint world positions, exactly as
    scripts/presim_gpen_aggregate.py does (HMD inputs come from the dataset, never the simulator,
    so a standalone forward pass reproduces the q_base a closed-loop rollout would have seen).
    Anchored on the known head position; then put on the SAME vertical convention as every other
    stream so pre- and post-sim numbers are comparable."""
    from utils import utils_option as option
    from data.select_dataset import define_Dataset
    from models.select_model import define_G
    from models.quest_body_model import QUEST_HEAD_IDX, QUEST_NUM_JOINTS
    from scipy.spatial.transform import Rotation as sRot

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    opt = option.dict_to_nonedict(option.parse(os.path.join(REPO, opt_path), is_train=True))
    ts = define_Dataset(opt["datasets"]["test"])
    window = opt["datasets"]["test"]["window_size"]
    names = [os.path.basename(x)[:-4] for x in ts.filename_list]
    net = define_G(opt).to(dev)
    sd = torch.load(os.path.join(REPO, ckpt), map_location=dev)
    net.load_state_dict(sd, strict=True)
    net.eval()

    def s6_to_q(x):
        v = x.reshape(-1, 6)
        c1, c2 = v[:, 0:3], v[:, 3:6]
        R = np.stack([c1, c2, np.cross(c1, c2)], -1)
        return sRot.from_matrix(R).as_quat().reshape(*x.shape[:-1], 4)

    out = {}
    for c in clips:
        if c not in names:
            continue
        d = ts[names.index(c)]
        L = d["L"].to(dev)
        head = d["Head_trans_global"].numpy()
        N = L.shape[0] - window
        if N < 5:
            continue
        win = torch.stack([L[k:k + window + 1] for k in range(N)], 0)
        with torch.no_grad():
            go, jr = net(win, do_fk=False)
        pq = s6_to_q(torch.cat([go, jr], -1).cpu().numpy().reshape(N, QUEST_NUM_JOINTS, 6))
        rel = quest_fk_all(tree, pq.astype(np.float32), np.zeros((N, 3), np.float32))
        kh = head[window - 1: window - 1 + N, :3, 3]
        P = rel + (kh - rel[:, QUEST_HEAD_IDX, :])[:, None, :]
        P[..., 2] -= float(P[:30, :, 2].min()) - 0.0871
        out[c] = P
    del net
    torch.cuda.empty_cache()
    return out


def load_stream(stream, clips, tree):
    """25-joint world positions per clip for one stream, in the shared native Z-up frame."""
    stage, src = STREAMS[stream]
    if src == "gt":
        return {c: gt_positions(c, tree) for c in clips}
    if src.startswith("backbone:"):
        return backbone_positions(src.split(":", 2)[1], src.split(":", 2)[2], clips, tree)
    d = joblib.load(src)
    keys = [str(k) for k in d["motion_keys"]]
    idx = {}
    for i, k in enumerate(keys):
        for c in clips:
            if c in k:
                idx[c] = i
    out = {c: np.asarray(d["pred_pos_all"][i], dtype=np.float64) for c, i in idx.items()}
    del d
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stream", required=True, choices=list(STREAMS))
    ap.add_argument("--clips", nargs="*", default=None, help="default: all test clips")
    ap.add_argument("--chunk", type=int, default=64, help="clips per batched IK call")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--iters-warm", type=int, default=600)
    ap.add_argument("--validate", action="store_true",
                    help="run only the 3 validated clips and compare against the stored values")
    args = ap.parse_args()

    os.makedirs(OUTDIR, exist_ok=True)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    corr = json.load(open(CORR))
    soff = M.load_smpl_rest_offsets(corr)
    tree = SkeletonTree.from_mjcf(M.MJCF)
    bm = BodyModel(bm_fname=BM_NEUTRAL, num_betas=16, num_dmpls=8,
                   dmpl_fname=DMPL_NEUTRAL).to(dev)

    if args.validate:
        clips = ["SUBJECT072.gestures.pkl_427_853", "SUBJECT086.walking.pkl_0_412",
                 "SUBJECT075.ROM.pkl_3820_4243"]
    elif args.clips:
        clips = args.clips
    else:
        d = os.path.join(REPO, "data/quest2_synthetic/test_seg_motion")
        clips = sorted(f[:-4] for f in os.listdir(d) if f.endswith(".pkl"))
    if args.limit:
        clips = clips[:args.limit]

    print(f"stream={args.stream} stage={STREAMS[args.stream][0]} clips={len(clips)} "
          f"device={dev} chunk={args.chunk} iters_warm={args.iters_warm}", flush=True)
    pos = load_stream(args.stream, clips, tree)
    clips = [c for c in clips if c in pos and pos[c] is not None and len(pos[c]) > 4]
    print(f"  loaded {len(clips)} clips", flush=True)

    results, t0 = {}, time.time()
    for s in range(0, len(clips), args.chunk):
        sub = clips[s:s + args.chunk]
        print(f"  chunk {s//args.chunk+1}/{(len(clips)+args.chunk-1)//args.chunk} "
              f"({len(sub)} clips)", flush=True)
        fits = batched_ik([pos[c] for c in sub], soff, corr, dev,
                          iters_warm=args.iters_warm, log=True)
        for c, p in zip(sub, fits):
            mean_mm, max_mm, rate = mesh_gpen_from_params(p, bm, dev)
            results[c] = dict(frames=int(len(p["root"])), ik_all_mean_mm=float(p["err"].mean()),
                              gpen_mean_mm=mean_mm, gpen_rate=rate, gpen_max_mm=max_mm)
        json.dump(results, open(os.path.join(OUTDIR, f"{args.stream}.json"), "w"), indent=2)
        print(f"    chunk done, {time.time()-t0:.0f}s total", flush=True)

    el = time.time() - t0
    print(f"\n[{args.stream}] {len(results)} clips in {el/60:.1f} min "
          f"({el/max(len(results),1):.1f}s/clip)")
    if args.validate:
        ref = json.load(open(os.path.join(REPO, "scripts/mesh_gpen_fourway.json")))
        key = {"gt": "gt", "simxr": "simxr", "finetuned_backbone": "finetuned_backbone",
               "naive": "naive"}.get(args.stream)
        print("\nVALIDATION vs the sequential per-frame pipeline:")
        print(f"  {'clip':34s}{'IK seq':>9s}{'IK bat':>9s}{'gpen seq':>10s}{'gpen bat':>10s}"
              f"{'rate seq':>10s}{'rate bat':>10s}")
        ok = True
        for c in clips:
            r0 = ref.get(c, {}).get(key)
            r1 = results[c]
            if not r0:
                continue
            print(f"  {c[:34]:34s}{r0['ik_all_mean_mm']:9.1f}{r1['ik_all_mean_mm']:9.1f}"
                  f"{r0['gpen_mean_mm']:10.2f}{r1['gpen_mean_mm']:10.2f}"
                  f"{r0['gpen_rate']:10.3f}{r1['gpen_rate']:10.3f}")
            if abs(r0["ik_all_mean_mm"] - r1["ik_all_mean_mm"]) > 2.0: ok = False
            if abs(r0["gpen_mean_mm"] - r1["gpen_mean_mm"]) > 2.0: ok = False
            if abs(r0["gpen_rate"] - r1["gpen_rate"]) > 0.03: ok = False
        print("  ->", "PASS (batched reproduces sequential)" if ok else "FAIL (divergence)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
