"""
STEP 6C of the Quest->SMPL retargeting work: convert the validated Step 5 SMPL sequences into the
EXISTING PHC SMPL motion format, then round-trip them back through PHC's own loader.

Format traced from the repository (not invented here):
  * schema        sample_data/avatarposer_all.pkl entries
  * construction  phc/scripts/data_process/convert_amass_isaac.py
  * consumption   phc/phc/utils/motion_lib_smpl.py::load_motion_with_skeleton

What that path actually requires:
  pose_aa            (T,72) f64  local axis-angle, **SMPL** joint order, last 6 values (L_Hand,
                                 R_Hand) zeroed -- AMASS's own convention. Both are LEAVES, so
                                 zeroing them moves no joint position (asserted below).
  pose_quat          (T,24,4) f32  LOCAL rotations, **MUJOCO** joint order, xyzw, post-upright
  pose_quat_global   (T,24,4) f64  GLOBAL rotations, **MUJOCO** joint order, xyzw, post-upright
                                 -- this is the field the loader actually reads (is_local=False)
  trans_orig         (T,3) f32   SMPL's `trans` PARAMETER
  root_trans_offset  (T,3) torch  the pelvis JOINT world position, == trans_orig +
                                 skeleton_tree.local_translation[0]; the loader calls .clone()
                                 on it, so it must stay a torch tensor. Step 5's smpl_trans is
                                 already a pelvis position, so it maps here directly.
  beta (16,) f64, gender str, fps float

Conventions, all verified against the repo rather than assumed:
  joint order     SMPL for pose_aa, MUJOCO for both quaternion fields (smpl_humanoid.xml's
                  node_names are exactly the mujoco list)
  quaternions     xyzw (scipy's convention, and poselib's)
  root trans      metres, native Z-up -- convert_amass_isaac applies NO world rotation to its
                  input, so the output world frame IS the input world frame (Step 4b's native)
  upright_start   True for smpl_humanoid (phc/data/cfg/robot/smpl_humanoid.yaml), so global
                  rotations are post-multiplied by inv(quat[0.5,0.5,0.5,0.5])
  velocities      NOT stored. PHC derives them in SkeletonMotion.from_skeleton_state /
                  compute_motion_dof_vels at the file's own fps. Nothing is copied from Quest.

Does NOT rerun IK, modify the Step 5 retargeting, or launch PHC/Isaac Gym.

Run:
    LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH python scripts/convert_step5_to_phc.py
"""
import json
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "scripts"))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "phc"))
sys.path.insert(0, os.path.join(REPO, "phc", "phc"))

import isaacgym  # noqa: F401,E402  -- must precede torch
import joblib  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from scipy.spatial.transform import Rotation as sRot  # noqa: E402
from poselib.poselib.skeleton.skeleton3d import (  # noqa: E402
    SkeletonMotion, SkeletonState, SkeletonTree)

SCRATCH = os.environ.get("CLAUDE_SCRATCH", "/tmp/claude-1000/-home-sia-irl-AvatarPoser/"
                         "198ae13f-92be-4147-9256-2b72f67cdfbf/scratchpad")
# The runtime skeleton is NOT the static mjcf/smpl_humanoid.xml: humanoid.py regenerates the
# humanoid per-beta via LocalRobot with upright_start=True (cfg/robot/smpl_humanoid.yaml), and
# that tree's rest offsets are R_up @ smpl_offsets, not smpl_offsets. Loading a known-good PHC
# motion under the static tree reconstructs a body lying along -Y; under the generated upright
# tree it stands correctly in Z. So the upright tree is generated here exactly as
# convert_amass_isaac.py does, and written to scratch rather than over a repo asset.
UPRIGHT_MJCF = os.path.join(SCRATCH, "smpl_humanoid_upright.xml")
FALLBACK_MJCF = os.path.join(REPO, "phc/phc/data/assets/mjcf/smpl_humanoid_1.xml")
CORR_JSON = os.path.join(REPO, "scripts/quest_smpl_correspondence.json")
REF_PKL = os.path.join(REPO, "sample_data/avatarposer_all.pkl")
OUT_DIR = os.path.join(REPO, "data/quest_smpl_retarget")
OUT_MD = os.path.join(REPO, "scripts/quest_smpl_phc_conversion.md")
OUT_JSON = os.path.join(REPO, "scripts/quest_smpl_phc_conversion.json")
SEQS = [("standing", os.path.join(REPO, "scripts/quest_smpl_sequence_fit.json")),
        ("walking", os.path.join(REPO, "scripts/quest_smpl_sequence_fit_walking.json"))]

MUJOCO_NAMES = ['Pelvis', 'L_Hip', 'L_Knee', 'L_Ankle', 'L_Toe', 'R_Hip', 'R_Knee', 'R_Ankle',
                'R_Toe', 'Torso', 'Spine', 'Chest', 'Neck', 'Head', 'L_Thorax', 'L_Shoulder',
                'L_Elbow', 'L_Wrist', 'L_Hand', 'R_Thorax', 'R_Shoulder', 'R_Elbow', 'R_Wrist',
                'R_Hand']
# The rest-frame correction convert_amass_isaac.py applies when upright_start is True.
Q_UPRIGHT = sRot.from_quat([0.5, 0.5, 0.5, 0.5])
PELVIS, L_HIP, R_HIP, HEAD, L_WRIST, R_WRIST, L_HAND, R_HAND = 0, 1, 2, 15, 20, 21, 22, 23

FAILS, LINES = [], []


def PR(s=""):
    print(s)
    LINES.append(s)


def check(name, ok, detail=""):
    PR(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  — {detail}" if detail else ""))
    if not ok:
        FAILS.append(name)


def build_upright_tree():
    """Regenerate the runtime humanoid exactly as convert_amass_isaac.py / humanoid.py do:
    LocalRobot with upright_start=True on neutral betas. Falls back to the checked-in
    smpl_humanoid_1.xml (verified bit-identical) if the SMPL body models aren't available."""
    try:
        from smpl_sim.smpllib.smpl_local_robot import SMPL_Robot as LocalRobot
        cwd = os.getcwd()
        os.chdir(os.path.join(REPO, "phc"))
        try:
            robot = LocalRobot({"mesh": False, "model": "smpl", "upright_start": True,
                                "body_params": {}, "joint_params": {}, "geom_params": {},
                                "actuator_params": {}}, data_dir="data/smpl")
            robot.load_from_skeleton(betas=torch.zeros(1, 16), gender=[0], objs_info=None)
            os.makedirs(os.path.dirname(UPRIGHT_MJCF), exist_ok=True)
            robot.write_xml(UPRIGHT_MJCF)
        finally:
            os.chdir(cwd)
        return SkeletonTree.from_mjcf(UPRIGHT_MJCF), UPRIGHT_MJCF
    except Exception as exc:                                          # noqa: BLE001
        print(f"  (LocalRobot unavailable: {exc}; using checked-in upright asset)")
        return SkeletonTree.from_mjcf(FALLBACK_MJCF), FALLBACK_MJCF


def poselib_velocity(p, dt):
    """poselib's SkeletonMotion._compute_velocity: gradient over time, then a sigma=2 Gaussian
    along the time axis. Reproduced here so the velocity check tests PHC's actual definition
    rather than raw finite differences."""
    from scipy.ndimage import gaussian_filter1d
    return gaussian_filter1d(np.gradient(p, axis=0) / dt, 2, axis=0, mode="nearest")


def geodesic_deg(Ra, Rb):
    """Per-rotation geodesic angle between two stacks of rotation matrices, in degrees."""
    c = ((Ra.transpose(0, 2, 1) @ Rb)[:, [0, 1, 2], [0, 1, 2]].sum(-1) - 1.0) / 2.0
    return np.degrees(np.arccos(np.clip(c, -1.0, 1.0)))


def convert(tag, step5_json, tree, s2m):
    """Step 5 SMPL parameters -> PHC motion dict, following convert_amass_isaac.py exactly."""
    d = json.load(open(step5_json))
    trans = np.array(d["smpl_trans"], dtype=np.float64)
    root_aa = np.array(d["smpl_root_orient_aa"], dtype=np.float64)
    body_aa = np.array(d["smpl_body_pose_aa"], dtype=np.float64)
    T, fps = len(trans), float(d.get("fps", 30.0))

    pose_aa_full = np.concatenate([root_aa[:, None, :], body_aa], axis=1).reshape(T, 72)
    # AMASS convention: zero the final 6 values (L_Hand, R_Hand). Positionally lossless -- both
    # are leaves of the kinematic tree, so no joint's position depends on their rotation.
    pose_aa = np.concatenate([pose_aa_full[:, :66], np.zeros((T, 6))], axis=1)

    pose_aa_mj = pose_aa.reshape(T, 24, 3)[:, s2m, :].copy()
    pose_quat_local = sRot.from_rotvec(pose_aa_mj.reshape(-1, 3)).as_quat().reshape(T, 24, 4)

    # Root convention. The format's invariant (verified on sample_data/avatarposer_all.pkl) is
    #     root_trans_offset = trans_orig + local_translation[0]
    # where trans_orig is SMPL's `trans` PARAMETER and root_trans_offset is the resulting pelvis
    # JOINT world position. Step 5's smpl_trans is already the pelvis joint position -- its FK
    # zeroes soff[0] and sets pos[0] = trans -- so it maps to root_trans_offset directly, and
    # trans_orig is recovered by subtracting the offset. Adding local_translation[0] to it instead
    # displaces the whole body by 225.1 mm (= |local_translation[0]|) uniformly.
    root_trans_offset = torch.from_numpy(trans).float()
    trans_orig = trans - tree.local_translation[0].numpy().astype(np.float64)

    sk = SkeletonState.from_rotation_and_root_translation(
        tree, torch.from_numpy(pose_quat_local), root_trans_offset, is_local=True)
    # upright_start=True for smpl_humanoid: re-express global rotations in the mjcf's rest frame
    pose_quat_global = (sRot.from_quat(sk.global_rotation.reshape(-1, 4).numpy())
                        * Q_UPRIGHT.inv()).as_quat().reshape(T, 24, 4)
    sk = SkeletonState.from_rotation_and_root_translation(
        tree, torch.from_numpy(pose_quat_global), root_trans_offset, is_local=False)
    pose_quat = sk.local_rotation.numpy()

    motion = {"pose_quat_global": pose_quat_global,
              "pose_quat": pose_quat.astype(np.float32),
              "trans_orig": trans_orig.astype(np.float32),
              "root_trans_offset": root_trans_offset,
              "beta": np.zeros(16, dtype=np.float64),
              "gender": "neutral",
              "pose_aa": pose_aa,
              "fps": fps}
    key = f"quest_retarget_{tag}"
    out = os.path.join(OUT_DIR, f"quest_smpl_{tag}_phc.pkl")
    joblib.dump({key: motion}, out)
    return motion, out, key, dict(T=T, fps=fps, trans=trans, root_aa=root_aa, body_aa=body_aa,
                                 pose_aa_full=pose_aa_full)


def reference_fk(src, parents, soff):
    """The validated Step 5/6B SMPL motion: retargeter FK on real SMPL rest offsets."""
    from retarget_quest_to_smpl import SmplSkeletonFK
    fk = SmplSkeletonFK(soff, parents, torch.device("cpu"))
    return np.stack([fk(torch.tensor(src["root_aa"][t]), torch.tensor(src["body_aa"][t]),
                        torch.tensor(src["trans"][t]))[0].numpy() for t in range(src["T"])])


def validate(tag, pkl_path, key, src, tree, s2m, m2s, parents, soff):
    from phc.utils.motion_lib_base import compute_motion_dof_vels

    PR("=" * 88)
    PR(f"{tag.upper()}  round trip: PHC file -> PHC loader -> reconstructed motion")
    PR("=" * 88)

    # ---- reload exactly as motion_lib_smpl.load_motion_with_skeleton does ----
    f = joblib.load(pkl_path)[key]
    trans_l = f["root_trans_offset"].clone()                 # .clone() => must be a torch tensor
    pqg = torch.from_numpy(np.asarray(f["pose_quat_global"]))
    sk = SkeletonState.from_rotation_and_root_translation(tree, pqg, trans_l, is_local=False)
    motion = SkeletonMotion.from_skeleton_state(sk, f["fps"])
    dof_vels = compute_motion_dof_vels(motion)

    T = src["T"]
    res = {}
    dur_src = (T - 1) / src["fps"]
    PR(f"  frames {len(motion.global_translation)} (source {T}) | fps {f['fps']:g} "
       f"(source {src['fps']:g}) | duration {dur_src:.4f} s")
    check(f"{tag}: frame count preserved", len(motion.global_translation) == T)
    check(f"{tag}: fps preserved", f["fps"] == src["fps"])
    res.update(frames=int(T), fps=float(f["fps"]), duration_s=dur_src)

    # ---- inverse conversion: PHC file -> local AA in SMPL order ----
    Rg_raw = (sRot.from_quat(np.asarray(f["pose_quat_global"]).reshape(-1, 4))
              * Q_UPRIGHT).as_matrix().reshape(T, 24, 3, 3)          # undo the upright transform
    Rl_mj = np.empty_like(Rg_raw)
    Rl_mj[:, 0] = Rg_raw[:, 0]
    for i in range(1, 24):
        p = int(tree.parent_indices[i])
        Rl_mj[:, i] = Rg_raw[:, p].transpose(0, 2, 1) @ Rg_raw[:, i]
    Rl_smpl = Rl_mj[:, m2s]                                          # MUJOCO -> SMPL order
    Rl_src = sRot.from_rotvec(src["pose_aa_full"].reshape(-1, 3)).as_matrix().reshape(T, 24, 3, 3)

    # the docstring's claim that zeroing the hands is positionally lossless, asserted directly
    kept = reference_fk(src, parents, soff)
    src_zeroed = dict(src, body_aa=np.asarray(f["pose_aa"]).reshape(T, 24, 3)[:, 1:, :],
                      root_aa=np.asarray(f["pose_aa"]).reshape(T, 24, 3)[:, 0, :])
    dz = np.linalg.norm(reference_fk(src_zeroed, parents, soff) - kept, axis=-1)
    check(f"{tag}: zeroing the hand rotations moves no joint position (both are leaves)",
          dz.max() == 0.0, f"max {dz.max():.1e} m over all 24 joints")

    ang = geodesic_deg(Rl_smpl.reshape(-1, 3, 3), Rl_src.reshape(-1, 3, 3)).reshape(T, 24)
    body = ang[:, 1:22]                                              # fitted joints (hands zeroed)
    PR(f"  joint-rotation discrepancy, fitted joints 1..21: mean {body.mean():.3e}° "
       f"max {body.max():.3e}°")
    PR(f"  root-orientation discrepancy: mean {ang[:,0].mean():.3e}° max {ang[:,0].max():.3e}°")
    PR(f"  hands (deliberately zeroed by AMASS convention): max {ang[:,[22,23]].max():.3f}°")
    check(f"{tag}: local body rotations survive the round trip", body.max() < 1e-3,
          f"max {body.max():.2e}°")
    check(f"{tag}: root orientation survives the round trip", ang[:, 0].max() < 1e-3,
          f"max {ang[:,0].max():.2e}°")
    res.update(rot_mean_deg=float(body.mean()), rot_max_deg=float(body.max()),
               root_rot_max_deg=float(ang[:, 0].max()),
               hand_rot_zeroed_max_deg=float(ang[:, [22, 23]].max()))

    # ---- root position ----
    dr = np.linalg.norm(trans_l.numpy().astype(np.float64) - src["trans"], axis=-1)
    inv = np.abs((np.asarray(f["trans_orig"], dtype=np.float64)
                  + tree.local_translation[0].numpy()) - trans_l.numpy()).max()
    PR(f"  root-position discrepancy: mean {dr.mean()*1e3:.3e} mm max {dr.max()*1e3:.3e} mm")
    check(f"{tag}: root translation survives the round trip (metres, native Z-up)",
          dr.max() < 1e-5, f"max {dr.max()*1e3:.2e} mm")
    check(f"{tag}: format invariant root_trans_offset == trans_orig + local_translation[0]",
          inv < 1e-6, f"max {inv:.2e} m")
    res.update(root_pos_max_mm=float(dr.max() * 1e3))

    # ---- joint positions: PHC's reconstructed motion vs the validated Step 5/6B motion ----
    P_ref = reference_fk(src, parents, soff)                         # SMPL order, real offsets
    P_phc = motion.global_translation.numpy().astype(np.float64)[:, m2s]   # -> SMPL order
    e = np.linalg.norm(P_phc - P_ref, axis=-1)
    non_hand = np.setdiff1d(np.arange(24), [L_HAND, R_HAND])
    PR(f"  joint-position discrepancy, all 24: mean {e.mean()*1e3:.4f} mm max {e.max()*1e3:.4f} mm")
    PR(f"    excluding the two hand leaves: mean {e[:,non_hand].mean()*1e3:.4f} mm "
       f"max {e[:,non_hand].max()*1e3:.4f} mm")
    PR(f"    L_Hand {e[:,L_HAND].mean()*1e3:.2f} mm | R_Hand {e[:,R_HAND].mean()*1e3:.2f} mm "
       f"— PHC's humanoid carries its own hand bone (40.8 / 38.0 mm from real SMPL's); both are "
       f"leaves, so this moves no other joint")
    check(f"{tag}: joint positions preserved (excluding hand leaves)",
          e[:, non_hand].max() < 0.5e-3, f"max {e[:,non_hand].max()*1e3:.4f} mm")
    check(f"{tag}: hand offset is exactly the known skeleton difference, not a conversion error",
          abs(e[:, L_HAND].mean() - 0.0408) < 2e-3 and abs(e[:, R_HAND].mean() - 0.0380) < 2e-3,
          f"L {e[:,L_HAND].mean()*1e3:.2f} mm vs 40.8, R {e[:,R_HAND].mean()*1e3:.2f} mm vs 38.0")
    res.update(pos_mean_mm=float(e.mean() * 1e3), pos_max_mm=float(e.max() * 1e3),
               pos_mean_mm_no_hands=float(e[:, non_hand].mean() * 1e3),
               pos_max_mm_no_hands=float(e[:, non_hand].max() * 1e3),
               l_hand_mm=float(e[:, L_HAND].mean() * 1e3),
               r_hand_mm=float(e[:, R_HAND].mean() * 1e3))

    # ---- velocities: PHC-derived, never copied from Quest ----
    v = motion.global_velocity.numpy()
    w = motion.global_angular_velocity.numpy()
    vr = np.abs(v - poselib_velocity(motion.global_translation.numpy(), 1.0 / f["fps"])).max()
    PR(f"  |linear velocity| max {np.linalg.norm(v,axis=-1).max():.3f} m/s | "
       f"|angular velocity| max {np.linalg.norm(w,axis=-1).max():.3f} rad/s | "
       f"dof_vels {tuple(dof_vels.shape)}")
    check(f"{tag}: linear velocity is PHC's own derivative of the reconstructed positions "
          f"at this file's fps", vr < 1e-6, f"max residual {vr:.2e} m/s")
    check(f"{tag}: velocities are physically plausible (not Quest-frame leftovers)",
          np.linalg.norm(v, axis=-1).max() < 20 and np.linalg.norm(w, axis=-1).max() < 60)
    check(f"{tag}: velocities/dof_vels finite",
          np.isfinite(v).all() and np.isfinite(w).all() and torch.isfinite(dof_vels).all())
    res.update(max_lin_vel=float(np.linalg.norm(v, axis=-1).max()),
               max_ang_vel=float(np.linalg.norm(w, axis=-1).max()),
               dof_vels_shape=list(dof_vels.shape))

    # ---- geometry + continuity on the reconstructed motion ----
    check(f"{tag}: no NaN/Inf anywhere in the reconstructed motion",
          np.isfinite(P_phc).all() and np.isfinite(np.asarray(f["pose_quat_global"])).all()
          and np.isfinite(np.asarray(f["pose_aa"])).all())

    up = P_phc[:, HEAD, 2] - P_phc[:, PELVIS, 2]
    PR(f"  head-above-pelvis in Z: min {up.min():.3f} m | Z extent {P_phc[...,2].ptp():.3f} m")
    check(f"{tag}: native Z-up preserved — head above pelvis every frame", up.min() > 0.2,
          f"min {up.min():.3f} m")

    lat = P_phc[:, L_HIP] - P_phc[:, R_HIP]
    lat /= np.linalg.norm(lat, axis=-1, keepdims=True)
    s = np.einsum("tjc,tc->tj", P_phc - P_phc[:, PELVIS:PELVIS + 1], lat)
    hip_ok = (s[:, L_HIP] - s[:, R_HIP] > 0).all()
    sho_ok = (s[:, 16] - s[:, 17] > 0).all()
    hand_ok = (s[:, L_HAND] - s[:, R_HAND]).mean() > 0
    PR(f"  L/R hand lateral separation: mean {(s[:,L_HAND]-s[:,R_HAND]).mean()*1e3:+.1f} mm; "
       f"L_Hand-to-L_Wrist {np.linalg.norm(P_phc[:,L_HAND]-P_phc[:,L_WRIST],axis=-1).mean()*1e3:.1f} mm, "
       f"R_Hand-to-R_Wrist {np.linalg.norm(P_phc[:,R_HAND]-P_phc[:,R_WRIST],axis=-1).mean()*1e3:.1f} mm")
    check(f"{tag}: left/right ordering preserved (hips, shoulders, hands)",
          hip_ok and sho_ok and hand_ok)

    step = np.linalg.norm(np.diff(P_phc, axis=0), axis=-1).max(1)
    med = float(np.median(step))
    jumps = int((step > max(10 * med, 0.05)).sum())
    PR(f"  temporal continuity: max per-frame joint step {step.max()*1e3:.1f} mm "
       f"(median {med*1e3:.1f} mm), {jumps} discontinuities")
    check(f"{tag}: no temporal discontinuities introduced by the conversion", jumps == 0)
    res.update(head_above_pelvis_min_m=float(up.min()), max_step_mm=float(step.max() * 1e3),
               n_discontinuities=jumps)
    return res


def main():
    from smpl_sim.smpllib.smpl_joint_names import SMPL_BONE_ORDER_NAMES as jn
    from retarget_quest_to_smpl import load_smpl_rest_offsets

    os.makedirs(OUT_DIR, exist_ok=True)
    corr = json.load(open(CORR_JSON))
    parents, soff = corr["smpl_parents"], load_smpl_rest_offsets(corr)
    tree, tree_path = build_upright_tree()
    s2m = [jn.index(q) for q in MUJOCO_NAMES]
    m2s = [MUJOCO_NAMES.index(q) for q in jn]

    PR("=" * 88)
    PR("0. TRACED FORMAT (from the repo, not invented)")
    PR("=" * 88)
    PR(f"  skeleton   {tree_path}")
    PR(f"             (LocalRobot, upright_start=True — the tree humanoid.py builds at runtime,")
    PR(f"              whose rest offsets are R_up @ smpl_offsets, not the static mjcf's)")
    PR(f"  node order == MUJOCO: {list(tree.node_names) == MUJOCO_NAMES}   "
       f"local_translation[0] = {np.round(tree.local_translation[0].numpy(),4)}")
    PR(f"  upright_start=True (cfg/robot/smpl_humanoid.yaml) -> post-multiply by "
       f"inv(quat[0.5,0.5,0.5,0.5]), xyzw")
    PR("  velocities: derived by PHC at load (SkeletonMotion + compute_motion_dof_vels), "
       "never stored, never copied from Quest")
    check("smpl_2_mujoco / mujoco_2_smpl are mutual inverses",
          np.array_equal(np.array(s2m)[m2s], np.arange(24)))

    out = {}
    for tag, js in SEQS:
        motion, pkl, key, src = convert(tag, js, tree, s2m)
        PR("")
        PR(f"  wrote {os.path.relpath(pkl, REPO)}  (key '{key}', {src['T']} frames)")
        out[tag] = validate(tag, pkl, key, src, tree, s2m, m2s, parents, soff)
        out[tag]["pkl"] = os.path.relpath(pkl, REPO)

    # ---- schema conformance against a known-good PHC SMPL motion ----
    PR("=" * 88)
    PR("SCHEMA CONFORMANCE vs sample_data/avatarposer_all.pkl")
    PR("=" * 88)
    ref = joblib.load(REF_PKL)
    ref_e = ref[list(ref.keys())[0]]
    mine = joblib.load(os.path.join(OUT_DIR, "quest_smpl_standing_phc.pkl"))
    mine_e = mine[list(mine.keys())[0]]
    PR(f"  {'field':<20}{'reference':<34}{'ours'}")
    for k in ref_e:
        r, m = ref_e[k], mine_e.get(k, "<MISSING>")
        fmt = lambda v: (f"{type(v).__name__} {tuple(np.asarray(v).shape)} "  # noqa: E731
                         f"{np.asarray(v).dtype}") if hasattr(v, "shape") else f"{type(v).__name__} {v}"
        PR(f"  {k:<20}{fmt(r):<34}{fmt(m)}")
    check("all reference fields present with matching types/dtypes",
          set(ref_e) == set(mine_e)
          and all(type(ref_e[k]) is type(mine_e[k]) for k in ref_e)
          and all(np.asarray(ref_e[k]).dtype == np.asarray(mine_e[k]).dtype
                  for k in ref_e if hasattr(ref_e[k], "shape")))

    PR("=" * 88)
    out["passed"] = not FAILS
    out["failed_checks"] = FAILS
    json.dump(out, open(OUT_JSON, "w"), indent=2, default=float)
    with open(OUT_MD, "w") as fh:
        fh.write("# Step 6C — Quest→SMPL sequences in PHC motion format\n\n```\n"
                 + "\n".join(LINES) + "\n```\n")
    if FAILS:
        PR(f"STEP 6C: FAIL — {len(FAILS)} check(s): {FAILS}")
        return 1
    PR("STEP 6C: PASS — PHC-format round trip reproduces the validated SMPL motion.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
