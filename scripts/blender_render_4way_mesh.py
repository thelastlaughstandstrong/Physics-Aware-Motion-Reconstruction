"""Four-way SMPL-X MESH render: Ground Truth | SimXR | Finetuned Backbone | Naive PHC Baseline.

Full-body mesh version of blender_render_4way.py. Each column is driven by the VALIDATED Step-5
quest->SMPL retarget output (smpl_trans / smpl_root_orient_aa / smpl_body_pose_aa) for that
stream's own rollout -- so every column keeps its own motion; nothing is re-simulated.

Reuses blender_render_addon.build_rig / animate_rig UNCHANGED (PHC Z-up direct, per-bone
rest-relative correction, pelvis corrected against its absolute rest) and
blender_render_comparison.build_ground / flat_material / add_label.

Usage:
    blender --background --python scripts/blender_render_4way_mesh.py -- \
        --retarget <dir> --tag <tag> --outdir <dir> [--stills 3] [--samples 16]
"""
import argparse
import json
import math
import os
import sys

import bpy
import numpy as np
from mathutils import Vector

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import blender_render_comparison as BRC

STREAMS = ["gt", "simxr", "fb", "naive"]
BRC.COL_X = {"gt": -1.95, "simxr": -0.65, "fb": 0.65, "naive": 1.95}
BRC.COLORS = {"quest": (0.15, 0.65, 0.25, 1.0),
              "gt":    (0.15, 0.65, 0.25, 1.0),
              "simxr": (0.85, 0.62, 0.12, 1.0),
              "fb":    (0.80, 0.18, 0.18, 1.0),
              "naive": (0.16, 0.42, 0.72, 1.0)}
BRC.LABELS = {"gt": "Ground Truth", "simxr": "SimXR",
              "fb": "Finetuned Backbone", "naive": "Naive PHC Baseline"}
DIST, HEIGHT, PITCH = 5.2, 0.95, 88.0
BRC.LABEL_DEPTH = 2.0


def get_args():
    argv = sys.argv
    argv = argv[argv.index("--") + 1:] if "--" in argv else []
    p = argparse.ArgumentParser()
    p.add_argument("--retarget", required=True)
    p.add_argument("--tag", required=True)
    p.add_argument("--outdir", required=True)
    p.add_argument("--stills", type=int, default=3)
    p.add_argument("--samples", type=int, default=16)
    p.add_argument("--res", type=int, nargs=2, default=[2560, 900])
    p.add_argument("--texture", default="smplx_texture_m_alb.png")
    p.add_argument("--stills_only", action="store_true")
    return p.parse_args(argv)


def aa_to_quat_xyzw(aa):
    """Axis-angle -> xyzw quaternion, numpy only (Blender's bundled Python has no scipy).
    Uses the sinc-safe half-angle form so theta -> 0 stays finite and exact to first order."""
    aa = np.asarray(aa, dtype=np.float64)
    theta = np.linalg.norm(aa, axis=-1, keepdims=True)
    half = 0.5 * theta
    # sin(half)/theta, guarded at theta=0 where the limit is 1/2
    small = theta < 1e-8
    k = np.where(small, 0.5 - theta ** 2 / 48.0, np.sin(half) / np.where(small, 1.0, theta))
    xyz = aa * k
    w = np.cos(half)
    return np.concatenate([xyz, w], axis=-1)


def load_stream(path):
    """Retarget JSON -> (positions[T,24,3] with pelvis in slot 0, rot_local[T,24,4] xyzw)."""
    d = json.load(open(path))
    trans = np.asarray(d["smpl_trans"], dtype=np.float64)              # (T,3) pelvis world
    root = np.asarray(d["smpl_root_orient_aa"], dtype=np.float64)      # (T,3) global root
    body = np.asarray(d["smpl_body_pose_aa"], dtype=np.float64)        # (T,23,3) local
    T = len(trans)
    aa = np.concatenate([root[:, None, :], body], axis=1)              # (T,24,3)
    rot_local = aa_to_quat_xyzw(aa.reshape(-1, 3)).reshape(T, 24, 4)
    positions = np.zeros((T, 24, 3), dtype=np.float64)
    positions[:, 0, :] = trans          # animate_rig only reads positions[t, 0]
    return positions, rot_local, d


def main():
    args = get_args()
    try:
        bpy.ops.preferences.addon_enable(module="bl_ext.user_default.smplx_blender_addon")
    except Exception as e:                                              # noqa: BLE001
        print("[render] addon_enable:", e)
    import blender_render_addon as BRA

    data = {s: load_stream(os.path.join(args.retarget, f"{args.tag}_{s}.json")) for s in STREAMS}
    T = min(len(v[0]) for v in data.values())
    fps = 30
    print(f"[render] {args.tag}: T={T} @ {fps}fps  " +
          " ".join(f"{s}={len(data[s][0])}(fit {data[s][2]['all_mean_mm']:.1f}mm)" for s in STREAMS))

    BRA.clear_scene()
    BRC.build_ground()

    for s in STREAMS:
        pos, rot, _ = data[s]
        mesh, arm, rest, prt = BRA.build_rig(f"rig_{s}", BRC.COL_X[s], args.texture)
        BRA.animate_rig(arm, rest, prt, pos[:T], rot[:T], BRC.COL_X[s], fps)
        mesh.data.materials.clear()
        mesh.data.materials.append(BRC.flat_material(f"mat_{s}", BRC.COLORS[s], rough=0.5))

    ref = data["gt"][0][:T, 0, :2]
    k = 31
    pad = np.pad(ref, ((k // 2, k // 2), (0, 0)), mode="edge")
    smooth = np.stack([np.convolve(pad[:, i], np.ones(k) / k, mode="valid") for i in range(2)], -1)

    bpy.ops.object.camera_add()
    cam = bpy.context.active_object
    cam.data.lens = 32.0
    cam.data.clip_end = 200.0
    scene = bpy.context.scene
    scene.camera = cam
    for t in range(T):
        cam.location = Vector((float(smooth[t, 0]), float(smooth[t, 1]) - DIST, HEIGHT))
        cam.rotation_euler = (math.radians(PITCH), 0.0, 0.0)
        cam.keyframe_insert(data_path="location", frame=t + 1)
        cam.keyframe_insert(data_path="rotation_euler", frame=t + 1)
    half_w = BRC.LABEL_DEPTH * math.tan(math.atan(18.0 / cam.data.lens))
    label_y = 0.87 * half_w / (args.res[0] / args.res[1])
    for s in STREAMS:
        txt = BRC.add_label(cam, s, BRC.COL_X[s] * BRC.LABEL_DEPTH / DIST, size=0.042)
        txt.location = (txt.location.x, label_y, txt.location.z)
    print(f"[render] labels at y={label_y:.3f}")

    bpy.ops.object.light_add(type="SUN", location=(4, -6, 12))
    sun = bpy.context.active_object
    sun.data.energy = 3.2
    sun.rotation_euler = (math.radians(48), 0, math.radians(38))
    bpy.ops.object.light_add(type="AREA", location=(-5, -7, 5))
    fill = bpy.context.active_object
    fill.data.energy = 260
    fill.data.size = 8
    world = bpy.data.worlds["World"]
    world.use_nodes = True
    world.node_tree.nodes["Background"].inputs["Color"].default_value = (0.05, 0.06, 0.08, 1)
    world.node_tree.nodes["Background"].inputs["Strength"].default_value = 1.0

    os.makedirs(args.outdir, exist_ok=True)
    scene.render.engine = "BLENDER_EEVEE"
    try:
        scene.eevee.taa_render_samples = args.samples
    except Exception:                                                   # noqa: BLE001
        pass
    scene.render.resolution_x, scene.render.resolution_y = args.res
    scene.render.resolution_percentage = 100
    scene.render.fps = fps
    scene.frame_start, scene.frame_end = 1, T
    scene.render.film_transparent = False

    stills = np.unique(np.linspace(1, T, args.stills + 2).astype(int))[1:-1]
    scene.render.image_settings.file_format = "PNG"
    for f in stills:
        scene.frame_set(int(f))
        scene.render.filepath = os.path.join(args.outdir, f"{args.tag}_mesh_still_f{int(f):04d}.png")
        bpy.ops.render.render(write_still=True)
        print(f"[render] still frame {int(f)}")

    if args.stills_only:
        print("[render] stills only -- skipping animation")
        return

    fdir = os.path.join(args.outdir, f"_{args.tag}_mesh_frames")
    os.makedirs(fdir, exist_ok=True)
    scene.render.filepath = os.path.join(fdir, "f_")
    bpy.ops.render.render(animation=True)
    print(f"[render] wrote {T} frames -> {fdir}")


if __name__ == "__main__":
    main()
