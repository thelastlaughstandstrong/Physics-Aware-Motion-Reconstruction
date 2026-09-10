"""Four-way presentation render: Ground Truth | SimXR | Finetuned Backbone | Naive PHC Baseline.

Extends blender_render_comparison.py from 3 columns to 4. All four streams are quest_humanoid
(25-joint) rollouts, so ALL FOUR use the same validated marker/skeleton representation that
Step 7 used for its Quest GT column -- there is no SMPL mesh for this rig, and retargeting each
rollout to SMPL would add unvalidated conversion error to a method comparison. Reuses
blender_render_comparison's build_ground / build_quest_markers / flat_material / add_label
unchanged; only layout, colours, labels and camera distance differ.

Keyframes stay LINEAR (inherited from build_quest_markers) so no easing smooths the motion.

Usage:
    blender --background --python scripts/blender_render_4way.py -- \
        --dir <export_dir> --tag <tag> --outdir <dir> [--stills 3] [--samples 16]
"""
import argparse
import math
import os
import sys

import bpy
import numpy as np
from mathutils import Vector

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import blender_render_comparison as BRC

STREAMS = ["gt", "simxr", "fb", "naive"]
BRC.COL_X = {"gt": -3.15, "simxr": -1.05, "fb": 1.05, "naive": 3.15}
BRC.COLORS = {"quest": (0.15, 0.65, 0.25, 1.0),   # build_quest_markers's own initial material
              "gt":    (0.15, 0.65, 0.25, 1.0),
              "simxr": (0.85, 0.62, 0.12, 1.0),
              "fb":    (0.80, 0.18, 0.18, 1.0),
              "naive": (0.16, 0.42, 0.72, 1.0)}
BRC.LABELS = {"gt": "Ground Truth", "simxr": "SimXR",
              "fb": "Finetuned Backbone", "naive": "Naive PHC Baseline"}
DIST, HEIGHT, PITCH = 7.6, 1.45, 87.0
BRC.LABEL_DEPTH = 2.0


def get_args():
    argv = sys.argv
    argv = argv[argv.index("--") + 1:] if "--" in argv else []
    p = argparse.ArgumentParser()
    p.add_argument("--dir", required=True)
    p.add_argument("--tag", required=True)
    p.add_argument("--outdir", required=True)
    p.add_argument("--stills", type=int, default=3)
    p.add_argument("--samples", type=int, default=16)
    p.add_argument("--res", type=int, nargs=2, default=[2560, 720])
    p.add_argument("--stills_only", action="store_true")
    return p.parse_args(argv)


def main():
    args = get_args()
    import blender_render_addon as BRA

    data = {s: np.load(os.path.join(args.dir, f"{args.tag}_{s}.npz"), allow_pickle=True)
            for s in STREAMS}
    fps = int(data["gt"]["fps"])
    T = min(int(d["frames"]) for d in data.values())
    print(f"[render] {args.tag}: T={T} @ {fps}fps  " +
          " ".join(f"{s}={int(data[s]['frames'])}" for s in STREAMS))

    BRA.clear_scene()
    BRC.build_ground()

    for s in STREAMS:
        obj = BRC.build_quest_markers(data[s]["positions_zup"][:T], data[s]["parents"],
                                      BRC.COL_X[s], fps)
        obj.data.materials.clear()
        obj.data.materials.append(BRC.flat_material(f"mat_{s}", BRC.COLORS[s], rough=0.4))

    # camera follows the GROUND-TRUTH root XY (one shared reference for every column), smoothed
    ref = data["gt"]["positions_zup"][:T, 0, :2].astype(np.float64)
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
    label_y = 0.80 * half_w / (args.res[0] / args.res[1])
    for s in STREAMS:
        txt = BRC.add_label(cam, s, BRC.COL_X[s] * BRC.LABEL_DEPTH / DIST, size=0.042)
        txt.location = (txt.location.x, label_y, txt.location.z)
    print(f"[render] labels at y={label_y:.3f} (half_width {half_w:.3f}, aspect {args.res[0]/args.res[1]:.2f})")

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
        scene.render.filepath = os.path.join(args.outdir, f"{args.tag}_blender_still_f{int(f):04d}.png")
        bpy.ops.render.render(write_still=True)
        print(f"[render] still frame {int(f)}")

    if args.stills_only:
        print("[render] stills only -- skipping animation")
        return

    fdir = os.path.join(args.outdir, f"_{args.tag}_frames")
    os.makedirs(fdir, exist_ok=True)
    scene.render.filepath = os.path.join(fdir, "f_")
    scene.render.image_settings.file_format = "PNG"
    bpy.ops.render.render(animation=True)
    print(f"[render] wrote {T} frames -> {fdir}")


if __name__ == "__main__":
    main()
