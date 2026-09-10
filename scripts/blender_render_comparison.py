"""
Step 7 -- presentation render: Quest GT | pre-physics retarget | post-PHC, side by side.

Reuses blender_render_addon.py's VALIDATED direct-rig implementation unchanged (build_rig /
animate_rig): PHC Z-up used directly, no Y-up conversion, no target_pelvis_zup_frame1, per-bone
SMPL-X rest-orientation correction, pelvis corrected against its ABSOLUTE rest (the addon rig's
extra `root` bone). This script only adds scene, camera, labels and materials.

Layout: three columns at fixed X offsets. Quest GT is a marker cloud -- it is a different 25-joint
rig, so there is no SMPL mesh for it.

Camera: follows a single shared reference (the pre-physics root XY, smoothed). It moves the CAMERA
only; no motion is altered, and every column keeps its own full trajectory including the post-PHC
root drift. The ground is a large checker so foot sliding stays visible under a moving camera.

Usage:
    blender --background --python scripts/blender_render_comparison.py -- \
        --npz mesh_<clip>.npz --quest quest_gt_<clip>.npz --tag <clip> --outdir <dir>
"""
import argparse
import math
import os
import sys

import bpy
import numpy as np
from mathutils import Vector

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

COL_X = {"quest": -2.3, "pre": 0.0, "post": 2.3}
LABEL_DEPTH = 2.0
COLORS = {"quest": (0.15, 0.65, 0.25, 1.0), "pre": (0.16, 0.42, 0.72, 1.0),
          "post": (0.80, 0.18, 0.18, 1.0)}
LABELS = {"quest": "Quest ground truth", "pre": "Pre-physics retarget", "post": "Post-PHC"}


def get_args():
    argv = sys.argv
    argv = argv[argv.index("--") + 1:] if "--" in argv else []
    p = argparse.ArgumentParser()
    p.add_argument("--npz", required=True)
    p.add_argument("--quest", required=True)
    p.add_argument("--tag", required=True)
    p.add_argument("--outdir", required=True)
    p.add_argument("--res", type=int, nargs=2, default=[1920, 1080])
    p.add_argument("--samples", type=int, default=16)
    p.add_argument("--stills", type=int, default=3)
    p.add_argument("--texture", default="smplx_texture_m_alb.png")
    return p.parse_args(argv)


def flat_material(name, rgba, rough=0.45):
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    bsdf = m.node_tree.nodes["Principled BSDF"]
    bsdf.inputs["Base Color"].default_value = rgba
    bsdf.inputs["Roughness"].default_value = rough
    return m


def build_ground():
    bpy.ops.mesh.primitive_plane_add(size=60, location=(0, 0, 0))
    g = bpy.context.active_object
    g.name = "Ground"
    m = bpy.data.materials.new("GroundChecker")
    m.use_nodes = True
    nt = m.node_tree
    bsdf = nt.nodes["Principled BSDF"]
    tex = nt.nodes.new("ShaderNodeTexChecker")
    tex.inputs["Scale"].default_value = 60.0          # ~0.5 m squares over the 60 m plane
    tex.inputs["Color1"].default_value = (0.80, 0.80, 0.82, 1)
    tex.inputs["Color2"].default_value = (0.62, 0.62, 0.66, 1)
    nt.links.new(tex.outputs["Color"], bsdf.inputs["Base Color"])
    bsdf.inputs["Roughness"].default_value = 0.9
    g.data.materials.append(m)
    return g


def build_quest_markers(positions, parents, x_offset, fps):
    """positions: (T,25,3) PHC Z-up. Quest is a different 25-joint rig, so this is a marker/
    skeleton representation, not a mesh: one vertex per joint, an edge per parent link, and a Skin
    modifier to give the bones visible thickness. Vertex coordinates are keyframed directly, which
    is far lighter than animating 25 spheres + 24 bone objects."""
    T, J, _ = positions.shape
    verts = [tuple(map(float, positions[0, j])) for j in range(J)]
    edges = [(int(parents[j]), j) for j in range(J) if int(parents[j]) >= 0]
    me = bpy.data.meshes.new("quest_skel_mesh")
    me.from_pydata(verts, edges, [])
    me.update()
    obj = bpy.data.objects.new("quest_skeleton", me)
    bpy.context.collection.objects.link(obj)
    obj.location = (x_offset, 0.0, 0.0)

    obj.modifiers.new("Skin", "SKIN")
    for v in obj.data.skin_vertices[0].data:
        v.radius = (0.028, 0.028)
    sub = obj.modifiers.new("Subsurf", "SUBSURF")
    sub.levels = sub.render_levels = 1
    obj.data.materials.append(flat_material("mat_quest", COLORS["quest"], rough=0.4))

    for t in range(T):
        for j in range(J):
            me.vertices[j].co = Vector((float(positions[t, j, 0]),
                                        float(positions[t, j, 1]),
                                        float(positions[t, j, 2])))
            me.vertices[j].keyframe_insert(data_path="co", frame=t + 1)
    # Blender 5.x moved fcurves under action layers/slots; walk whichever exists so the
    # keyframes stay LINEAR (default Bezier easing would visibly smooth the motion, which would
    # alter what the render is evidence for).
    act = me.animation_data.action if me.animation_data else None
    curves = []
    if act is not None:
        curves = list(getattr(act, "fcurves", []))
        if not curves:
            for layer in getattr(act, "layers", []):
                for strip in getattr(layer, "strips", []):
                    for cbag in getattr(strip, "channelbags", []):
                        curves.extend(cbag.fcurves)
    for fc in curves:
        for kp in fc.keyframe_points:
            kp.interpolation = "LINEAR"
    print(f"[render] quest skeleton: {len(curves)} fcurves set to LINEAR")
    return obj


def recolor(mesh_obj, key):
    mesh_obj.data.materials.clear()
    mesh_obj.data.materials.append(flat_material(f"mat_{key}", COLORS[key]))


def add_label(cam, key, x_local, size=0.062):
    bpy.ops.object.text_add()
    txt = bpy.context.active_object
    txt.data.body = LABELS[key]
    txt.data.align_x = "CENTER"
    txt.data.size = size
    txt.data.extrude = 0.0
    m = flat_material(f"lab_{key}", COLORS[key], rough=0.6)
    m.use_nodes = True
    b = m.node_tree.nodes["Principled BSDF"]
    b.inputs["Emission Color"].default_value = COLORS[key]
    b.inputs["Emission Strength"].default_value = 1.4
    txt.data.materials.append(m)
    txt.parent = cam                      # screen-locked: columns hold their screen X as cam moves
    txt.location = (x_local, 0.47, -LABEL_DEPTH)
    txt.rotation_euler = (0, 0, 0)
    return txt


def main():
    args = get_args()
    try:
        bpy.ops.preferences.addon_enable(module="bl_ext.user_default.smplx_blender_addon")
    except Exception as e:                                              # noqa: BLE001
        print("[render] addon_enable:", e)
    import blender_render_addon as BRA

    d = np.load(args.npz, allow_pickle=True)
    q = np.load(args.quest, allow_pickle=True)
    fps = int(d["fps"])
    pre_pos, post_pos = d["gt_pos"], d["pred_pos"]
    pre_rot, post_rot = d["gt_rot_local"], d["pred_rot_local"]
    qpos = q["positions_zup"]
    T = min(len(pre_pos), len(post_pos), len(qpos))
    print(f"[render] {args.tag}: T={T} @ {fps}fps "
          f"(mesh {len(pre_pos)}, quest {len(qpos)})")

    BRA.clear_scene()
    build_ground()

    pre_mesh, pre_arm, pre_rest, pre_prt = BRA.build_rig("pre_physics", COL_X["pre"], args.texture)
    BRA.animate_rig(pre_arm, pre_rest, pre_prt, pre_pos, pre_rot, COL_X["pre"], fps)
    recolor(pre_mesh, "pre")

    post_mesh, post_arm, post_rest, post_prt = BRA.build_rig("post_phc", COL_X["post"], args.texture)
    BRA.animate_rig(post_arm, post_rest, post_prt, post_pos, post_rot, COL_X["post"], fps)
    recolor(post_mesh, "post")

    build_quest_markers(qpos[:T], q["parents"], COL_X["quest"], fps)

    # ---- camera: follows one shared reference (pre-physics root XY), smoothed ----
    ref = pre_pos[:T, 0, :2].astype(np.float64)
    k = 31
    pad = np.pad(ref, ((k // 2, k // 2), (0, 0)), mode="edge")
    smooth = np.stack([np.convolve(pad[:, i], np.ones(k) / k, mode="valid") for i in range(2)], -1)

    bpy.ops.object.camera_add()
    cam = bpy.context.active_object
    cam.data.lens = 32.0
    cam.data.clip_end = 200.0
    scene = bpy.context.scene
    scene.camera = cam
    DIST, HEIGHT, PITCH = 5.9, 1.45, 87.0
    for t in range(T):
        cx = float(smooth[t, 0])
        cy = float(smooth[t, 1]) - DIST
        cam.location = Vector((cx, cy, HEIGHT))
        cam.rotation_euler = (math.radians(PITCH), 0.0, 0.0)
        cam.keyframe_insert(data_path="location", frame=t + 1)
        cam.keyframe_insert(data_path="rotation_euler", frame=t + 1)
    # a world offset X at depth DIST projects like a label at depth LABEL_DEPTH
    # placed at X * LABEL_DEPTH / DIST -- so each label sits exactly over its own column
    for key in ("quest", "pre", "post"):
        add_label(cam, key, COL_X[key] * LABEL_DEPTH / DIST)

    # ---- lighting ----
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

    # ---- output ----
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
        scene.render.filepath = os.path.join(args.outdir, f"{args.tag}_still_f{int(f):04d}.png")
        bpy.ops.render.render(write_still=True)
        print(f"[render] still frame {int(f)}")

    # This Blender build has no FFMPEG output enum, so render a PNG sequence and encode with
    # the system ffmpeg afterwards. Frame numbering is 1..T, one file per motion frame -- no
    # duplication or drops.
    seq = os.path.join(args.outdir, f"_{args.tag}_frames")
    os.makedirs(seq, exist_ok=True)
    scene.render.image_settings.file_format = "PNG"
    scene.render.filepath = os.path.join(seq, "f_")
    bpy.ops.render.render(animation=True)
    n = len([f for f in os.listdir(seq) if f.endswith(".png")])
    print(f"[render] {args.tag}: wrote {n} PNG frames to {seq} (expected {T})")


if __name__ == "__main__":
    main()
