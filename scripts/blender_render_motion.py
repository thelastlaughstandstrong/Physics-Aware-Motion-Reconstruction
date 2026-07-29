"""Render predicted-vs-ground-truth motion (from export_blender_motion.py's npz)
as two side-by-side capsule skeletons in Blender.

PHC's world frame is already Z-up with the floor at Z=0 (see
phc_positions_for_eval.pkl's own doc comment in im_amp_players.py), which
matches Blender's default axes exactly -- no coordinate conversion needed.

Positions only (no rotations/FK): each joint is a small sphere, each bone a
cylinder re-fit every frame to connect parent->child world positions. All
transforms are baked as explicit keyframes (not runtime constraints), so the
result is fully deterministic and safe to open/scrub/render outside this script.

Usage (must run through Blender's own Python, not a conda env):
    blender --background --python scripts/blender_render_motion.py -- \\
        --npz phc/output/HumanoidIm/<exp>/blender_motion_<key>.npz \\
        [--out out.blend] [--offset 1.4] [--radius 0.025]
"""
import argparse
import math
import sys

import bpy
import mathutils
import numpy as np


def get_args():
    argv = sys.argv
    argv = argv[argv.index("--") + 1:] if "--" in argv else []
    p = argparse.ArgumentParser()
    p.add_argument("--npz", required=True)
    p.add_argument("--out", default=None, help="Output .blend path (default: alongside the npz)")
    p.add_argument("--offset", type=float, default=1.4, help="World-space separation between the pred and gt rigs")
    p.add_argument("--radius", type=float, default=0.025, help="Bone cylinder radius (metres)")
    return p.parse_args(argv)


def clear_scene():
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)


def make_material(name, color):
    mat = bpy.data.materials.get(name) or bpy.data.materials.new(name)
    # diffuse_color drives Solid-shading viewport display (Color: Material) --
    # set independently of the node-based BSDF color below, which only shows
    # up in Material Preview / Rendered shading modes.
    mat.diffuse_color = (*color, 1.0)
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes.get("Principled BSDF")
    if bsdf is not None:
        bsdf.inputs["Base Color"].default_value = (*color, 1.0)
    return mat


def build_joints(name, num_joints, radius, material):
    objs = []
    for j in range(num_joints):
        bpy.ops.mesh.primitive_uv_sphere_add(radius=radius * 1.6, location=(0, 0, 0))
        obj = bpy.context.active_object
        obj.name = f"{name}_joint_{j}"
        obj.data.materials.append(material)
        objs.append(obj)
    return objs


def build_bones(name, parents, radius, material):
    objs = {}
    for j, par in enumerate(parents):
        if par < 0:
            continue
        bpy.ops.mesh.primitive_cylinder_add(radius=radius, depth=1.0, location=(0, 0, 0))
        obj = bpy.context.active_object
        obj.name = f"{name}_bone_{par}_{j}"
        obj.rotation_mode = "QUATERNION"
        obj.data.materials.append(material)
        objs[j] = obj
    return objs


def fit_bone(obj, p0, p1):
    """Re-fit a unit cylinder (local +Z axis, depth=1 => spans [-0.5, 0.5]) to
    connect world points p0 -> p1."""
    p0, p1 = mathutils.Vector(p0), mathutils.Vector(p1)
    diff = p1 - p0
    length = diff.length
    obj.location = (p0 + p1) / 2.0
    if length < 1e-6:
        obj.scale = (1e-3, 1e-3, 1e-3)
        return
    obj.scale = (1.0, 1.0, length)
    obj.rotation_quaternion = mathutils.Vector((0, 0, 1)).rotation_difference(diff.normalized())


def animate_rig(name, positions, parents, x_offset, radius, mat_joint, mat_bone, fps):
    """positions: (T, J, 3) world-space, already Z-up."""
    T, J, _ = positions.shape
    joints = build_joints(name, J, radius, mat_joint)
    bones = build_bones(name, parents, radius, mat_bone)

    scene = bpy.context.scene
    scene.render.fps = fps
    scene.frame_start = 1
    scene.frame_end = max(scene.frame_end, T)

    shift = np.array([x_offset, 0.0, 0.0])
    for t in range(T):
        frame = t + 1
        pts = positions[t] + shift
        for j in range(J):
            joints[j].location = pts[j]
            joints[j].keyframe_insert(data_path="location", frame=frame)
        for j, par in enumerate(parents):
            if par < 0:
                continue
            fit_bone(bones[j], pts[par], pts[j])
            bones[j].keyframe_insert(data_path="location", frame=frame)
            bones[j].keyframe_insert(data_path="rotation_quaternion", frame=frame)
            bones[j].keyframe_insert(data_path="scale", frame=frame)


def main():
    args = get_args()
    d = np.load(args.npz, allow_pickle=True)
    pred, gt = d["pred_pos"], d["gt_pos"]
    parents = d["parents"].tolist()
    fps = int(d["fps"])
    motion_key = str(d["motion_key"])

    clear_scene()

    mat_pred_joint = make_material("PredJoint", (0.85, 0.25, 0.20))
    mat_pred_bone = make_material("PredBone", (0.90, 0.40, 0.30))
    mat_gt_joint = make_material("GTJoint", (0.20, 0.55, 0.85))
    mat_gt_bone = make_material("GTBone", (0.30, 0.65, 0.90))

    bpy.ops.mesh.primitive_plane_add(size=6, location=(0, 0, 0))
    bpy.context.active_object.name = "Ground"

    animate_rig("pred", pred, parents, -args.offset / 2, args.radius, mat_pred_joint, mat_pred_bone, fps)
    animate_rig("gt", gt, parents, args.offset / 2, args.radius, mat_gt_joint, mat_gt_bone, fps)

    bpy.ops.object.camera_add(location=(0, -4.5, 1.6), rotation=(math.radians(80), 0, 0))
    bpy.context.scene.camera = bpy.context.active_object
    bpy.ops.object.light_add(type="SUN", location=(0, -2, 5))

    # Default every 3D viewport to Material Preview and "Color: Material" in
    # Solid mode, so the file opens already showing red/blue instead of flat
    # gray -- headless --background runs still have the UI layout baked into
    # the saved file even though there's no live window to render into here.
    for screen in bpy.data.screens:
        for area in screen.areas:
            if area.type != "VIEW_3D":
                continue
            for space in area.spaces:
                if space.type == "VIEW_3D":
                    space.shading.type = "MATERIAL"
                    space.shading.color_type = "MATERIAL"

    out_path = args.out or args.npz.replace(".npz", ".blend")
    bpy.ops.wm.save_as_mainfile(filepath=out_path)
    print(f"[blender] wrote {out_path} ({pred.shape[0]} frames @ {fps}fps, seq={motion_key}) "
          f"— red=pred (v7_validate), blue=ground truth")


if __name__ == "__main__":
    main()
