"""Render predicted-vs-ground-truth motion using the official SMPL-X Blender
addon's rig/mesh/skinning (academic license, installed via
extensions.package_install_files) -- but drives it with our own pose
application instead of the addon's object.smplx_add_animation operator.

Why not use that operator directly: it expects a raw AMASS .npz and applies
poses via set_pose_from_rodrigues, which assumes the FED rotations already
match each bone's own (non-uniform) rest orientation relative to its parent.
Two problems with going through it: (1) it introduced an unexplained ~0.74m
translation offset (confirmed via isolated bpy testing -- feeding identical
location+rotation directly to a bare pelvis bone reproduces the intended
position exactly, but the full operator does not), and (2) even after
calibrating that out, the head ended up ~0.3m too low relative to pelvis --
because unlike our own custom rig (blender_render_mesh.py's "uniform bone
direction" trick, which deliberately makes every bone's rest orientation
identity-relative-to-parent), THIS rig's bones point along their natural
T-pose limb directions, so applying our SMPL-convention local rotations
directly is wrong for every non-identity-rest bone, not just the root.

Correct approach: for each bone, apply the per-bone rest-relative-to-parent
correction pose_quat[i] = bone_rest_relative[i].inverted() @ local_smpl[i].
This works for every bone -- our custom rig only needed this for the root
specifically because its uniform-direction construction made every OTHER
bone's rest-relative-to-parent identity by construction; this rig has no such
property, so every bone needs it. Confirmed empirically (not just assumed):
the rig's own rest pose already sits correctly in Blender's native Z-up
world (armature.matrix_world is identity, head sits above pelvis via the
ordinary Z axis) -- so unlike the abandoned AMASS-npz route, no Y-up
conversion is applied here at all; PHC's Z-up data is used directly, exactly
like blender_render_mesh.py already does.

Bone hierarchy note: this rig has an extra "root" bone above "pelvis"
(root.parent=None, pelvis.parent=root) that SMPL's own convention doesn't
have -- handled the same way as every other bone (rest-relative correction),
not specially.

Usage (must run through Blender's own Python, not a conda env):
    blender --background --python scripts/blender_render_addon.py -- \\
        --npz phc/output/HumanoidIm/<exp>/blender_mesh_<key>.npz \\
        [--out out.blend] [--offset 1.4] [--texture smplx_texture_m_alb.png]
"""
import argparse
import math
import sys

import bpy
import numpy as np
from mathutils import Quaternion, Vector

SMPL_TO_SMPLX_BONE = {
    0: "pelvis", 1: "left_hip", 2: "right_hip", 3: "spine1", 4: "left_knee",
    5: "right_knee", 6: "spine2", 7: "left_ankle", 8: "right_ankle", 9: "spine3",
    10: "left_foot", 11: "right_foot", 12: "neck", 13: "left_collar", 14: "right_collar",
    15: "head", 16: "left_shoulder", 17: "right_shoulder", 18: "left_elbow",
    19: "right_elbow", 20: "left_wrist", 21: "right_wrist",
}


def get_args():
    argv = sys.argv
    argv = argv[argv.index("--") + 1:] if "--" in argv else []
    p = argparse.ArgumentParser()
    p.add_argument("--npz", required=True)
    p.add_argument("--out", default=None)
    p.add_argument("--offset", type=float, default=1.4)
    p.add_argument("--texture", default="smplx_texture_m_alb.png")
    return p.parse_args(argv)


def clear_scene():
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)


def np_quat_xyzw_to_blender(q):
    return Quaternion((float(q[3]), float(q[0]), float(q[1]), float(q[2])))


def build_rig(label, x_offset, texture):
    wm = bpy.context.window_manager
    wm.smplx_tool.body_model = "smplx"
    wm.smplx_tool.smplx_gender = "neutral"
    bpy.ops.scene.smplx_add_gender()

    mesh_obj = bpy.context.view_layer.objects.active
    armature = mesh_obj.parent
    armature.name = label
    for pbone in armature.pose.bones:
        pbone.rotation_mode = "QUATERNION"

    # Per-bone rest-relative-to-parent rotation, in ARMATURE space (matrix_local
    # is relative to the armature, not directly parent-relative -- compose
    # with the parent's inverse to get the true parent-relative rest rotation).
    rest_relative = {}
    pelvis_rest_translation = None
    for smpl_idx, name in SMPL_TO_SMPLX_BONE.items():
        bone = armature.data.bones[name]
        parent = bone.parent
        if parent is None:
            rest_relative[smpl_idx] = bone.matrix_local.to_quaternion()
        else:
            rest_relative[smpl_idx] = (parent.matrix_local.inverted() @ bone.matrix_local).to_quaternion()
        if name == "pelvis":
            pelvis_rest_translation = bone.matrix_local.translation.copy()

    bpy.context.window_manager.smplx_tool.smplx_texture = texture
    bpy.ops.object.smplx_set_texture()

    armature.location.x = x_offset
    return mesh_obj, armature, rest_relative, pelvis_rest_translation


def animate_rig(armature, rest_relative, pelvis_rest_translation, positions, rot_local, x_offset, fps):
    """positions: (T, 24, 3) PHC Z-up world, SMPL_BONE_ORDER_NAMES order.
    rot_local:  (T, 24, 4) xyzw, local/parent-relative (root = raw global),
    same SMPL convention already validated in export_blender_mesh_motion.py."""
    T = positions.shape[0]
    scene = bpy.context.scene
    scene.render.fps = fps
    scene.frame_start = 1
    scene.frame_end = max(scene.frame_end, T)

    rest_inv = {i: q.inverted() for i, q in rest_relative.items()}
    pbones = armature.pose.bones

    for t in range(T):
        frame = t + 1
        # Cancel this rig's own (small, fixed) pelvis rest offset so
        # armature.location alone determines the pelvis's exact world
        # position, matching positions[t, 0] directly -- same principle as
        # blender_render_mesh.py's custom rig, just applied post-hoc here
        # since this rig's rest geometry isn't ours to construct.
        pelvis_world = Vector((float(positions[t, 0, 0]), float(positions[t, 0, 1]), float(positions[t, 0, 2])))
        armature.location = pelvis_world - pelvis_rest_translation + Vector((x_offset, 0, 0))
        armature.keyframe_insert(data_path="location", frame=frame)

        for smpl_idx, bone_name in SMPL_TO_SMPLX_BONE.items():
            local_smpl = np_quat_xyzw_to_blender(rot_local[t, smpl_idx])
            corrected = rest_inv[smpl_idx] @ local_smpl
            pbones[bone_name].rotation_quaternion = corrected
            pbones[bone_name].keyframe_insert(data_path="rotation_quaternion", frame=frame)


def verify_frame1(armature, label):
    bpy.context.scene.frame_set(1)
    depsgraph = bpy.context.evaluated_depsgraph_get()
    arm_eval = armature.evaluated_get(depsgraph)
    for bone_name in ("pelvis", "head"):
        pbone = arm_eval.pose.bones.get(bone_name)
        if pbone is None:
            continue
        world_pos = arm_eval.matrix_world @ pbone.matrix.translation
        print(f"[verify] {label} frame1 {bone_name} world = {world_pos}")


def main():
    args = get_args()
    d = np.load(args.npz, allow_pickle=True)
    pred_pos, gt_pos = d["pred_pos"], d["gt_pos"]
    pred_rot_local, gt_rot_local = d["pred_rot_local"], d["gt_rot_local"]
    fps = int(d["fps"])
    motion_key = str(d["motion_key"])

    clear_scene()
    bpy.ops.mesh.primitive_plane_add(size=6, location=(0, 0, 0))
    bpy.context.active_object.name = "Ground"

    pred_mesh, pred_arm, pred_rest, pred_pelvis_rest_t = build_rig("pred_v7_validate", -args.offset / 2, args.texture)
    gt_mesh, gt_arm, gt_rest, gt_pelvis_rest_t = build_rig("gt_ground_truth", args.offset / 2, args.texture)

    animate_rig(pred_arm, pred_rest, pred_pelvis_rest_t, pred_pos, pred_rot_local, -args.offset / 2, fps)
    animate_rig(gt_arm, gt_rest, gt_pelvis_rest_t, gt_pos, gt_rot_local, args.offset / 2, fps)

    verify_frame1(pred_arm, "pred")
    verify_frame1(gt_arm, "gt")

    bpy.ops.object.camera_add(location=(0, -4.5, 1.6), rotation=(math.radians(80), 0, 0))
    bpy.context.scene.camera = bpy.context.active_object
    bpy.ops.object.light_add(type="SUN", location=(0, -2, 5))

    for screen in bpy.data.screens:
        for area in screen.areas:
            if area.type != "VIEW_3D":
                continue
            for space in area.spaces:
                if space.type == "VIEW_3D":
                    space.shading.type = "MATERIAL"

    out_path = args.out or args.npz.replace(".npz", "_addon.blend").replace("blender_mesh_", "blender_addon_")
    bpy.ops.wm.save_as_mainfile(filepath=out_path)
    print(f"[blender] wrote {out_path} ({pred_pos.shape[0]} frames @ {fps}fps, seq={motion_key}) "
          f"— left=pred (v7_validate), right=ground truth")


if __name__ == "__main__":
    main()
