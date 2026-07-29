"""Render predicted-vs-ground-truth motion (from export_blender_mesh_motion.py's
npz) as two skinned SMPL body meshes side by side, using a custom-built armature
+ Linear Blend Skinning rig from the raw SMPL-H model file already in this
project (support_data/body_models/smplh/neutral/model.npz) -- no external
Blender addon or licensed asset download required.

Companion to blender_render_motion.py (capsule skeletons, positions only).

Rig construction trick: every bone is built with the SAME rest direction
(local +Z, arbitrary fixed length) and roll=0, so every bone's rest
orientation *relative to its parent* is the identity transform -- meaning
PoseBone.rotation_quaternion can be set directly from our already-computed
SMPL-convention local rotations (pred_rot_local/gt_rot_local from the export
script) with no further basis conversion. Skinning deformation is driven
entirely by vertex-group weights (SMPL's own weights_prior matrix), not by
each bone's visual edit-mode direction, so this uniform-direction choice has
no effect on mesh quality -- only on how the rig looks in Edit Mode.

L_Hand/R_Hand (last 2 of the 24 joints) are left at rest/identity rotation:
PHC runs with robot.freeze_hand=True, and export_blender_mesh_motion.py's own
FK check found R_Hand's rotation data has a ~158cm-equivalent residual on this
sequence (vs. <0.1cm for every actuated joint) -- consistent with frozen,
unreliable data for those two joints specifically.

Usage (must run through Blender's own Python, not a conda env):
    blender --background --python scripts/blender_render_mesh.py -- \\
        --npz phc/output/HumanoidIm/<exp>/blender_mesh_<key>.npz \\
        [--body_model support_data/body_models/smplh/neutral/model.npz] \\
        [--out out.blend] [--offset 1.4]
"""
import argparse
import math
import sys

import bpy
import numpy as np
from mathutils import Quaternion, Vector

SMPL_PARENTS = [-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19, 20, 21]
SMPL_JOINT_NAMES = [
    "Pelvis", "L_Hip", "R_Hip", "Torso", "L_Knee", "R_Knee", "Spine", "L_Ankle",
    "R_Ankle", "Chest", "L_Toe", "R_Toe", "Neck", "L_Thorax", "R_Thorax", "Head",
    "L_Shoulder", "R_Shoulder", "L_Elbow", "R_Elbow", "L_Wrist", "R_Wrist", "L_Hand", "R_Hand",
]
HAND_JOINTS = {22, 23}  # left at rest -- see module docstring
BONE_REST_DIR = Vector((0, 0, 0.08))  # arbitrary fixed direction/length, same for every bone
MIN_WEIGHT = 1e-4  # skip near-zero skinning weights when building vertex groups


def get_args():
    argv = sys.argv
    argv = argv[argv.index("--") + 1:] if "--" in argv else []
    p = argparse.ArgumentParser()
    p.add_argument("--npz", required=True)
    p.add_argument("--body_model", default="support_data/body_models/smplh/neutral/model.npz")
    p.add_argument("--out", default=None)
    p.add_argument("--offset", type=float, default=1.4)
    return p.parse_args(argv)


def clear_scene():
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)


def load_body_model(path):
    d = np.load(path, allow_pickle=True)
    v_template = np.asarray(d["v_template"], dtype=np.float64)         # (V, 3)
    faces = np.asarray(d["f"], dtype=np.int64)                          # (F, 3)
    weights = np.asarray(d["weights_prior"], dtype=np.float64)          # (V, 24)
    J_regressor = np.asarray(d["J_regressor_prior"], dtype=np.float64)  # (24, V)

    # The raw SMPL-H file is in the model's own native Y-up convention (confirmed
    # empirically: rest pelvis Y=-0.23, head Y=+0.36, ankles Y=-1.08 -- i.e. Y is
    # the up axis) -- but PHC's world (and the positions/rotations driving this
    # rig) is Z-up. Rotate +90 deg about X: (x, y, z) -> (x, -z, y).
    v_template = v_template[:, [0, 2, 1]] * np.array([1.0, -1.0, 1.0])

    rest_joints = J_regressor @ v_template                              # (24, 3)
    return v_template, faces, weights, rest_joints


def make_material(name, color):
    mat = bpy.data.materials.get(name) or bpy.data.materials.new(name)
    mat.diffuse_color = (*color, 1.0)
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes.get("Principled BSDF")
    if bsdf is not None:
        bsdf.inputs["Base Color"].default_value = (*color, 1.0)
        bsdf.inputs["Roughness"].default_value = 0.6
    return mat


def build_rig(name, v_template, faces, weights, rest_joints, x_offset, material):
    # Rest joints relative to pelvis, so the armature's own object-space origin
    # is the pelvis rest position -- per-frame translation then just becomes
    # setting the armature object's world location to the current pelvis position.
    rel_joints = rest_joints - rest_joints[0]

    arm_data = bpy.data.armatures.new(f"{name}_armdata")
    arm_obj = bpy.data.objects.new(f"{name}_armature", arm_data)
    bpy.context.collection.objects.link(arm_obj)
    bpy.context.view_layer.objects.active = arm_obj

    bpy.ops.object.mode_set(mode="EDIT")
    edit_bones = arm_data.edit_bones
    created = {}
    for i, jname in enumerate(SMPL_JOINT_NAMES):
        eb = edit_bones.new(jname)
        head = Vector(rel_joints[i])
        eb.head = head
        eb.tail = head + BONE_REST_DIR  # uniform direction -> identity rest orientation vs. parent
        eb.roll = 0.0
        created[i] = eb
    for i, jname in enumerate(SMPL_JOINT_NAMES):
        p = SMPL_PARENTS[i]
        if p >= 0:
            created[i].parent = created[p]
            created[i].use_connect = False

    # Uniform bone direction/roll makes every NON-root bone's rest orientation
    # identity *relative to its parent* (both share the same absolute rest
    # rotation, so the relative transform between them cancels to identity) --
    # but the ROOT (pelvis) has no parent to cancel against, so its absolute
    # rest orientation is whatever Blender's edit-bone-from-direction math
    # happens to produce for BONE_REST_DIR, which has no reason to match
    # "identity" in the data's own convention (pred_rot_local[0]/global_orient
    # = identity meaning "true world-axis-aligned", per PHC's convention).
    # Capture it here so animate_rig can cancel it out for the pelvis only.
    pelvis_rest_quat = created[0].matrix.to_quaternion()
    bpy.ops.object.mode_set(mode="OBJECT")

    for pbone in arm_obj.pose.bones:
        pbone.rotation_mode = "QUATERNION"

    mesh_data = bpy.data.meshes.new(f"{name}_mesh")
    mesh_data.from_pydata(v_template.tolist(), [], faces.tolist())
    mesh_data.update()
    mesh_obj = bpy.data.objects.new(name, mesh_data)
    bpy.context.collection.objects.link(mesh_obj)
    mesh_data.materials.append(material)

    for i, jname in enumerate(SMPL_JOINT_NAMES):
        vg = mesh_obj.vertex_groups.new(name=jname)
        col = weights[:, i]
        nz = np.nonzero(col > MIN_WEIGHT)[0]
        for vidx in nz:
            vg.add([int(vidx)], float(col[vidx]), "REPLACE")

    mesh_obj.parent = arm_obj
    mod = mesh_obj.modifiers.new(name="Armature", type="ARMATURE")
    mod.object = arm_obj

    arm_obj.location = (rest_joints[0, 0] + x_offset, rest_joints[0, 1], rest_joints[0, 2])
    return mesh_obj, arm_obj, pelvis_rest_quat


def np_quat_xyzw_to_blender(q):
    return Quaternion((float(q[3]), float(q[0]), float(q[1]), float(q[2])))


def animate_rig(armature, positions, rot_local, x_offset, fps, pelvis_rest_quat):
    """positions: (T, 24, 3) world space, SMPL_JOINT_NAMES order.
    rot_local:  (T, 24, 4) xyzw, local/parent-relative, same joint order."""
    T = positions.shape[0]
    scene = bpy.context.scene
    scene.render.fps = fps
    scene.frame_start = 1
    scene.frame_end = max(scene.frame_end, T)

    pelvis_rest_inv = pelvis_rest_quat.inverted()
    pbones = armature.pose.bones
    for t in range(T):
        frame = t + 1
        armature.location = Vector((positions[t, 0, 0] + x_offset, positions[t, 0, 1], positions[t, 0, 2]))
        armature.keyframe_insert(data_path="location", frame=frame)
        for j, jname in enumerate(SMPL_JOINT_NAMES):
            if j in HAND_JOINTS:
                continue
            pbone = pbones[jname]
            q = np_quat_xyzw_to_blender(rot_local[t, j])
            if j == 0:
                # Root only: cancel the bone's own arbitrary rest orientation so
                # "identity" in the data means true world-axis-aligned, not
                # "aligned with BONE_REST_DIR" -- see the comment in build_rig.
                q = pelvis_rest_inv @ q
            pbone.rotation_quaternion = q
            pbone.keyframe_insert(data_path="rotation_quaternion", frame=frame)


def main():
    args = get_args()
    d = np.load(args.npz, allow_pickle=True)
    pred_pos, gt_pos = d["pred_pos"], d["gt_pos"]
    pred_rot_local, gt_rot_local = d["pred_rot_local"], d["gt_rot_local"]
    fps = int(d["fps"])
    motion_key = str(d["motion_key"])

    v_template, faces, weights, rest_joints = load_body_model(args.body_model)

    clear_scene()

    bpy.ops.mesh.primitive_plane_add(size=6, location=(0, 0, 0))
    bpy.context.active_object.name = "Ground"

    mat_pred = make_material("PredSkin", (0.85, 0.30, 0.25))
    mat_gt = make_material("GTSkin", (0.25, 0.55, 0.85))

    pred_mesh, pred_arm, pred_pelvis_rest = build_rig("pred_v7_validate", v_template, faces, weights, rest_joints, -args.offset / 2, mat_pred)
    gt_mesh, gt_arm, gt_pelvis_rest = build_rig("gt_ground_truth", v_template, faces, weights, rest_joints, args.offset / 2, mat_gt)

    animate_rig(pred_arm, pred_pos, pred_rot_local, -args.offset / 2, fps, pred_pelvis_rest)
    animate_rig(gt_arm, gt_pos, gt_rot_local, args.offset / 2, fps, gt_pelvis_rest)

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

    out_path = args.out or args.npz.replace(".npz", ".blend")
    bpy.ops.wm.save_as_mainfile(filepath=out_path)
    print(f"[blender] wrote {out_path} ({pred_pos.shape[0]} frames @ {fps}fps, seq={motion_key}) "
          f"— left=pred (v7_validate), right=ground truth")


if __name__ == "__main__":
    main()
