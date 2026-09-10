"""
Numerically validate the DIRECT-RIG Blender path (blender_render_addon.py) against the PHC
trajectories, by driving the real rig inside Blender and reading back every bone's world position.

Runs blender_render_addon.py's OWN build_rig/animate_rig -- imported, not reimplemented -- so this
tests the code that will actually render, then dumps world positions for all 22 driven bones on
every frame to an npz for numpy-side comparison.

x_offset is forced to 0 here: the +/- offset/2 in the render script only separates the two rigs
side by side for viewing, so removing it puts the rig in PHC's own world coordinates.

Usage (inside Blender):
    blender --background --python scripts/validate_blender_addon_rig.py -- \
        --npz <mesh_clip.npz> --out <dump.npz>
"""
import sys
import os

import bpy
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__))))


def get_args():
    argv = sys.argv
    argv = argv[argv.index("--") + 1:] if "--" in argv else []
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--npz", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--texture", default="smplx_texture_m_alb.png")
    return p.parse_args(argv)


def main():
    args = get_args()
    try:
        bpy.ops.preferences.addon_enable(module="bl_ext.user_default.smplx_blender_addon")
    except Exception as e:                                              # noqa: BLE001
        print("[validate] addon_enable:", e)

    import blender_render_addon as BRA

    d = np.load(args.npz, allow_pickle=True)
    fps = int(d["fps"])
    BRA.clear_scene()

    results = {}
    for stream, pos_key, rot_key in [("pred", "pred_pos", "pred_rot_local"),
                                     ("gt", "gt_pos", "gt_rot_local")]:
        positions = d[pos_key]
        rot_local = d[rot_key]
        # x_offset = 0 -> rig sits in PHC's own world, not the side-by-side layout
        mesh_obj, arm, rest_rel, pelvis_rest_t = BRA.build_rig(f"val_{stream}", 0.0, args.texture)
        BRA.animate_rig(arm, rest_rel, pelvis_rest_t, positions, rot_local, 0.0, fps)

        T = positions.shape[0]
        names = [BRA.SMPL_TO_SMPLX_BONE[i] for i in sorted(BRA.SMPL_TO_SMPLX_BONE)]
        world = np.full((T, len(names), 3), np.nan, dtype=np.float64)
        quats = np.full((T, len(names), 4), np.nan, dtype=np.float64)   # wxyz, world
        for t in range(T):
            bpy.context.scene.frame_set(t + 1)
            dg = bpy.context.evaluated_depsgraph_get()
            ae = arm.evaluated_get(dg)
            mw = ae.matrix_world
            for j, nm in enumerate(names):
                pb = ae.pose.bones[nm]
                m = mw @ pb.matrix
                world[t, j] = np.array(m.translation)
                q = m.to_quaternion()
                quats[t, j] = np.array([q.w, q.x, q.y, q.z])
        results[f"{stream}_world"] = world
        results[f"{stream}_quat_wxyz"] = quats
        print(f"[validate] {stream}: {T} frames x {len(names)} bones dumped")
        # remove so the second rig starts clean
        bpy.ops.object.select_all(action="SELECT")
        bpy.ops.object.delete(use_global=False)

    np.savez(args.out, bone_names=np.asarray(names), fps=fps,
             motion_key=str(d["motion_key"]), **results)
    print(f"[validate] wrote {args.out}")


if __name__ == "__main__":
    main()
