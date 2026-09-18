# Quest2 → SMPL-H/SMPL-X → Blender Rendering — verified pipeline trace

Purpose: identify exactly which large generated files (converted pkls, npz intermediates,
rendered frames, video) can be **regenerated on demand** instead of migrated, and which upstream
source/scripts must be preserved to regenerate them. SMPL/SMPL-H body-model assets are already
migrated and out of scope here.

**Important correction to the assumed pipeline shape**, found by reading the actual code rather
than the filenames: there are **two separate, code-verified pipelines** in this repo that both
touch "Quest2 → SMPL," and only one of them feeds the actual Blender renders. Documented below as
Pipeline A and Pipeline B.

---

## Pipeline A — the actual Blender rendering path (verified: `BLENDER_EXPORT_README.md` calls this
"THE rendering path (only this one)", and I independently confirmed this by reading both scripts
directly)

```
data/quest2_synthetic/test_seg_motion/*.pkl  (Quest2 ground truth, quest_humanoid 25-joint skeleton)
        |
        v  (scripts/auto_eval_quest.sh -> run_hydra.py, test=True im_eval=True)
phc_positions_for_eval.pkl   (PHC-simulated humanoid rollout: pred_pos_all/pred_rot_all/
                               ref_pos_all/gt_rot_all, Mujoco body order, PHC's rotated
                               "upright_start" frame)
        |
        v  scripts/export_blender_mesh_motion.py
mesh_<clip>.npz   (positions + LOCAL quaternions, SMPL bone order/names -- NOT SMPL-H mesh
                    parameters, NOT vertices)
        |
        v  scripts/blender_render_addon.py  (validation)  /  scripts/blender_render_comparison.py (final)
posed SMPL-X addon rig, inside Blender itself (mesh vertices exist only transiently in Blender's
own runtime -- never written to disk as a separate mesh file)
        |
        v  PNG frame sequence -> ffmpeg
final .mp4 / stills
```

**This path never computes or stores SMPL-H mesh vertices at all.** The body actually rendered is
the official **SMPL-X** Blender addon's rig (`wm.smplx_tool.body_model = "smplx"`,
`bpy.ops.scene.smplx_add_gender()`, `blender_render_addon.py:81-83`), driven directly by PHC's
joint rotations via a per-bone rest-relative correction — not by fitting SMPL-H pose parameters and
evaluating a body model. A `support_data/body_models/smplh/neutral/model.npz` reference **is**
loaded in `export_blender_mesh_motion.py` (`load_rest_offsets`, default arg,
line 142), but only to compute rest-pose joint offsets for an **FK sanity-check residual** printed
to the console — its output is never written into `mesh_<clip>.npz` and never reaches Blender.

### Stage A1 — simulated humanoid motion (input to this pipeline)

- **Script**: `scripts/auto_eval_quest.sh` (documented fully in `QUEST2_TRAINING_REPRODUCTION.md` §12).
- **Output**: `phc_positions_for_eval.pkl` (or a step-tagged copy, e.g.
  `phc_positions_for_eval_0380160.pkl`), ~577MB. Dict of `motion_keys`, `pred_pos_all`,
  `pred_rot_all`, `ref_pos_all`, `gt_rot_all` — `[T, 25, 3]` positions / `[T, 25, 4]` xyzw
  quaternions per clip, quest_humanoid Mujoco joint order.
- **Coordinate convention**: PHC's own "upright_start" rotated frame — `has_upright_start: False`
  for `quest_humanoid` means PHC does **not** apply the SMPL-style upright rotation on top of the
  raw MJCF frame for this robot specifically, but `pred_rot_all`/`gt_rot_all` still carry the
  generic AMASS-import `upright_rot_inv` baked in from `convert_avatarposer_to_phc.py` (see
  `export_blender_mesh_motion.py`'s `_UPRIGHT_UNDO_QUAT` comment, lines 40-51) — undone explicitly
  downstream (Stage A2).
- **FPS**: 30 (Quest2 native capture rate, unchanged through the whole pipeline).

### Stage A2 — retargeting / skeleton-order conversion + PHC-compatible → SMPL-bone-order motion representation

- **Script/function**: `scripts/export_blender_mesh_motion.py`, `main()`.
- **Exact command**:
  ```bash
  python scripts/export_blender_mesh_motion.py --pos_dump <phc_positions_for_eval*.pkl> \
      --seq_idx <N> --body_model support_data/body_models/smplh/neutral/model.npz \
      --out <mesh_<clip>.npz> --fps 30
  ```
- **Working directory**: repo root, `avatarposer` (or equivalent CPU-numpy) conda env — no GPU/Isaac
  Gym needed at this stage, plain numpy.
- **Required input**: one `phc_positions_for_eval*.pkl` (Stage A1) + `--seq_idx` selecting one clip
  by index into `motion_keys`.
- **Transform chain** (verified line-by-line, `export_blender_mesh_motion.py:151-167`):
  1. Reorder Mujoco body order → `SMPL_BONE_ORDER_NAMES` order (`_MJ_2_SMPL` index map, line 38).
  2. Right-multiply every global quaternion by `_UPRIGHT_UNDO_QUAT = [0.5,0.5,0.5,0.5]` (xyzw) to
     undo PHC's baked-in upright rotation (validated independently via the FK residual check:
     mean residual dropped from 67cm to 6.8cm once applied).
  3. Convert global → local quaternions: `local[i] = conj(global[parent[i]]) * global[i]`
     (`global_to_local`, line 93), root (`i=0`) keeps its global value (no parent).
- **Quaternion convention**: xyzw (scalar-last) throughout, standard SMPL convention.
- **Root handling**: root **position** is taken directly from `pred_pos`/`gt_pos` (never
  reconstructed via FK); root **rotation** has no parent so local = global.
- **Joint ordering/count**: 24 joints, `SMPL_BONE_ORDER_NAMES` (SMPL-24, i.e. body-only — 22
  rendered per `SMPL_TO_SMPLX_BONE`, hands (indices 22/23) excluded — `robot.freeze_hand=True` is
  set on every PHC command in this project, so hand rotations are physically frozen anyway).
- **Output**: `mesh_<clip>.npz` — keys `pred_pos (T,24,3)`, `gt_pos (T,24,3)`,
  `pred_rot_local (T,24,4)`, `gt_rot_local (T,24,4)`, `joint_names`, `fps=30`, `motion_key`.
  ~560-600KB per clip (verified: 3 files, 563-599KB each in
  `data/quest_smpl_retarget/blender_export/`).
- **Built-in self-check**: an FK-consistency residual is printed at export time (expected ~2-4cm/joint
  against a modified SMPL rest pose — PHC's own startup log confirms "Using modified SMPL starting
  pose" — a residual far outside that range means a real bug, per the script's own docstring). This
  is diagnostic only; it does not gate or alter the written npz.
- **Fully reproducible**: yes, deterministic numpy transform of an existing `phc_positions_for_eval.pkl`
  — no randomness, no network calls, no GPU.

### Stage A3 — Blender rig construction and animation (validation build)

- **Script**: `scripts/blender_render_addon.py`.
- **Must run inside Blender's own Python**, not a conda env: `bpy` is Blender's embedded module.
- **Blender version**: **5.2** (confirmed by 3 independent doc references —
  `scripts/quest_smpl_addon_rig_validation.md:3`, `scripts/BLENDER_EXPORT_README.md:63`,
  `scripts/validate_addon_rig_report.py:69` — and `blender_render_comparison.py:108` has an
  explicit compatibility branch for "Blender 5.x['s] fcurves under action layers/slots").
- **Addon dependency**: the **official SMPL-X Blender addon** (academic license), installed via
  Blender's `extensions.package_install_files` (`blender_render_addon.py:2-3`). This addon is an
  **external Blender installation artifact, not part of this git repo**, and per
  `BLENDER_EXPORT_README.md`'s own note, the addon's optional SMPL-H body-model file
  (`smplh_model_20260511.blend`) is **not currently installed** — only the SMPL-X body model is
  available in this Blender setup.
- **Exact command**:
  ```bash
  blender --background --python scripts/blender_render_addon.py -- \
      --npz data/quest_smpl_retarget/blender_export/mesh_<clip>.npz \
      [--out out.blend] [--offset 1.4] [--texture smplx_texture_m_alb.png]
  ```
- **Required input**: `mesh_<clip>.npz` (Stage A2).
- **What it does** (`build_rig`/`animate_rig`, lines 79-150):
  - Creates the SMPL-X addon's mesh+armature (`bpy.ops.scene.smplx_add_gender()`, gender=neutral,
    **betas = 0** i.e. the rig's default neutral shape — matches the physics-simulated humanoid,
    which also runs with no shape variation).
  - Computes each bone's **rest-relative-to-parent** rotation once (`rest_relative` dict).
  - Per frame: sets `armature.location` from `positions[t,0]` (root/pelvis world position,
    corrected for the rig's own fixed pelvis-rest offset), and for each of the 22 mapped bones sets
    `rotation_quaternion = rest_relative[i].inverted() @ local_smpl[i]`.
- **Root/bone-hierarchy handling — verified bug + fix**: this rig has an extra `root` bone above
  `pelvis` (not part of the SMPL chain, never posed) carrying a fixed **+90° rotation about X** —
  this is how the rig encodes a Y-up/Z-up convention difference internally. Using a naive
  parent-relative correction for `pelvis` leaves that 90° uncancelled (measured: 568mm mean /
  1383mm max joint error, 91.05° residual rotation). **Fix, already applied in the committed code**
  (`build_rig`, lines 99-109): pelvis is corrected against its **absolute** rest rotation instead of
  parent-relative, since its parent (`root`) is outside the SMPL chain. Verified after fix: 34.5mm
  mean joint error (skeleton-shape difference, see below), 0.000° residual rotation.
- **Native Z-up — verified, not assumed**: "the rig's own rest pose already sits correctly in
  Blender's native Z-up world (`armature.matrix_world` is identity, head sits above pelvis via the
  ordinary Z axis)" (`blender_render_addon.py:26-29`) — **no Y-up conversion is applied anywhere on
  this path.** (A separate, abandoned route — `scripts/export_amass_for_addon.py` producing the
  `amass_gt_*.npz`/`amass_pred_*.npz` files still present in `blender_export/` — did attempt a
  Y-up conversion + `target_pelvis_zup_frame1` calibration; it introduced an unexplained ~0.74m
  offset and is **not used** by the final render. Those files are historical leftovers.)
- **SMPL-H vs. rendered-rig mismatch — quantified, not hand-waved**: the ~34mm mean/joint
  discrepancy between the fitted rotations (fitted against an SMPL-H-shaped skeleton upstream) and
  the rendered SMPL-X rig's own rest skeleton correlates at **r≈0.98–0.99** with each joint's known
  SMPL-X-vs-SMPL-H rest-offset difference (mean 31.4mm/max 74.4mm, stature 62.0 vs 57.6cm) — i.e.
  it is the rig's own body shape, not a pipeline bug, and is **not closable without installing the
  addon's SMPL-H `.blend` model**, which is not currently installed.
- **Validation evidence**: `scripts/validate_blender_export.py` (export fidelity, Stage A2 vs PHC:
  0.077mm mean/0.212mm max over 22 joints, 48/48 checks) and
  `scripts/validate_addon_rig_report.py` (rig fidelity, Stage A3 vs PHC, 54/54 checks, table
  reproduced in `scripts/quest_smpl_addon_rig_validation.md`).
- **Output of this stage**: a `.blend` file (only when `--out` is used for inspection/debugging) —
  not part of the production render path below, which calls `build_rig`/`animate_rig` directly
  without persisting an intermediate `.blend`.
- **Fully reproducible**: yes, deterministic given `mesh_<clip>.npz` + the same Blender version +
  addon — but **requires the SMPL-X addon to be installed in that Blender instance**, which is a
  manual, licensed, one-time setup step outside this repo's file tree (see §5, unresolved gaps).

### Stage A4 — final comparison render (production path)

- **Script**: `scripts/blender_render_comparison.py` — reuses `build_rig`/`animate_rig`
  **unchanged** from Stage A3's validated code, adds scene/camera/labels/materials only.
- **Required input**: `mesh_<clip>.npz` (predicted + GT-through-PHC-frame streams) **and**
  `quest_gt_<clip>.npz` (raw Quest ground-truth joint positions, a 25-joint marker-cloud skeleton —
  different rig from the SMPL-X body — for the third "Quest GT" column). Only 17 of
  `quest_gt`'s `ik_target_joints` are populated (`validation_groups.high_confidence` in
  `scripts/quest_smpl_correspondence.json`: head, l/r elbow, l/r wrist, l/r hip, l/r knee, l/r
  ankle — 11 high-confidence + a `structural_mismatch` group of 5 more with known-poorer fit;
  hands and toes are explicitly excluded/synthesized-out, `retarget_quest_to_smpl.py:503` — see
  §"Quest skeleton has no toe target" below).
- **Exact command**:
  ```bash
  blender --background --python scripts/blender_render_comparison.py -- \
      --npz data/quest_smpl_retarget/blender_export/mesh_<clip>.npz \
      --quest data/quest_smpl_retarget/blender_export/quest_gt_<clip>.npz \
      --tag <clip> --outdir <dir>
  ```
  All three clips at once: `bash scripts/render_step7_all.sh`, then
  `bash scripts/make_step7_summary.sh` for the combined summary video.
- **Scene layout**: Quest GT (x=−2.3) | pre-physics (x=0) | post-PHC (x=+2.3), 0.5m checker floor
  at Z=0, camera follows pre-physics root XY (31-frame smoothed), keyframes forced **LINEAR**
  interpolation (not Blender's default Bezier — deliberate, so the render doesn't visually smooth
  over exactly what it's evidence for).
- **Output**: one PNG per frame into `<outdir>/_<clip>_frames/f_%04d.png`, plus 3 still frames as
  standalone PNGs.
- **Encoding step** (separate, this Blender build has no built-in FFMPEG output enum):
  ```bash
  $CONDA_PREFIX/bin/ffmpeg -framerate 30 -i <dir>/_<clip>_frames/f_%04d.png \
      -c:v libx264 -pix_fmt yuv420p -crf 18 <dir>/<clip>_comparison.mp4
  ```
- **Verification**: `scripts/verify_step7_renders.py`, 18/18 checks (frame count == motion length
  exactly, no dropped/duplicated PNGs, 30fps, correct duration, resolution).
- **Fully reproducible**: yes, given `mesh_<clip>.npz` + `quest_gt_<clip>.npz` + Blender 5.2 + SMPL-X
  addon + ffmpeg.

---

## Pipeline B — the SEPARATE SMPL-H retargeting/mesh-penetration path (used for the Ground
Penetration metric, NOT for the rendered videos)

This is a genuinely different pipeline, documented fully in the previous audit turn
(`README.md` §5b) and repeated here only for the parts relevant to "does the repo ever compute real
SMPL-H mesh vertices" (it does — just not for rendering):

- **Script**: `scripts/retarget_quest_to_smpl.py` (`fit_single_frame`, per-frame position-based IK,
  single-frame only, no sequence/export mode of its own) + `scripts/mesh_gpen_quest.py` (sequence
  driver, caches per-clip IK results in `data/quest_smpl_retarget/mesh_gpen_cache/*.pkl`).
- **Flow**: Quest GT local `pose_quat` + `root_trans_offset` → Quest FK (poselib, MJCF skeleton) →
  world targets in Quest's **native Z-up frame** → 16 fit targets (from the correspondence
  artifact) → optimize SMPL **root_orient + body_pose (axis-angle) + root translation** → SMPL FK
  (lightweight differentiable skeleton FK, **not** the mesh BodyModel, during optimization) →
  residuals. The actual mesh (**SMPL-H NEUTRAL**, `support_data/body_models/smplh/neutral/model.npz`
  + `dmpls/neutral/model.npz`, betas=0) is only evaluated afterward, in `mesh_gpen_quest.py`'s
  `mesh_gpen()` function, to get real 6890-vertex meshes for the ground-penetration metric.
  **Root translation**: taken from the IK fit directly, not reconstructed. **Rotation
  representation**: axis-angle (`root_orient_aa`, `body_pose_aa`) for this pipeline, not quaternion
  — different convention from Pipeline A's local xyzw quaternions.
- **Output**: `data/quest_smpl_retarget/mesh_gpen_cache/<clip>__<label>.pkl` — cached per-clip IK
  fit records (`root_orient_aa`, `body_pose_aa`, `trans`, plus per-frame IK-error diagnostics), 15
  files, ~29MB total.
- **Not an input to any Blender render** — confirmed by reading both `mesh_gpen_quest.py` (never
  imports/calls `bpy` or writes any file Blender reads) and `blender_render_comparison.py`/`blender_render_addon.py`
  (never read `mesh_gpen_cache/`).
- **Fully reproducible**: yes, from `data/quest2_synthetic/test_seg_motion/` + the correspondence/
  calibration JSONs + the SMPL-H neutral body model (already migrated) — the IK optimization is
  deterministic gradient descent from a fixed init (warm-started frame-to-frame), not random.

---

## Verified facts checklist (as requested)

| Claim | Verified? | Evidence |
|---|---|---|
| Native Z-up throughout | **Yes** | `blender_render_addon.py:26-29` — rig rest pose sits correctly in Blender's native Z-up with identity `armature.matrix_world`; no Y-up conversion applied on the production path |
| Quest skeleton has no toe target | **Yes** | `quest_joint_names` (`scripts/quest_smpl_correspondence.json`) ends its leg chain at `b_l_foot_twist`/`b_r_foot_twist` — no toe joint exists in the 25-name list; `retarget_quest_to_smpl.py:503` explicitly excludes "hands, toes" as synthesized/not-fitted |
| SMPL-H mapping | **Partially — and the mismatch itself is the finding** | Pipeline A's rendered rig is **SMPL-X**, not SMPL-H; SMPL-H is only referenced for an FK sanity-check reference in Stage A2 and for the genuinely separate Pipeline B (Ground Penetration mesh) |
| Root-offset handling | **Yes** | Pipeline A: root position taken directly from `pred_pos[:,0]`/`gt_pos[:,0]`, never FK-reconstructed; rig's own fixed pelvis-rest offset is cancelled explicitly (`animate_rig`, line 142-143) so `armature.location` alone determines exact world position |
| Quaternion ordering | **Yes** | xyzw (scalar-last) throughout Pipeline A; `np_quat_xyzw_to_blender` (line 75-76) explicitly reorders to Blender's own wxyz `Quaternion()` constructor at the point of use |
| Blender root orientation | **Yes, bug-then-fix documented** | Extra `root` bone (+90° X, unposed) above `pelvis`; naive parent-relative correction left it uncancelled (568mm/1383mm error); fixed by correcting pelvis against absolute rest rotation (`build_rig`, lines 99-109) |
| SMPL-H vs. Blender rig mismatch | **Yes, quantified** | ~34mm mean joint error, r≈0.98-0.99 correlated with known SMPL-X-vs-SMPL-H rest-skeleton shape difference (31.4mm mean/74.4mm max) — not closable without installing the addon's (currently absent) SMPL-H `.blend` model |

---

## Large generated outputs

| Path | Size | Regeneration source/command | SAFE TO OMIT? |
|---|---|---|---|
| `data/quest_smpl_retarget/blender_export/mesh_{gestures,martialarts,sports}.npz` | ~1.7MB (3 files) | Stage A2: `python scripts/export_blender_mesh_motion.py --pos_dump <pkl> --seq_idx N ...` from an existing `phc_positions_for_eval.pkl` | **REVIEW** — trivially cheap to regenerate given the source pkl below, but see gap in §5 (exact `--seq_idx` per clip not separately recorded) |
| `data/quest_smpl_retarget/blender_export/quest_gt_{gestures,martialarts,sports}.npz` | ~800KB (3 files) | Produced alongside `mesh_*.npz` by the same export step (companion Quest-GT marker extraction) | **REVIEW** — same caveat |
| `data/quest_smpl_retarget/blender_export/amass_gt_*.npz`, `amass_pred_*.npz` | ~1.7MB (6 files) | `scripts/export_amass_for_addon.py` — **abandoned route, not used by any current render** | **YES** — confirmed unused by the production path; safe to omit unconditionally |
| `data/quest_smpl_retarget/renders/_{sports,gestures,martialarts}_frames/` (raw PNG sequences) | 878MB + 921MB + 865MB ≈ **2.66GB** | Stage A4: re-run `blender_render_comparison.py` (or `scripts/render_step7_all.sh`) against the small `mesh_*.npz`/`quest_gt_*.npz` files above | **YES** — this is the single largest reclaimable chunk in this whole pipeline; the encoded `.mp4` already captures everything these frames show |
| `data/quest_smpl_retarget/renders/{sports,gestures,martialarts}_comparison.mp4`, `summary_all_three.mp4` | ~28.5MB total | Encode step: `ffmpeg -framerate 30 -i .../f_%04d.png ...` from the PNG frames above, or full Stage A4 re-render | **NO** — this is the actual deliverable video; regenerating it is possible but pointless to discard when it's this small and IS the artifact people look at |
| `data/quest_smpl_retarget/renders/*_still_f*.png` (9 stills) | ~19.6MB | Byproduct of the same Stage A4 render run | **NO** — tiny, and these are cited illustrations, not intermediates |
| `data/quest_smpl_retarget/mesh_gpen_cache/*.pkl` (15 files) | 29MB | Pipeline B: `scripts/mesh_gpen_quest.py` — cache is rebuilt automatically (checked via `os.path.isfile(ck)`, line 160) on next invocation if missing | **YES** — script-confirmed disposable cache, not authored output |
| `phc_positions_for_eval*.pkl` (per experiment, e.g. `quest_distill_trackerr/`) | ~577MB each | `scripts/auto_eval_quest.sh <exp> <step>` — full Isaac Gym physics re-simulation (expensive: GPU, ~minutes-to-tens-of-minutes) | **REVIEW** — cheap to *describe* as regeneratable, but expensive to actually regenerate (needs GPU + IsaacGym + the student checkpoint); keep if disk allows, omit only if you're confident you'll re-run eval anyway |
| `output/HumanoidIm/ap_phc_distill_v7_validate/*.blend` (3 files) | not sized here — different pipeline (AMASS track) | Out of scope for this Quest2-focused document | **REVIEW** — flagged only, not traced here |

**Total safely omittable now, with high confidence (Yes rows only): ≈ 2.66GB + 29MB + ~1.7MB ≈ 2.7GB.**
Adding the `REVIEW` rows (small npz exports + the 577MB rollout dumps) could reclaim meaningfully
more, but the rollout dumps trade disk for GPU-hours on the new machine — a genuine tradeoff, not a
free win, so left as your call.

---

## Answers

**1. Minimum upstream source data that must be preserved:**
- `data/quest2_synthetic/test_seg_motion/` (already migrated — the ground-truth Quest2 motion this
  whole pipeline ultimately traces back to)
- The trained student checkpoint you want to render (e.g. `student_0380160.pth`) + its teacher/init
  dependencies — already documented in `QUEST2_TRAINING_REPRODUCTION.md`
- `scripts/quest_smpl_correspondence.json`, `scripts/quest_smpl_frame_calibration.json` (retargeting
  correspondence/calibration artifacts — small, git-tracked, load-bearing for Pipeline B and for
  `quest_gt_*.npz`'s `ik_target_joints`)
- SMPL-X Blender addon installation + (optionally) SMPL-H `support_data/body_models/smplh/neutral/`
  — already migrated per this task's framing

**2. Source/scripts that must be preserved** (all git-tracked, all small):
`scripts/export_blender_mesh_motion.py`, `scripts/blender_render_addon.py`,
`scripts/blender_render_comparison.py`, `scripts/render_step7_all.sh`,
`scripts/make_step7_summary.sh`, `scripts/verify_step7_renders.py`,
`scripts/validate_blender_export.py`, `scripts/validate_addon_rig_report.py`,
`scripts/retarget_quest_to_smpl.py`, `scripts/mesh_gpen_quest.py` (Pipeline B),
`scripts/auto_eval_quest.sh` (Stage A1).

**3. Generated outputs that can safely be regenerated:**
- All raw PNG frame sequences under `data/quest_smpl_retarget/renders/_*_frames/` (2.66GB) —
  re-derivable from the tiny `mesh_*.npz`/`quest_gt_*.npz` files via Stage A4.
- The abandoned `amass_gt_*.npz`/`amass_pred_*.npz` files (1.7MB) — confirmed unused by any current
  render.
- `data/quest_smpl_retarget/mesh_gpen_cache/*.pkl` (29MB) — script-confirmed disposable cache.

**4. Total storage that can therefore be omitted from migration:**
**≈ 2.7GB** with high confidence (the `_*_frames/` PNG dumps dominate this). The `mesh_*.npz`/
`quest_gt_*.npz` files themselves are tiny (~2.5MB total) and not worth the risk of omitting given
the minor uncertainty in #5 below — keep them; they cost nothing.

**5. Unresolved reproducibility gaps:**
- The exact `--seq_idx` (or equivalent clip-selection argument) used to produce each of the 3
  `mesh_<clip>.npz`/`quest_gt_<clip>.npz` pairs was not found recorded anywhere (not in the npz
  itself beyond `motion_key`, not in a driving shell script) — regenerating them requires knowing
  which `motion_keys` index in `phc_positions_for_eval.pkl` corresponds to
  `SUBJECT070.sports.pkl_0_417` etc.; recoverable by searching `motion_keys` for a matching
  substring, but not a one-line rerun.
- The SMPL-X Blender addon's installation (and specifically the absent SMPL-H `.blend` model file)
  lives outside this repo's file tree entirely — not migrated by any file-copy of this repo, and
  its academic-license download process wasn't traced here (out of scope: SMPL/SMPL-H already
  resolved per this task).
- `scripts/export_amass_for_addon.py` (producer of the abandoned `amass_*.npz` files) was not read
  in this pass — its own I/O contract is undocumented here since its output is confirmed unused.

STOP — documentation only, nothing run, no source modified, nothing deleted.
