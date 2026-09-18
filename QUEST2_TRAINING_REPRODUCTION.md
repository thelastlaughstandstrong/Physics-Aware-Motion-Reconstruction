# Quest2 Training Reproduction — verified pipeline trace

This document exists for one purpose: reproducing the **current, reported Quest2 result**
(`trackerr@380k`, `student_0380160.pth`) — either by re-evaluating that exact checkpoint or by
retraining from scratch. It is a line-by-line trace of the actual code path, not a description of
intent. Every file/class/function/shape/command below was read or executed (read-only) directly
from the repository on 2026-09-18; nothing here is inferred from filenames or comments alone unless
explicitly marked "reconstructed" or "not directly observed."

SMPL/SMPL-H body-model assets are **out of scope here** — already migrated separately, and (per
`README.md` §5b, verified by tracing the GPen code path) not required for this pipeline at all: the
Quest2 headline metrics and student inference never load a body model.

---

## Pipeline trace, stage by stage

### 1. Quest2 source data

- **Dataset path**: `data/quest2_synthetic/train_seg_motion/` (training) and
  `data/quest2_synthetic/test_seg_motion/` (evaluation, 1072 sequences) — real SimXR
  HMD+controller recordings, segmented into per-clip `.pkl` files, schema
  `{pose_quat, root_trans_offset, fps}` matching `phc/phc/utils/motion_lib_quest.py`'s expected
  format (confirmed by `scripts/convert_avatarposer_quest2_to_phc.py`'s own module docstring,
  which targets this exact schema).
- Sibling artifacts in the same directory (`train_seg/`, `test_seg/`, `test_seg_avatarposer_pred/`,
  `extract_motion_only.log`, and matching `.zip`s) record earlier/alternate extraction stages; the
  two `_seg_motion` directories are what training and eval actually read (verified: `MOTION_FILE`
  in `scripts/train_distill_quest_with_auto_recovery.sh` line 60, `TEST_MOTION` in
  `scripts/auto_eval_quest.sh` line 27).
- **Already migrated in full** — this is the 511GB dataset payload described in `README.md` §1b;
  not re-derived here.

### 2. Preprocessing

The training/test `_seg_motion` pkls are themselves the preprocessed artifact (already migrated,
see above) — no preprocessing step needs to run again to train or evaluate `trackerr`. A related
but **separate** preprocessing script exists for a different experiment (the "naive PHC baseline",
not `trackerr`): `scripts/convert_avatarposer_quest2_to_phc.py` converts AvatarPoser's own
kinematic prediction (not ground truth) into the same pkl schema, targeting
`data/quest2_synthetic/test_seg_avatarposer_pred/`. **Not required for reproducing `trackerr`.**

### 3. Sparse head + left/right wrist input

- **File/function**: `phc/phc/learning/distill_agent_quest.py:270`, `_build_hmd_features_quest(ref_state_cur, ref_state_prev)`.
- **Input**: two per-frame reference states (current + previous), each holding global rigid-body
  rotation (`rb_rot`) and position (`rg_pos`) restricted to `_QUEST_HMD_JOINT_IDS` (the 3
  HMD-tracked bodies: head, left wrist, right wrist).
- **Output**: `(N, 54)` tensor per frame = concat of `[rot6d(18), rot_vel6d(18), pos(9), pos_vel(9)]`
  — 6D rotation representation for 3 joints (3×6=18) for both current-frame rotation and
  frame-to-frame rotational velocity, plus 3 joints × 3D position (9) and its frame-to-frame delta
  (9).
- Frames are accumulated into a rolling window buffer: `self._hmd_buf` = `(num_envs, window_size+1, 54)`
  = `(num_envs, 41, 54)` with `window_size=40` (`distill_agent_quest.py:387,473`) — this is the
  "HMD window (41 × 54D)" in the architecture diagram at the top of that file.

### 4. Temporal AvatarPoser backbone

- **Class**: `_AvatarPoserBackbone` (`phc/phc/learning/distill_agent.py:262`), shared with the
  SMPL-track pipeline; `forward()` at line 278.
- **Verified shapes** (read directly from `student_0380160.pth`'s `backbone` state dict):
  - `linear_embedding`: `(256, 54)` — projects each 54D frame into the 256D transformer channel.
  - `transformer_encoder`: 3 layers (`layers.0/1/2`), self-attention `out_proj (256,256)` each.
  - Output: **z, 256D** latent (one vector per window, per the architecture diagram).
- **Trainable here** (unlike the SMPL-track pipeline, where the backbone is frozen) — `trackerr`'s
  config has `freeze_backbone: False` (`phc/phc/data/cfg/learning/distill_quest.yaml`), the whole
  point of the "Finetuned Backbone" name.
- **Initialization weights**: loaded from `avatarposer_checkpoint` before any training step (see
  Teacher/Init section below).

### 5. Latent z

256D vector, output of the backbone above, consumed by both the q_base decoder (kinematic branch)
and `state_enc` (control branch) — see architecture diagram in `distill_agent_quest.py:7-46`.

### 6. q_base pose decoder

- **Function**: `_AvatarPoserBackbone.decode(z)` (part of the same class, `distill_agent.py:262`
  region), invoked at `distill_agent_quest.py:395,407`.
- **Verified shapes**: `stabilizer` (root decoder): `256→256→6` (`stabilizer.2.weight: (6,256)`) →
  **q_base_root, 6D** (rotation-6D root orientation). `joint_rotation_decoder`: `256→256→144`
  (`joint_rotation_decoder.2.weight: (144,256)`) → **q_base_joints, 144D** (24 joints × 6D rotation
  for `quest_humanoid`, vs. 126D/21 joints for the SMPL skeleton — confirmed in `README.md` §"AMASS-pretrained
  backbone" section and via `avatarposer_checkpoint`'s own final-layer shape mismatch note).
- **Loss**: `kinematic_loss` = `root_loss + joint_rot_loss + joint_pos_loss` against the
  **ground-truth pose from the motion file** — always on, every batch, regardless of
  `explicit_pose_action_input`. This is the term visible in the training log as
  `kinematic 0.0086 (root ... jrot ... jpos ...)`.
- q_base is **not used at inference/control time by default** (`explicit_pose_action_input: False`
  for `trackerr` — confirmed in the reconstructed launch config, no override present) — it only
  supplies the kinematic-loss training signal; `action_head`/`value_head` condition on `z`, not
  `q_base`, for this checkpoint.

### 7. Simulator state encoder h

- **Class**: `state_enc`, a 3-layer MLP, part of `_AvatarPoserPHCStudent`
  (`distill_agent.py:313`).
- **Verified shapes**: `net.0`: `(256, 372)` → `net.2`: `(256,256)` → `net.4`: `(256,256)`. Input
  372D = PHC's own proprioceptive self-observation (`self_obs_dim=372`, printed at agent
  construction and matching `env.task._num_self_obs`). Output: **h, 256D**.

### 8. Student controller

- **Class**: `_AvatarPoserPHCStudent` (`distill_agent.py:313`), `forward()` at line 386,
  `_action_feat()` at line 366.
- **Feature assembly** (default, `explicit_pose_action_input=False`, `use_tracking_error=True`):
  `feat = concat(z.detach(), h, track_err)` = `256 + 256 + 27 = 539D` — verified exactly:
  `action_head.0.weight: (512, 539)`.
- **`track_err`, 27D**: `_build_tracking_error_quest()` (`distill_agent_quest.py:291`) — closed-loop
  reference-minus-simulated delta at the 3 HMD-tracked bodies, in the humanoid's own heading frame
  (`3 bodies × (3 pos + 6 rot-tan-norm) = 27`). This is the `use_tracking_error` fix; without it
  `feat` would be 512D and the checkpoint would not load into a student built with it enabled (and
  vice versa — the two checkpoint families are structurally incompatible, not just numerically
  different).
- **`action_head`**: `539 → 512 → 512 → 72` → **control, 72D** (`action_dim`, PHC's DOF count for
  `quest_humanoid`). Always trained in Stage E.
- **`value_head`**: same shape, output `1D` scalar. **Present in the checkpoint but never trained
  in Stage E** — `_train_on_batch` never calls it; it exists only for a would-be Stage R run
  (confirmed: Stage R critic starts from ~random, per `results_comparison.md`'s documented root
  cause of Stage R collapse).
- **`action_log_std`**: `(72,)` tensor, also present but likewise inert for Stage E inference
  (only meaningful for Stage R's stochastic sampling).

### 9. Frozen PHC teacher

- **File**: `output/HumanoidIm/phc_prim_quest/Humanoid.pth` (355,341,223 bytes, dated 2024-08-19).
- **Loaded**: unconditionally, in `DistillAgentQuest.__init__` (`distill_agent_quest.py:372-384`),
  every time the agent is constructed — training **and** evaluation. `requires_grad=False`
  immediately after load; `self.model.eval()`.
- **Role**: supplies `teacher_action` during `_collect_batch()` (Pass 1) as the label for
  `action_loss` in Pass 2. **Never used to step the simulator directly** and **never used at
  evaluation time for anything the reported metrics measure** — but the code will fail to
  construct the agent at all without a loadable file at this path, so it is a hard dependency for
  running eval on `student_0380160.pth`, not only for training.

### 10. Stage E supervised/distillation training

- **Config file**: `phc/phc/data/cfg/learning/distill_quest.yaml` (hydra group `learning=distill_quest`).
- **Key config values** (as committed, i.e. the defaults `trackerr` ran with — no evidence of
  overrides beyond `max_distill_step`, see below):
  `seed: 0`; `student_lr: 1e-4`; `backbone_lr: 1e-4`; `lambda_action: 0.1`; `lambda_kinematic: 1.0`;
  `kinematic_root_weight: 0.02`; `foot_pos_weight: 1.0`; `freeze_backbone: False`;
  `explicit_pose_action_input: False`; `use_tracking_error: True`; `window_size: 40`;
  `enable_stage_r: False`; `avatarposer_checkpoint: "../results/AvatarPoseEstimationQuest2/models/15000_G.pth"`;
  `teacher_checkpoint: "../output/HumanoidIm/phc_prim_quest/Humanoid.pth"`.
- **Working directory**: `/home/sia-irl/AvatarPoser/phc` (the launch script `cd`s there before
  invoking `run_hydra.py`; all relative paths above, e.g. `../results/...`, are relative to this).
- **Conda env**: `rlgpu`.
- **Launch script**: `scripts/train_distill_quest_with_auto_recovery.sh` — confirmed by the
  presence of `output/HumanoidIm/.recovery_logs/quest_distill_trackerr/attempt_1.log` (a single
  attempt — this run never crashed/restarted, unlike several sibling experiments).
- **Exact command constructed by that script** (`train_distill_quest_with_auto_recovery.sh:104-113`):
  ```bash
  cd /home/sia-irl/AvatarPoser/phc
  conda run -n rlgpu --no-capture-output python phc/run_hydra.py \
      learning=distill_quest exp_name=quest_distill_trackerr \
      env=env_quest_im_basic robot=quest_humanoid \
      env.motion_file=/home/sia-irl/AvatarPoser/data/quest2_synthetic/train_seg_motion \
      env.num_envs=128 env.cycle_motion=True env.min_length=5 \
      learning.params.config.max_distill_step=500000 \
      headless=True no_virtual_display=True no_log=True disable_multiprocessing=False \
      test=False im_eval=False
  ```
  **Reconstructed, not directly recovered**: the literal shell invocation used for this specific
  run was not found in `.bash_history` (only sibling experiments' invocations were). This command
  is assembled from (a) the launcher script's own fixed template, (b) its single-attempt recovery
  log confirming no crash/resume occurred, and (c) the training log's own final line —
  `max_distill_step=500000` is certain (the log literally says
  `hit max_distill_step cap (500000) at step 500000 -- stopping`); `num_envs=128` and the recovery
  script's other positional defaults are inferred from the script's own defaults, not independently
  confirmed for this exact run.
- **Determinism**: `phc/phc/run_hydra.py:322` calls `set_seed(cfg.get("seed", -1), cfg.get("torch_deterministic", False))`.
  `seed=0` is set; `torch_deterministic` was **not found set anywhere in the resolved config** and
  defaults to `False` in that call. **This training run is seeded but not guaranteed bit-exact
  reproducible** — Isaac Gym's GPU physics and unpinned non-deterministic CUDA/cuDNN ops mean a
  from-scratch retrain with identical config will very likely reach a similar but not
  numerically-identical `student_0380160.pth`.
- **Total training steps**: 500,000 (config default `max_distill_step` is actually 195,328; this
  run's launch explicitly overrode it to 500,000, per the reconstructed command above).
- **Checkpoint save interval**: every 480 steps (verified directly from the training log: `saved ->
  student_0000480.pth`, `_0000960.pth`, `_0001440.pth`, ... — one save every 15 "batches" ×
  `horizon_length=32` = 480 steps). 1042 checkpoints total for the full run.

### 11. Student checkpoint

- **Save function**: `_save_student` (inherited unchanged from `DistillAgent`, per
  `DistillAgentQuest`'s class docstring, `distill_agent_quest.py:364`).
- **Load function** (resume/eval): `_resume_from_checkpoint` (`distill_agent.py:524-544`) — loads
  `backbone`, `state_enc`, `action_head`, and (if present) `value_head`/`action_log_std` from the
  checkpoint dict, **fully overwriting** whatever `avatarposer_checkpoint` initialized moments
  earlier in `__init__`. Also restores `self._distill_step` and `self._stage` from the checkpoint.
- **Best checkpoint**: `student_0380160.pth` — selected via `scripts/quest_checkpoint_scan.sh`'s
  automated full-range scan over the completed run (`output/HumanoidIm/quest_distill_trackerr/checkpoint_scan.log`),
  scanning every ~10k steps from 130000 to 500000 with patience=99 (i.e., effectively exhaustive,
  not early-stopped): `DONE. Best checkpoint for quest_distill_trackerr: student_0380000.pth
  (success_rate=0.73787)` (log rounds to the nearest 10k; the exact saved file is
  `student_0380160.pth`). Confirmed internally consistent: `torch.load(...)['step'] == 380160`,
  `['stage'] == 'E'`, `['track_err_dim'] == 27`.
- **Size**: 21,913,723 bytes.
- **Nothing supersedes it**: the Stage R follow-on (`quest_distill_trackerr_stager`, seeded from
  this exact checkpoint) collapsed monotonically and never beat the seed; the only currently-active
  experiment (`quest_distill_footw2`) has no completed full-range scan as of this writing.

### 12. Evaluation

- **Script**: `scripts/auto_eval_quest.sh <exp_name> <target_step> [extra_overrides] [num_envs]`.
- **Working directory**: `/home/sia-irl/AvatarPoser/phc` (script `cd`s there internally).
- **Two-part pipeline it runs**:
  1. Physics rollout (`conda run -n rlgpu`): same `run_hydra.py` entry as training, but
     `env.motion_file=data/quest2_synthetic/test_seg_motion`, `env.cycle_motion=False`,
     `test=True im_eval=True`, and
     `learning.params.config.student_checkpoint=output/HumanoidIm/quest_distill_trackerr/student/student_0380160.pth`
     — writes `output/HumanoidIm/quest_distill_trackerr/phc_positions_for_eval.pkl`, copied to
     `..._0380160.pkl`.
  2. Metrics (`conda run -n avatarposer`): `scripts/evaluate_phc_metrics_quest.py --pos_dump
     phc_positions_for_eval_0380160.pkl --out quest_eval_0380160.json` — pure joint-position math
     (no SMPL/mesh), computing `rot_error_deg`, `pos_error_cm`/`eg_mpjpe_mm`, `vel_error_cm_s`,
     `jitter_m_s3`, `foot_skate_cm_s`, `gpen_*` (near-zero, rigid-body, not the mesh GPen used
     elsewhere in `results_comparison.md`), `success_rate_frame_head`, `success_rate_phc_survival`.
- **Expected result files**: `output/HumanoidIm/quest_distill_trackerr/quest_eval_0380160.json`
  (and matching `.log`), `phc_positions_for_eval_0380160.pkl` (577MB — physics rollout dump).
- **Current reference metrics** (byte-exact from `quest_eval_0380160.json`, matching
  `results_comparison.md`'s "Finetuned Backbone" column):
  ```
  success_rate_phc_survival : 0.7378731343283582   (reported 0.738)
  eg_mpjpe_mm                : 87.43231743574142     (reported 87.43)
  rot_error_deg               : 15.231768526054646    (reported 15.23)
  jitter_m_s3                 : 524.7776038726764     (reported 524.78)
  foot_skate_cm_s              : 17.14328549789923     (reported 17.14)
  success_rate_frame_head     : 0.9891149288977072
  vel_error_cm_s               : 34.245696086818775
  ```

---

## Why the teacher and Phase-A checkpoints load before the student — and whether that's required

`DistillAgentQuest.__init__` executes, in this fixed order, **every time the agent is constructed**
(training or eval, no branch skips any of these three steps):

1. Load `teacher_checkpoint` into `self.model`, freeze it (`distill_agent_quest.py:372-384`).
2. Construct `self.student`, then load `avatarposer_checkpoint` into `self.student.backbone`
   (`distill_agent_quest.py:406-418`, via `_load_backbone_weights`, `distill_agent.py:427`).
3. Call `self._resume_from_checkpoint(cfg)` (`distill_agent_quest.py:495` →
   `distill_agent.py:524`), which — **only if `student_checkpoint` is set** — loads and fully
   overwrites `backbone`, `state_enc`, `action_head`, `value_head`, `action_log_std` from the
   student checkpoint.

**Is this strictly required by the current code path?** Yes, structurally — steps 1 and 2 are
unconditional in the constructor; there is no code path (train or eval, `student_checkpoint` set or
not) that skips them. **But their actual weights are functionally irrelevant once
`student_checkpoint` is set**: step 3 completely overwrites everything step 2 wrote before a single
forward pass happens on real data, and the teacher (step 1) is never involved in producing the
metrics evaluated in §12 above — only in supplying training-time labels. In short: **the files must
exist and load successfully for `evaluate_phc_metrics_quest.py`'s pipeline to even start, but
neither file's actual content affects `student_0380160.pth`'s reported numbers.** This was
confirmed by reading the constructor and `_resume_from_checkpoint` directly (not inferred).

---

## File classification

| Path | Size | Class | Notes |
|---|---|---|---|
| `output/HumanoidIm/quest_distill_trackerr/student/student_0380160.pth` | 21.9MB | **A. MUST MIGRATE** | The reported result IS this file; irreplaceable |
| `output/HumanoidIm/phc_prim_quest/Humanoid.pth` | 339MB | **A. MUST MIGRATE** | Unconditionally loaded at construction (train+eval); origin untraceable in-repo (dated 2024-08-19, predates every Quest2 script, not in `phc/download_data.sh`) — treat as irreplaceable |
| `results/AvatarPoseEstimationQuest2/models/15000_G.pth` | 15.8MB | **A. MUST MIGRATE** (for training/eval to construct at all) / **E. REGENERATED BY TRAINING** (its weights, specifically, are moot once `student_checkpoint` is set — see above) | Structural dependency of the constructor; content overwritten immediately if evaluating an existing student checkpoint |
| `data/quest2_synthetic/{train,test}_seg_motion/` | part of 511GB | **B. ALREADY MIGRATED** | Per this task's framing; not re-derived here |
| `phc/phc/data/cfg/learning/distill_quest.yaml`, `phc/phc/data/cfg/env/env_quest_im_basic.yaml`, `phc/phc/data/cfg/robot/quest_humanoid.yaml`, `phc/phc/data/assets/mjcf/quest_humanoid.xml` | small | **A. MUST MIGRATE** | Git-tracked (submodule), comes with `git clone --recurse-submodules`; listed for completeness since they define the exact recipe |
| `phc/phc/learning/distill_agent.py`, `distill_agent_quest.py` | small | **A. MUST MIGRATE** | Git-tracked source; the architecture itself |
| `scripts/train_distill_quest_with_auto_recovery.sh`, `scripts/auto_eval_quest.sh`, `scripts/quest_checkpoint_scan.sh`, `scripts/evaluate_phc_metrics_quest.py` | small | **A. MUST MIGRATE** | Git-tracked; the exact commands documented above depend on these |
| `output/HumanoidIm/quest_distill_trackerr/quest_eval_0380160.json`, `checkpoint_scan.log`, `quest_distill_trackerr.log` | small | **A. MUST MIGRATE** | Only surviving *evidence* linking the checkpoint to the reported numbers; not regeneratable without a full re-run + rescan |
| `output/HumanoidIm/quest_distill_trackerr/student/student_*.pth` (the other 1041 checkpoints) | ~21.9GB | **F. OPTIONAL/HISTORICAL** | Not needed to evaluate or cite the current result; useful only if you want to re-run the full-range scan yourself |
| `output/HumanoidIm/quest_distill_trackerr/quest_eval_*.json` (other steps) | small | **F. OPTIONAL/HISTORICAL** | Scan intermediate evidence; `_0380160.json` alone is sufficient for the headline claim |
| SMPL/SMPL-H assets | — | out of scope | Already resolved per this task; confirmed not required for this pipeline regardless (see intro) |
| IsaacGym Preview 4 SDK | — | **C. REDOWNLOADABLE** | Gated NVIDIA download, not in-repo; see `README.md` §3b |
| `output/HumanoidIm/quest_distill_finetuned_backbone/` (old, pre-fix run) | 23GB | **F. OPTIONAL/HISTORICAL** | Superseded; not part of the current reported result (see prior audit turn) |

---

## Answers

**1. Exact files needed to retrain `trackerr` from scratch:**
- `output/HumanoidIm/phc_prim_quest/Humanoid.pth` (frozen teacher)
- `results/AvatarPoseEstimationQuest2/models/15000_G.pth` (backbone initialization)
- `data/quest2_synthetic/train_seg_motion/` (already migrated)
- Git-tracked: `phc/phc/learning/distill_agent{,_quest}.py`, `phc/phc/data/cfg/{learning/distill_quest.yaml,env/env_quest_im_basic.yaml,robot/quest_humanoid.yaml}`, `phc/phc/data/assets/mjcf/quest_humanoid.xml`, `scripts/train_distill_quest_with_auto_recovery.sh`
- IsaacGym Preview 4 + the `rlgpu` conda env (see `README.md` §3)
- Note: not bit-exact reproducible (`torch_deterministic=False`) — expect a similar, not identical, result.

**2. Exact files needed to immediately evaluate `student_0380160.pth`:**
- `output/HumanoidIm/quest_distill_trackerr/student/student_0380160.pth`
- `output/HumanoidIm/phc_prim_quest/Humanoid.pth` (constructor dependency, doesn't affect the number)
- `results/AvatarPoseEstimationQuest2/models/15000_G.pth` (constructor dependency, doesn't affect the number)
- `data/quest2_synthetic/test_seg_motion/` (already migrated)
- Git-tracked: `scripts/auto_eval_quest.sh`, `scripts/evaluate_phc_metrics_quest.py`, plus the same config/source files as above
- IsaacGym + `rlgpu` env for the physics rollout; `avatarposer` env for the metrics script

**3. Unresolved reproducibility gaps:**
- The exact original launch command for `quest_distill_trackerr` (num_envs, retry/threshold args) was reconstructed, not recovered verbatim — low risk (script defaults), but not independently confirmed.
- `phc_prim_quest/Humanoid.pth`'s origin/provenance is untraceable from this repo alone (predates all Quest2-specific scripts, not in `phc/download_data.sh`).
- Training is not bit-exact deterministic (`torch_deterministic=False`), so "retrain from scratch" reproduces the *recipe*, not the *exact weights*.

**4. Untracked source/config files that must be preserved:**
- None beyond what's already itemized above as class A — the entire reproduction-critical source and config set for this specific pipeline is git-tracked. The three checkpoint files and the eval JSON/logs are the only untracked (`.gitignore`d) items that are load-bearing.

STOP — documentation only, nothing trained, no source modified, nothing deleted.
