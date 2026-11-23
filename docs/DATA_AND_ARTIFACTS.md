# Data, dependencies and artifact status

This page states what the public repository contains, what you must obtain yourself, and which
experimental artifacts from the original project were **not** preserved. It also states how far
each reported result can currently be checked.

## A. Included in this repository

- **Original AvatarPoser–PHC integration code:** Quest2 dataloader and FK model, AMASS→Quest2
  fine-tuning, evaluation scripts, retargeting, diagnostics and rendering scripts (`data/`,
  `models/`, `main_*`, `scripts/`, `utils/physical_metrics.py`).
- **Research documentation:** `README.md`, `results_comparison.md`, `results_per_clip_renders.md`,
  `QUEST2_TRAINING_REPRODUCTION.md`, `QUEST2_BLENDER_REPRODUCTION.md`, the analysis and diagnosis
  reports under `scripts/**/*.md`, and `Quest2_Research_Notebook.ipynb`.
- **Aggregate results and per-clip metric tables:** `results/baselines/*.json`,
  `baselines/*.json`, `scripts/**/*_metrics.json`, `scripts/**/per_clip*.json`,
  `scripts/dataset_mesh_gpen_*.{json,csv}`, and diagnostic summaries. These hold scalar metrics
  and clip identifiers, no motion data.
- **Figures** (`scripts/*.png`, notebook outputs) and raw experiment logs (`*.log` at the root).
- **Upstream AvatarPoser material** inherited from the fork: code, `data_split/` and figures,
  distributed by upstream under MIT. The pretrained checkpoint is **not** included (see B).

## B. External dependencies (obtain separately)

| Dependency | Source | Version used | Notes |
|---|---|---|---|
| AvatarPoser (upstream) | https://github.com/eth-siplab/AvatarPoser | `0f97cb2` (2025-02-20), the base of this fork | MIT (code) |
| AvatarPoser pretrained model `model_zoo/avatarposer.pth` | Official release: https://github.com/eth-siplab/AvatarPoser/releases/download/v1.0/avatarposer.pth | release `v1.0` (2022-07-29), sha256 `02f8888e807579771a7e9959160d1461b62b7112a1e99557aae76bedc46f9e76`, 16,515,227 bytes | Trained on AMASS (non-commercial research licence; training networks for commercial use is prohibited). Upstream's MIT statement does not explicitly cover the weights, so they are not redistributed here. Used by `options/test_avatarposer.json`, `options/train_avatarposer_quest2_finetune.json` and `scripts/train_stage_r_with_early_stop.sh` |
| PHC fork (submodule `phc/`) | Public release: `https://github.com/thelastlaughstandstrong/PHC-AvatarPoser` | `413516d081f0ba6681e505c120713955c921bff6` (`publication-v2`) | A clean copy of the private research fork. See [`PHC_SUBMODULE.md`](PHC_SUBMODULE.md) |
| PHC (upstream) | https://github.com/ZhengyiLuo/PHC | base `846988d433ce1f341e85ac6fbd2cd51911bb3341` | BSD 3-Clause Clear |
| SimXR | https://github.com/ZhengyiLuo/SimXR | `39081b831aa1c1e52a4a974fb78a1ed93f7580c2` | **No license declared.** Required for all Quest2 work (rig, configs, motion library, baseline) |
| Isaac Gym Preview 4 | https://developer.nvidia.com/isaac-gym | Preview 4 | NVIDIA developer login required |
| SMPL-H / DMPL body models | https://mano.is.tue.mpg.de/ (SMPL-H), https://smpl.is.tue.mpg.de/ (DMPL) | — | Registration required. Needed for AMASS preprocessing and mesh-GPen diagnostics; see `docs/MIGRATION.md` §5b for which genders each script loads |
| SMPL (PHC's SMPL humanoid) | https://smpl.is.tue.mpg.de/ | v1.1.0 | Only for the AMASS/SMPL pipeline |
| PHC teacher checkpoints | PHC: `phc/download_data.sh`; SimXR: `download_data.sh` (`phc_prim_quest`) | — | Not redistributed here |
| Conda environments | `setup/environment-avatarposer.yml`, `setup/environment-rlgpu.yml` | Python 3.8, PyTorch 2.4.1 + CUDA 12.1 | Exported from the original machine |

## C. Datasets (obtain independently)

### AMASS (AMASS/SMPL pipeline)

1. Register and download from https://amass.is.tue.mpg.de/. The splits in `data_split/` use the
   **CMU**, **BioMotionLab_NTroje** and **MPI_HDM05** subsets.
2. Put them under `amass/<subset>/` and the SMPL-H/DMPL models under
   `support_data/body_models/{smplh,dmpls}/<gender>/model.npz`, the paths `prepare_data.py`
   expects.
3. Run `python prepare_data.py` (upstream AvatarPoser preprocessing). It writes
   `data_cameraready_fps60/<subset>/{train,test}/`.
4. Run `python scripts/build_phc_splits.py`, which writes `phc/sample_data/phc_train.pkl` (about
   4,811 sequences) and `phc_test.pkl` (536 sequences) for PHC.

### SimXR synthetic Quest2 data (Quest2 pipeline)

1. Download the **processed synthetic sequences** linked from SimXR's README, section "Data":
   [Train](https://drive.google.com/drive/folders/1jMld_d6JmyNkq0w1mBWH5nSts9tsO47b) /
   [Test](https://drive.google.com/drive/folders/1RACtEleG5saxvjyt1KRe-p0jA4WmAGMS). Use is
   subject to SimXR's terms. `scripts/download_simxr_data.sh` is the resumable downloader used
   originally. Its paths and conda env are hardcoded for the original machine; edit `REPO`,
   `GDOWN` and `PYTHON` first.
2. SimXR documents splitting with `python scripts/data_process/split_data.py --data_dir <path>`,
   run in the SimXR repository.
3. This project trained and evaluated on per-clip `data/quest2_synthetic/{train,test}_seg_motion/`
   pickles: 4,097 train and 1,072 test clips, the same counts as SimXR's paper. They were derived
   from `*_seg/` folders by `scripts/extract_motion_only.py`, which drops vision fields and
   re-slices each segment. **The exact command that produced the `*_seg/` folders is not recorded
   in this repository.** Verify the clip counts above before comparing results.
4. The split is by take, not by subject. 30.4% of test clips come from subjects seen in training
   (see `results_comparison.md`, "Dataset caveats").

## D. Artifacts that were not preserved

The original workstation was lost. These artifacts existed there, were never in Git, and are
**not recoverable**:

- all trained student checkpoints, including the reported `trackerr` `student_0380160.pth`;
- the Quest2 AvatarPoser backbones (`results/AvatarPoseEstimationQuest2*/`);
- all rollout dumps (`phc_positions_for_eval*.pkl`), and the evaluation JSONs, training logs and
  checkpoint-scan logs of every run not listed below. **No output of the `trackerr` run was ever
  committed;**
- the SimXR checkout, with its no-vision baseline checkpoints and dumps;
- AMASS-student evaluation outputs;
- uncommitted working-tree code, including the AMASS `v8` variant.

Git history still holds `output/HumanoidIm/` material for some earlier Quest2 runs, committed
2026-07-29 and removed from the tree 2026-08-14. It covers checkpoint-scan eval JSONs (aggregate
metrics plus clip names), training/recovery logs, Hydra configs and TensorBoard scalars for
`quest_distill_option2_scratch`, `quest_distill_option3_actionloss`, `quest_distill_explicit_pose`
and several short `quest_distill_{v1,v2,diag*,bisect*,verify}` runs.

### What each reported result rests on

| Result | Evidence available | Can be re-run? |
|---|---|---|
| Phase A (Quest2 kinematic), naive PHC baseline, Mesh-GPen rows | **Tracked artifacts:** `results/baselines/*.json`, `scripts/dataset_mesh_gpen_aggregate.json` | Yes, with the SimXR data, SimXR files and the teacher; requires retraining Phase A |
| Pre-fix Quest2 runs (Option2/3, `explicit_pose`) | Eval JSONs and logs in Git history (`output/HumanoidIm/quest_distill_*`, until 2026-08-14) | Retrain from code |
| `trackerr` (headline Quest2 student) | **Contemporaneous notes only:** `results_comparison.md`, `QUEST2_TRAINING_REPRODUCTION.md` (which quotes the eval JSON to full precision). Checkpoint and raw outputs lost | Retrain from code (not bit-exact; selection was on the test set) |
| SimXR no-vision baseline | Notes only: `scripts/simxr_novision/README.md`, `simxr_ref/README.md` | Retrain in SimXR with `scripts/simxr_novision/` plus the instrumentation |
| AMASS distillation rows (`v7`, `v7_validate`, …) | Notes only (`results_comparison.md`) | Retrain from code. AMASS `v8` code was not preserved |
| Root `*.log` experiments (Stage R, critic warm-up) | The logs themselves | Retrain from code |

**Nothing in this repository has been re-run since recovery.** No training or evaluation result
has been independently reproduced.

## E. Excluded from the public release

These were tracked in the original private repository and are **not** distributed. Rights to
redistribute data derived from SimXR or AMASS could not be established. The excluded items are:

| Excluded | Content | How to regenerate |
|---|---|---|
| `simxr_ref/phc/**` (2 files) | Copied SimXR source | `scripts/simxr_novision/apply_eval_instrumentation.py` on an official checkout (see `simxr_ref/README.md`) |
| `scripts/ablation_footweight/evals/footw1_*_actions.npz` (8) | Per-frame 72-DoF student actions on 90 clips (a validation slice of the SimXR test split) | `scripts/ablation_footweight/eval_checkpoint.py`. Needs the lost `footw1` checkpoints |
| `scripts/diag_fb_multiclip/mc_fb.npz`, `mc_naive.npz` | Per-frame student/teacher actions on 56 clips | `scripts/diag_fb_multiclip/log_actions_multi.py`. Needs the lost checkpoints |
| `scripts/diag_simxr_ankle/mc_simxr.npz` | Per-frame SimXR-policy actions on 56 clips | `scripts/diag_simxr_ankle/log_simxr_actions.py`. Needs the lost SimXR checkpoint |
| `scripts/quest_smpl_sequence_fit.json`, `scripts/quest_smpl_sequence_fit_walking.json` | Per-frame SMPL fits (about 426 frames) of SimXR test clips | `scripts/retarget_quest_to_smpl_sequence.py --motion_file <clip>` |
| `scripts/quest_smpl_single_frame_fit.json` | One full-body SMPL pose fitted to a SimXR frame | `scripts/retarget_quest_to_smpl.py` |
| `support_data/github_data/amass_sample.npz`, `dmpl_sample.npz` | Raw AMASS sample sequences (poses, markers) shipped by upstream AvatarPoser | Upstream AvatarPoser repository, or AMASS |
| `model_zoo/avatarposer.pth` | Upstream pretrained AvatarPoser weights (trained on AMASS) | Official release v1.0 (see B); identical file, sha256 above |

The aggregate results computed from these files remain in the reports next to them (`*.md`,
`*_metrics.json`, `per_clip*.json`). Scripts that read these files need them regenerated first:
`analyze.py`, `mechanism.py`, `analyze_simxr.py`, `score_arms.py`, `charstudy_convert.py`,
`convert_step5_to_phc.py`, `preflight_phc_load.py`, `test_smpl_rest_offsets.py` and
`validate_smpl_fk_paths.py`.
