# The `phc/` submodule

`phc/` is a modified fork of [PHC](https://github.com/ZhengyiLuo/PHC). It contains the
distillation agents (`phc/learning/distill_agent.py`, `distill_agent_quest.py`), the Quest2
configs and the evaluation dump used by every PHC experiment here.

The public release points the submodule at **`PHC-AvatarPoser`**
(`https://github.com/thelastlaughstandstrong/PHC-AvatarPoser`). This is a clean-publication copy
of the private research fork:

- Six Quest2 files derived from SimXR are **not distributed**, in the tree or in history.
  `phc/scripts/install_simxr_quest_files.py` rebuilds them byte-for-byte from your own SimXR
  checkout.
- Commits that once contained those files were recreated, so they have new SHAs. All other
  commits keep their original SHAs.
- Three commits on top add machine-path fixes, optional Quest2 imports and a fork notice. The
  fork notice and `NOTICE` cover licensing and attribution.

Pinned publication commit: **`413516d081f0ba6681e505c120713955c921bff6`** (tag `publication-v2`,
branch `direct-pd-action`).

## Research commit ↔ publication commit

Results were produced at private commit `759f0904`. Superproject commits made before the public
release pin private SHAs. Use this table, or the `research-pin/<sha8>` tags in `PHC-AvatarPoser`.

| Research commit (private) | Date | Publication commit | |
|---|---|---|---|
| `846988d433ce` | 2025-08-21 | `846988d433ce` | upstream PHC base, unchanged |
| `9f3245e11760` | 2026-06-30 | `9f3245e11760` | unchanged |
| `7b4806b79d08` | 2026-06-30 | `7b4806b79d08` | unchanged |
| `cf5cfa5c7179` | 2026-07-03 | `cf5cfa5c7179` | unchanged |
| `5c14f502fd9c` | 2026-07-07 | `5c14f502fd9c` | unchanged |
| `20a8cac68c97` | 2026-07-29 | `56d0d9142d45` | recreated without SimXR files |
| `f7b24a232b97` | 2026-07-31 | `1bffe03cf6a6` | recreated without SimXR files |
| `c7a62e0402de` | 2026-08-13 | `c5c657927863` | recreated without SimXR files |
| `759f09048f04` | 2026-09-10 | `e31fdfeddb24` | recreated; **code state of the reported results** |
| — | 2026-11-23 | `413516d081f0` | publication tip (pinned) |

After installing the SimXR files, `research-pin/759f0904` rebuilds to a tree identical to the
private commit. This was verified locally before release.

## Setup order

```bash
# 1. Clone with the submodule
git clone --recurse-submodules <this repository> AvatarPoser && cd AvatarPoser

# 2. Obtain SimXR at the pinned revision (separately; SimXR's own terms apply)
git clone https://github.com/ZhengyiLuo/SimXR.git /path/to/SimXR
git -C /path/to/SimXR checkout 39081b831aa1c1e52a4a974fb78a1ed93f7580c2

# 3. Rebuild the SimXR-derived Quest2 files inside phc/ (hash-verified; files are git-ignored)
python phc/scripts/install_simxr_quest_files.py --repo phc --simxr /path/to/SimXR --check
python phc/scripts/install_simxr_quest_files.py --repo phc --simxr /path/to/SimXR

# 4. Environments, Isaac Gym, body models and datasets: see docs/DATA_AND_ARTIFACTS.md and docs/MIGRATION.md
```

To reproduce the SimXR no-vision **baseline** itself, apply this repository's evaluation
instrumentation to the same SimXR checkout (`simxr_ref/README.md`). It is independent of step 3.

## What needs the SimXR files

Without step 3, the SMPL/AMASS pipeline and upstream PHC tasks work. Quest2 classes raise an
error explaining how to install the files. Everything Quest2 needs them:

- `phc/`:
  - `robot=quest_humanoid`, `env=env_quest_im_basic`, `learning=im_quest` / `distill_quest`;
  - `HumanoidQuest` and its task wrappers;
  - `MotionLibQuest`.
- Main repository:
  - `models/quest_body_model.py`, which reads `phc/phc/data/assets/mjcf/quest_humanoid.xml` and
    is used by Quest2 AvatarPoser training and testing (`main_test_avatarposer_quest2.py`,
    `options/train_avatarposer_quest2*.json`);
  - `scripts/auto_eval_quest.sh`, `quest_checkpoint_scan.sh`,
    `train_distill_quest_with_{auto_recovery,periodic_eval}.sh`;
  - `scripts/evaluate_phc_metrics_quest.py`, `score_naive_baseline_vs_gt.py`,
    `convert_avatarposer_quest2_to_phc.py`;
  - `scripts/mesh_gpen_{quest,dataset}.py`, `run_mesh_gpen_study.sh`, `presim_gpen_aggregate.py`;
  - `scripts/retarget_quest_to_smpl*.py`, `calibrate_quest_to_smpl_frame.py`,
    `inspect_quest_smpl_skeletons.py`, `verify_quest_rotation_convention.py`,
    `export_blender_motion_quest.py`;
  - `scripts/smoke_test_humanoid_*quest*.py`;
  - `scripts/ablation_footweight/`, `scripts/diag_fb_*/`, `scripts/diag_simxr_ankle/`;
  - `scripts/run_stage_r_collapse_diagnostic.sh`.

Most shell scripts also hardcode the original machine's paths (`/home/sia-irl/...`,
`/home/sia-irl/anaconda3/bin/conda`). Adjust them first; see `docs/MIGRATION.md` §6.

## Fresh-clone check (to run after both repositories are public)

1. `git clone --recurse-submodules` with no credentials. Confirm
   `git -C phc rev-parse HEAD == 413516d081f0ba6681e505c120713955c921bff6`.
2. Before installing: `ls phc/phc/env/tasks/humanoid_quest.py` must fail.
   `python -c "import ast; ast.parse(open('phc/phc/utils/quest_optional.py').read())"` must pass.
3. Run the installer with `--check`, then without it. All 6 files should report `installed`.
4. On a machine with Isaac Gym: `python scripts/smoke_test_humanoid_im_quest.py` and an AMASS
   `learning=distill` smoke run. Neither has been executed for the public release.
