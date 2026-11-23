> **About this file.** This is the root `README.md` as it stood at commit `54158c8` (2026-09-18),
> moved here unchanged on 2026-11-23 when the root README was rewritten as a research overview.
> It is a machine-specific setup and migration log. Paths such as `/home/sia-irl/AvatarPoser`,
> disk-usage figures and the conda-environment inventory describe the original workstation, not
> requirements. Relative references such as `results_comparison.md` and `setup/*.yml` are relative
> to the **repository root**, not to `docs/`.

# AvatarPoser + PHC + Quest2/SimXR — Machine Setup & Migration Guide

This repo builds on [AvatarPoser](https://github.com/eth-siplab/AvatarPoser) (ECCV 2022, kinematic
HMD/controller pose tracking) and [PHC](https://github.com/ZhengyiLuo/PHC) (physics-based humanoid
control in Isaac Gym), with a large amount of custom work on top: a PHC-based "distillation"
pipeline that turns AvatarPoser's kinematic output into physically-simulated motion, and a
Quest2/SimXR track that retargets real headset+controller recordings onto SMPL and drives the same
PHC pipeline from real (not synthetic AMASS) input.

This document exists for one purpose: **moving this whole setup to a different laptop** without
losing anything or having to rediscover how the pieces fit together. It assumes the new machine
starts from nothing (fresh Ubuntu install) and walks through every step in order. Research
findings, experiment results, and per-topic details live in the various `*.md` / `results_*.md`
files scattered through the repo (notably `results_comparison.md`) — this file is only about
**getting the environment and data working again**, not what the experiments found.

Everything below was captured by inspecting the actual current machine on 2026-09-18. Sizes,
paths, and environment names are what's *really* there today, not assumptions.

---

## 0. Read this first — the current machine has known problems

Before copying anything off the old laptop, be aware of things sitting in `data/`/`output/` right
now:

1. **The disk is tight: 74GB free out of 1.9TB (96% used) as of 2026-09-18.** There is not enough
   room to make a second local copy of the big datasets. Don't try to `zip`/`tar` everything into a
   staging archive on this machine — plan a direct transfer instead (see §4). (This was as low as
   111MB free earlier the same day — an in-progress GUI archive operation had been consuming disk
   space unattended and was interrupted; see git/session history if you need the details. The
   corrupted `data/quest2_synthetic.zip` byproduct of that has already been deleted.)
2. **Three duplicate copies of the same directory exist**: `data/quest_smpl_retarget/`,
   `data/quest_smpl_retarget (2)/`, and `data/quest_smpl_retarget (3)/` are byte-identical
   (verified with `diff -rq`), 2.8GB each — classic "extracted the same zip twice more" leftovers.
   Only one copy needs to survive:
   ```bash
   rm -rf "/home/sia-irl/AvatarPoser/data/quest_smpl_retarget (2)" \
          "/home/sia-irl/AvatarPoser/data/quest_smpl_retarget (3)"   # reclaims 5.6GB
   ```
   (`data/quest_smpl_retarget.zip`, 2.9GB / 1394 files, is a *valid* zip — that one's fine to keep
   or delete once you've confirmed the extracted folder transferred correctly.)

This remaining item reclaims ~5.6GB and removes confusion about which copy is authoritative.
**None of this has been deleted yet — run this command yourself (or ask me to) before
proceeding**, they're just flagged here so the migration doesn't propagate the mess.

---

## 1. What actually needs to move

### 1a. Git-tracked (comes for free with `git clone`)
All source code, configs, small result summaries (`results/baselines/*.json`), and this README.
The `phc/` directory is a **git submodule** pointing at `https://github.com/ZhengyiLuo/PHC.git`,
currently pinned at commit `759f0904` on branch `direct-pd-action` — cloning with
`--recurse-submodules` reproduces this exactly.

### 1b. Local data that is `.gitignore`d and must be physically copied
Nothing below travels with `git clone`. This is the actual payload of the migration. Sizes are
current, measured on the old machine:

| Path | Size | What it is |
|---|---|---|
| `data/quest2_synthetic/` | 511GB | Real Quest2/SimXR HMD+controller recordings + rendered vision frames (train/test, `_seg` = segmented variants). This is the core dataset for the Quest2 track. |
| `output/` | 204GB | All PHC training run checkpoints/logs (`HumanoidIm/<exp_name>/...`), renders, sim states. **These are the actual trained models your experiments reference** (e.g. `student_0260000.pth`, `finetuned_backbone@440k`) — not reproducible without re-running training. |
| `amass/` | 11GB | Raw AMASS mocap data (kinematic ground truth for the non-Quest track). |
| `data_cameraready_fps60/` | 5.4GB | A specific fps60 dataset snapshot. |
| `results/` | 1.2GB | Non-baseline eval artifacts (the small `baselines/*.json` subset is already git-tracked; the rest isn't). |
| `sample_data/` | 3.2GB | Preprocessed PHC-format pickles (`phc_train.pkl`, `phc_test.pkl`, `avatarposer_all.pkl`) built from AMASS by this repo's own conversion scripts. Regeneratable from `amass/` + `data_split/`, but copying is much faster than regenerating. |
| `data/quest_smpl_retarget/` | 2.8GB | Quest→SMPL retargeting pipeline outputs (keep **one** copy — see §0). |
| `phc/data/` | 708MB | Almost entirely `phc/data/smpl/*.pkl` (classic SMPL, all 3 genders). **Only needed for the legacy AMASS/SMPL track, not the current Quest2 pipeline** — verified in §5b. |
| `phc/wandb/` | 65MB | Offline W&B run logs from PHC training (see §3c — wandb runs offline, no account needed). |
| `results_decoder_finetune/` | 174MB | Decoder finetuning results. |
| `support_data/body_models/` | ~500MB | SMPL-H + DMPL body model files. **Only the `neutral/` subfolder is needed for the current Quest2 pipeline** (its optional Mesh-GPen diagnostic); `male/`+`female/` only matter for the legacy AMASS track — verified breakdown in §5b. Gated download either way. |
| `model_zoo/*.pth` | 79MB | This project's own trained checkpoints (5 files) — `.pth` files here are gitignored. |
| `data_split/` | 256KB | Train/test split text files (small enough it could be git-tracked, currently isn't). |

**Minimum viable transfer** (code + data + final checkpoints, skipping the 204GB `output/` run
history): **~540GB**. **Full transfer including all training runs**: **~745GB**. Plan your
transfer medium (external drive / direct network copy) accordingly — see §4.

### 1c. Things that live *outside* this repo and also need to move
- **IsaacGym Preview 4** — `/home/sia-irl/isaacgym_workspace/isaacgym/` (the extracted SDK) and/or
  `/home/sia-irl/Downloads/IsaacGym_Preview_4_Package.tar.gz` (the original archive). Not
  pip-installable from PyPI — see §3b.
- Nothing else outside the repo was found to be load-bearing (no stray `.env`, credentials file,
  or `.netrc` — this project doesn't need any API keys).

---

## 2. Hardware / OS baseline (what the old machine has, for reference)

- Ubuntu 24.04.4 LTS
- NVIDIA GPU with CUDA 12.x driver (old machine: RTX 4090 Laptop, 16GB VRAM, driver 535.309.01,
  `nvcc` 12.0). Isaac Gym + PyTorch CUDA wheels here are built for **CUDA 12.1**, so the new
  machine's driver needs to support at least that (any driver ≥ 530 does).
- ~750GB+ free disk space is genuinely needed if you want the full dataset + all training outputs
  present (see the table above).

---

## 3. Setting up the new laptop — environment

### 3a. System-level prerequisites
```bash
# NVIDIA driver + CUDA (use Ubuntu's driver installer or NVIDIA's .run installer; either way,
# confirm afterwards):
nvidia-smi          # should show a driver supporting CUDA 12.x
nvcc --version       # optional; PyTorch ships its own CUDA runtime via pip wheels anyway

# Miniconda/Anaconda (whichever you prefer — commands below assume `conda` is on PATH)
```

### 3b. Get IsaacGym Preview 4 onto the new machine
IsaacGym is **not on PyPI**; it's a manual download gated behind an NVIDIA developer login
(https://developer.nvidia.com/isaac-gym — sign in, download "Preview 4"). The old machine already
has it at `/home/sia-irl/Downloads/IsaacGym_Preview_4_Package.tar.gz` — **copy that tarball across
instead of re-downloading**, it saves dealing with the login flow again:
```bash
mkdir -p ~/isaacgym_workspace
tar -xzf IsaacGym_Preview_4_Package.tar.gz -C ~/isaacgym_workspace
# result: ~/isaacgym_workspace/isaacgym/
```
Don't run its `create_conda_env_rlgpu.sh` yet — the conda envs below already have everything that
script would install except IsaacGym itself, which gets installed as an editable package into each
env individually (§3c).

### 3c. Recreate the two conda environments this project actually uses
Only two environments matter for running this repo's own code: **`avatarposer`** (kinematic
AvatarPoser training/eval, no GPU sim needed) and **`rlgpu`** (PHC/Isaac Gym physics training —
also used for the `isaac` and `simxr` steps of this project, which share the same stack). Both are
**Python 3.8.20 + PyTorch 2.4.1 built for CUDA 12.1**. Exact exports of both, captured from the
old machine, are checked into this repo at `setup/environment-avatarposer.yml` and
`setup/environment-rlgpu.yml`:
```bash
conda env create -f setup/environment-avatarposer.yml   # creates env "avatarposer"
conda env create -f setup/environment-rlgpu.yml          # creates env "rlgpu"
```
These YAML exports deliberately do **not** include IsaacGym (conda correctly detected it as a
local editable install and excluded it) — install it manually into **both** envs after they're
created:
```bash
cd ~/isaacgym_workspace/isaacgym/python
conda run -n avatarposer pip install -e .
conda run -n rlgpu       pip install -e .
```
Verify each env:
```bash
conda run -n avatarposer python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
conda run -n rlgpu       python -c "import isaacgym, torch; print('isaacgym OK', torch.cuda.is_available())"
```
Both `pip install` runs for the YAML files above will also resolve a handful of `git+https://...`
packages (custom `smplx`, `SMPLSim`, `chumpy` forks) — these need network access during env
creation, nothing gated, just make sure the new machine has internet access at this step.

No W&B account is needed: every script in this repo that touches PHC sets `WANDB_MODE=offline`
before launching training, so logging stays entirely local (`phc/wandb/`).

### 3d. Clone the repo (with the PHC submodule)
```bash
git clone --recurse-submodules git@github.com:thelastlaughstandstrong/AvatarPoser-with-PHC-Implementation.git AvatarPoser
```
(SSH clone — make sure your GitHub SSH key is set up on the new machine first, e.g.
`ssh-add ~/.ssh/id_ed25519` and the corresponding public key added to your GitHub account. If you'd
rather not deal with SSH keys on the new machine, clone over HTTPS instead using the same repo
path.)

If you're instead copying the whole working directory across wholesale (see §4) rather than doing
a fresh `git clone`, the submodule is already checked out inside it and no extra step is needed —
just note the pinned commit above in case you ever need to re-sync it (`git submodule update`).

---

## 4. Moving the data across

Given the size (§1b) and that the old machine only has 94GB free (§0), **do not** try to stage
everything into a single archive on the old laptop first. Two options, in order of preference:

**Option A — direct machine-to-machine transfer** (fastest, no intermediate copy, resumable):
```bash
# from the NEW machine, pulling from the OLD one (adjust user@host):
rsync -avh --progress \
  --exclude='data/quest_smpl_retarget (2)' \
  --exclude='data/quest_smpl_retarget (3)' \
  --exclude='data/quest2_synthetic.zip' \
  sia-irl@<old-machine-ip>:/home/sia-irl/AvatarPoser/ /home/sia-irl/AvatarPoser/
```
`rsync` resumes cleanly if interrupted — for 500GB+ over a home network this matters. Run it once
for the repo itself, and separately (or in the same pass) for anything outside the repo you're
bringing across (`~/isaacgym_workspace/`, `~/Downloads/IsaacGym_Preview_4_Package.tar.gz`).

**Option B — external drive**: needs ≥1TB free to be comfortable (§1b totals ~745GB before
overhead). Copy with `rsync -avh` (not plain `cp -r`) so permissions/symlinks survive and it's
resumable if the drive disconnects mid-copy — note `phc/output` and `phc/sample_data` are
**symlinks** back into the repo root, so a naive archive tool that doesn't preserve symlinks (or
that dereferences them) will silently duplicate hundreds of GB. `rsync -avh` and `tar` both handle
this correctly by default; double-check whatever tool you use does too.

Either way, put the new machine's copy at the **same absolute path**: `/home/sia-irl/AvatarPoser`.
See §6 for why that matters.

---

## 5. Datasets that come from outside this project

### 5a. AMASS
`amass/` (11GB) holds AMASS mocap sequences. AMASS requires a free account at
https://amass.is.tue.mpg.de/ to download — there's no scriptable URL. **Copying the already-agreed
folder across (§4) is faster than re-registering and re-downloading on the new machine.** If for
some reason you need the original source, that's where it comes from.

### 5b. SMPL / SMPL-H body models — verified per-pipeline, not "required by everything"

There are **two separate body-model asset families** in this repo, and — verified by tracing the
actual Ground Penetration (GPen) code path, not by filename — **neither is needed to reproduce the
current Quest2 headline results** (`trackerr@380k`, the numbers in `results_comparison.md`'s main
table). They're only needed for specific optional/legacy pieces:

**`phc/data/smpl/SMPL_{NEUTRAL,MALE,FEMALE}.pkl`** (classic SMPL, ~742MB, all 3 genders) — loaded
only when PHC's `humanoid_type` config is `smpl`/`smplh`/`smplx` (gated in
`phc/phc/env/tasks/humanoid.py` line 777 onward, and in `phc/phc/utils/motion_lib_smpl.py`).
`phc/data/cfg/robot/quest_humanoid.yaml` sets `humanoid_type: quest`, which never matches that
gate — **the Quest2 pipeline (student inference, Isaac Gym simulation, and
`scripts/evaluate_phc_metrics_quest.py`'s headline metrics) never touches this file.** It's a
dependency of the separate, older **AMASS/SMPL track** only (`ap_phc_distill_*` experiments, Phase
A, and the AMASS data-processing scripts under `phc/scripts/data_process/`), where all three
genders are genuinely loaded (see `motion_lib_smpl.py:56-64`, `humanoid_amp.py:127-131`).

**`support_data/body_models/{smplh,dmpls}/{male,female,neutral}/model.npz`** (~500MB for all 3
genders × 2 model types) — used by two *different, optional* mesh-based Ground Penetration
diagnostics, each pinned to one specific gender file, not "all of them":
- **Quest2's own mesh-GPen** (`scripts/mesh_gpen_quest.py`, lines 52-53) — the script that produces
  the "Mesh GPen Mean/Rate" rows in `results_comparison.md` for `trackerr`/Finetuned Backbone/Naive
  PHC/SimXR — hardcodes `support_data/body_models/smplh/**neutral**/model.npz` and
  `dmpls/**neutral**/model.npz` only. It never opens the `male/` or `female/` subfolders. This
  script is **simulator-independent**: it re-IKs stored joint-position dumps
  (`phc_positions_for_eval_*.pkl`) or ground truth onto a neutral SMPL-H mesh; it is not part of
  the student-inference or physics-sim path.
- **The AMASS-track's mesh-GPen** (`scripts/evaluate_phc_metrics.py`, line 363-366) hardcodes
  `body_models/smplh/**male**/model.npz` / `dmpls/**male**/model.npz` instead — a different,
  unrelated script for the separate AMASS-track table further down in `results_comparison.md`.
- Neither script ever opens `female/`. It's only reachable via general AMASS
  motion-processing code paths, not by either GPen evaluator.

**Bottom line — what you actually need, by use case:**

| Use case | Assets required | Needed for the current headline results? |
|---|---|---|
| Student inference (running a trained `.pth`) | None | — |
| Quest2 Isaac Gym physics simulation + headline metrics (success/MPJPE/MPJRE/jitter/foot-skate) | None — `evaluate_phc_metrics_quest.py`'s own `compute_gpen` is pure joint-position math, no mesh, no body model | **Yes, and needs nothing from this section** |
| Quest2 Mesh-GPen diagnostic (the "Mesh GPen" table rows) | `support_data/body_models/smplh/neutral/model.npz` + `dmpls/neutral/model.npz` **only** | Yes, for those specific rows only |
| AMASS-track training/eval (separate, legacy pipeline) | `phc/data/smpl/SMPL_{NEUTRAL,MALE,FEMALE}.pkl` (all 3) + `support_data/body_models/smplh/male/model.npz` + `dmpls/male/model.npz` | No — not part of the current Quest2 report |
| Blender rendering | See below | Optional, cosmetic only |

If you're migrating for the current Quest2 report only, you can skip `phc/data/smpl/` entirely and
keep just the `neutral/` subfolder of `support_data/body_models/{smplh,dmpls}/`. Both families are
still gated academic downloads if you ever need to re-fetch them fresh
(https://mano.is.tue.mpg.de/ for SMPL/SMPL-H, https://smpl.is.tue.mpg.de/ for DMPL — free account +
manual license acceptance, no scriptable URL) — copying the existing folders (§4) is faster than
re-registering.

**Blender rendering** — two unrelated pipelines, both optional/cosmetic, neither gates the reported
numbers:
- SimXR repo's `scripts/vis/render_bvh_blender.py` imports a BVH (skeleton/armature only, no mesh)
  and draws simple procedural primitives for joints — **no body model of any kind**.
- This repo's `scripts/blender_render_addon.py` (documented in `scripts/BLENDER_EXPORT_README.md`)
  poses the **official SMPL-X Blender addon's** rig/mesh — a body-model asset that lives inside
  Blender's own addon installation, **not in this repo's file tree at all** and, per that doc, not
  currently installed even on the old machine (`smplh_model_20260511.blend` "is not installed").
  This is a separate, third asset family from both `phc/data/smpl/` and
  `support_data/body_models/smplh/` — if you want this specific rendering path on the new machine,
  you'll need to install the SMPL-X addon in Blender there from scratch; it was never fully set up
  on the old one either.

### 5c. PHC's own pretrained checkpoints and AMASS-derived sample data
PHC (the submodule) ships its own download script for its base pretrained humanoid controllers and
filtered-AMASS sample pickles, fetched via Google Drive with `gdown`:
```bash
conda run -n rlgpu pip install gdown   # or use the old machine's dedicated `gdown_dl` env
bash phc/download_data.sh
```
This is independent of this project's own `sample_data/`, `data/`, and `model_zoo/` — it populates
`phc/sample_data/` and `phc/output/HumanoidIm/phc_*` with PHC's *own* released checkpoints
(distinct from the checkpoints this project trained itself, which live in the top-level `output/`
you copied in §4). Google Drive rate-limits anonymous downloads per file — if a `gdown` call stalls
with "too many people have requested this file today," it's a transient Drive-side block, not a
bug; retry later. (This exact failure mode, and a `--folder`-based workaround for it, is documented
in `scripts/download_simxr_data.sh` if you ever need to re-pull the SimXR Quest2 dataset itself —
that raw dataset is separate from the already-processed `data/quest2_synthetic/` you're copying in
§4 and shouldn't be needed unless you're rebuilding that processing pipeline from scratch.)

---

## 6. The hardcoded-path problem

Several shell scripts under `scripts/` hardcode the absolute path `/home/sia-irl/AvatarPoser`
(52 references across the repo as of this writing — e.g. `scripts/auto_eval.sh` sets
`REPO=/home/sia-irl/AvatarPoser` directly), and also hardcode
`/home/sia-irl/anaconda3/bin/conda` as the conda binary path.

**Simplest fix**: keep the same username (`sia-irl`) and the same repo location
(`/home/sia-irl/AvatarPoser`) and same anaconda install path (`~/anaconda3`) on the new machine.
Nothing needs to change.

**If the new machine must use a different username or path**, find and fix every reference before
relying on these scripts:
```bash
grep -rl "/home/sia-irl" --include="*.sh" --include="*.py" . \
  | grep -vE '^\./(data|output|amass|results|support_data|phc/(data|wandb|output))/'
# then, once you've confirmed the new path:
grep -rl "/home/sia-irl" --include="*.sh" . | xargs sed -i 's|/home/sia-irl|/home/<new-user>|g'
```
(The exclusions above skip data directories, where the same string could coincidentally appear
inside dataset content rather than as a path a script relies on.)

---

## 7. Verifying the migration worked

Two smoke tests, cheapest first:

```bash
# 1. Kinematic AvatarPoser path (no GPU sim, no IsaacGym needed — just torch + the data pickles)
conda run -n avatarposer python main_test_avatarposer.py -opt options/test_avatarposer.json

# 2. PHC + IsaacGym physics path (confirms IsaacGym install + GPU sim actually work)
conda run -n rlgpu python scripts/smoke_test_humanoid_im_quest.py
```
If both run without import errors or CUDA errors, the environment side of the migration is sound.
To confirm the *data* side, re-run an existing evaluation against a checkpoint you know the answer
to (any exp_name/step referenced in `results_comparison.md` or `results/baselines/*.json`) and
check the numbers match:
```bash
bash scripts/auto_eval.sh <exp_name> <checkpoint_step>
```

---

## 8. Conda environments on the old machine — what to actually bring

The old machine has 9 conda environments; only two belong to this project:

| Env | Bring it? | Why |
|---|---|---|
| `avatarposer` | **Yes** | Kinematic AvatarPoser train/eval. Recreate via `setup/environment-avatarposer.yml`. |
| `rlgpu` | **Yes** | PHC + Isaac Gym physics training/eval (also covers what the `isaac`/`simxr` steps of this project need — same stack). Recreate via `setup/environment-rlgpu.yml`. |
| `isaac` | No | Same package set as `rlgpu` (Python 3.8.20, torch 2.4.1+cu121) but not referenced by any script in this repo — looks like an earlier/parallel env, superseded by `rlgpu`. |
| `simxr` | No | Same package set again; not referenced by any script here either — this project's SimXR work runs under `rlgpu`. |
| `gdown_dl` | Optional | A minimal Python 3.12 env with just `gdown`, used for bulk Google-Drive downloads (§5c). Trivial to recreate with `conda create -n gdown_dl python=3.12 && pip install gdown` instead of migrating it. |
| `colmap_env`, `dexgrasp`, `dexgraspnet`, `sugar` | No | Unrelated to this project. |

---

## 9. License

This project extends [AvatarPoser](https://github.com/eth-siplab/AvatarPoser) (MIT License,
Copyright © 2022 ETH Sensing, Interaction & Perception Lab — see `LICENSE`) and incorporates
[PHC](https://github.com/ZhengyiLuo/PHC) as a git submodule (see `phc/LICENSE` for its own terms).
