# SimXR no-vision ablation — reproducible pipeline

Reproduces the **71.7 % success / 112.5 mm MPJPE_g** SimXR baseline row in
`results_comparison.md`. Everything here is a *record and re-run recipe* for an already-completed
run; nothing in the SimXR checkout (`$SIMXR`) is modified by these scripts except the output directory you
point them at.

```
train.sh    train the no-vision imitator from scratch  (~6104 epochs)
eval.sh     evaluate one checkpoint on the 1072-clip held-out test set
verify.sh   check an existing eval log against the reference numbers (no GPU, instant)
```

## Reference result

| | value |
|---|---|
| checkpoint | `Humanoid_00001500.pth` (epoch 1500 of 6104 — the **peak**, not the final) |
| success rate | **0.7173507463** |
| MPJPE_g / _l / _pa | 112.527 / 95.979 / 79.372 mm |
| accel_dist / vel_dist | 11.513 / 13.681 |
| paper (Table 5, no-vision row) | 73.0 % / 118.9 mm |

Two independent evaluations — `eval_peak_1500.log` (2026-07-22) and
`eval_fulltest_1500_rerun.log` (2026-08-20) — agree to **every printed digit on every metric**, so
the evaluation path is deterministic given the checkpoint. The August rerun produced the
`phc_positions_for_eval.pkl` dump used by the mesh-GPen study.

## Environment

| | |
|---|---|
| repo | `$SIMXR`, a separate SimXR checkout (set `SIMXR=/path/to/SimXR`; the scripts' default is the original workstation's path) |
| conda env | **`isaac`** — *not* `rlgpu`, which is the AvatarPoser/PHC env |
| entry point | `phc/run_hydra.py`, run with cwd = repo root |
| seed | `0` (`config.yaml:12`), `torch_deterministic: False` |

## 1. Training

```bash
./train.sh                       # or: MAX_EPOCHS=6104 NUM_ENVS=512 ./train.sh
```

Expands to:

```
python phc/run_hydra.py \
  exp_name=simxr_quest_novision_lrdecay \
  env=env_simxr_quest learning=im_simxr robot=quest_humanoid \
  env.motion_file=<DATA>/train_seg_motion/ \
  +env.pretrain=True \
  env.num_envs=512 \
  learning.params.config.lr_schedule=adaptive \
  +learning.params.config.kl_threshold=0.008 \
  learning.params.config.max_epochs=6104 \
  headless=True no_virtual_display=True
```

**`+env.pretrain=True` is what disables vision.** Verified in source, not assumed: at
`phc/env/tasks/humanoid_ego_img.py:233`, for `obs_v: 6` the observation is
`len(track_bodies) * num_traj_samples * 24`, and `np.prod(self.img_dim)` is added **only if
`not self.pretrain`**. The image channels are absent from the observation entirely — the ResNet is
never fed, so this is a true no-vision ablation rather than vision fed with blanks.

**The two LR overrides are coupled and both are required.** `im_simxr.yaml:58` ships
`lr_schedule: constant` @ `learning_rate: 2e-5`. In rl_games `a2c_common.py:122`,
`is_adaptive_lr = config['lr_schedule'] == 'adaptive'`; line 127 then reads `config['kl_threshold']`
*unconditionally*. That key does not exist in `im_simxr.yaml`, so `lr_schedule=adaptive` without
`+…kl_threshold=0.008` raises `KeyError`. Hence `+` on the second override and no `+` on the first.

**Why 6104 epochs:** `6104 = 100,000,000 / (horizon_length 32 x num_envs 512)`, matching the paper's
stated ~0.1 B environment interactions. `im_simxr.yaml`'s own `max_epochs: 10000000` is effectively
unbounded and must be overridden.

Checkpoints land every 500 epochs (`save_frequency: 500`), giving
`Humanoid_00000500.pth … Humanoid_00006000.pth` plus a rolling `Humanoid.pth`.

Training data: `data/quest2_synthetic/train_seg_motion/`, **4097 clips** — the same split used by
this project's own Quest2 runs (4097 train / 1072 test), so the SimXR row is split-comparable with
the rest of the table.

Key env settings inherited from `env_simxr_quest.yaml`: `control_mode: isaac_pd`,
`controlFrequencyInv: 2` (30 Hz), `obs_v: 6`.

## 2. Evaluation

```bash
./eval.sh                        # or: EPOCH=1500 EXP=<fresh_name> ./eval.sh
```

```
python phc/run_hydra.py \
  exp_name=simxr_quest_novision_lrdecay \
  env=env_simxr_quest learning=im_simxr robot=quest_humanoid \
  env.motion_file=<DATA>/test_seg_motion/ \
  +env.pretrain=True env.cycle_motion=False \
  epoch=1500 test=True im_eval=True \
  headless=True no_virtual_display=True no_log=True
```

- `epoch=1500` resolves to `Humanoid_00001500.pth` (`name + '_' + str(epoch).zfill(8) + '.pth'`).
- `+env.pretrain=True` **must be repeated at eval** — it changes the observation width, so a
  vision-enabled eval of a no-vision checkpoint is a shape mismatch, not a silently worse score.
- `env.cycle_motion=False` stops short clips from looping, so each clip is scored once.
- `im_eval=True` is what emits the `Success Rate` / `All: mpjpe_g …` lines and the
  `phc_positions_for_eval.pkl` dump.

> **Overwrite warning.** The eval writes `phc_positions_for_eval.pkl` into
> `output/HumanoidIm/$EXP/`, replacing any dump already there. Pass a fresh `EXP=` to preserve the
> existing one. This exact overwrite has already destroyed run records twice in this project.

## 3. Verification without a GPU

```bash
./verify.sh                                        # checks the archived eval_peak_1500.log
./verify.sh /path/to/your/eval_epoch1500.log       # checks a fresh run
```

Greps the success rate and `mpjpe_g` out of a log and diffs them against the reference constants.

## Choosing the checkpoint — read before re-running

The reported number comes from the **peak**, not the end of training:

| checkpoint (this run) | success | log |
|---|---|---|
| epoch 1500 | **0.7174** | `eval_peak_1500.log`, `eval_fulltest_1500_rerun.log` |
| epoch 6105 (final) | 0.5140 | `eval_final_6105.log` |

Late-training degradation is a property of this run, so a fresh training run must be evaluated
across its 500-epoch checkpoints rather than only at the end. Selecting on the test set is
model selection on held-out data — acceptable for reproducing a published ablation, but it should be
stated as such wherever the 71.7 % is cited.

## Which output directory is authoritative

| directory | what it is |
|---|---|
| `simxr_quest_novision_lrdecay` | **the reported run** — adaptive LR, source of 71.7 % |
| `simxr_quest_novision` | superseded constant-LR run, trained far longer: 42.8 % @ 12500, decaying to 22.0 % @ 24416. Confusingly it holds the shared `ablation_results.md`, which documents *both* runs. |
| `simxr_quest_repro` | unrelated side run on `train_seg/`, `num_envs=16` |

## Provenance — what is and isn't machine-verifiable

Hydra rewrites `.hydra/overrides.yaml` and `.hydra/config.yaml` on **every** invocation into the
same output dir. Later evals (including two single-clip `SUBJECT075.ROM` runs on 2026-08-20 from a
different session) have overwritten the training run's record, so `_lrdecay/.hydra/config.yaml` now
shows the *eval's* inherited defaults — `num_envs: 1`, `lr_schedule: constant`. Those are not the
training settings.

| setting | how it is evidenced |
|---|---|
| dataset, 512 motions/batch, epoch count, seed | `train.log` — direct |
| checkpoint cadence | `Humanoid_*.pth` on disk + `save_frequency: 500` |
| `pretrain=True` disables vision | `humanoid_ego_img.py:233` — source-verified |
| eval config + results | `eval_peak_1500.log`, reproduced bit-identically |
| **`lr_schedule=adaptive`, `kl_threshold=0.008`** | **prose in `ablation_results.md` only.** The LR is not echoed to `train.log` and the Hydra record was overwritten. Not machine-verifiable from run artifacts. |

The last row is the one real gap. It is consistent with the evidence — the constant-LR run in the
sibling directory reached only 51.4 %, which is why the LR was changed — but it rests on a written
note, not an artifact. `RUN_CONFIG.md`, dropped next to the checkpoints, now carries these overrides
in a form that later runs will not overwrite.

To close the gap properly, a fresh `train.sh` run reaching ~71 % at epoch 1500 would confirm it. That
costs a full training run and nothing downstream currently depends on the exact LR value, so it is
recorded as a known limitation rather than scheduled.

## Validation status of these scripts

Checked on 2026-09-03 without consuming GPU time or touching any run output:

| check | result |
|---|---|
| training overrides parse and resolve (`hydra.compose`) | pass — `lr_schedule=adaptive`, `kl_threshold=0.008`, `max_epochs=6104` all land at the documented config paths |
| derived interaction budget | `32 x 512 x 6104 = 100,007,936` ≈ the paper's 0.1 B |
| eval overrides parse and resolve | pass — `epoch=1500` → `Humanoid_00001500.pth`, confirmed present on disk |
| `env.pretrain=True` reaches the env | pass — resolves to `env.pretrain: True` with `obs_v: 6`, `control_mode: isaac_pd` |
| `verify.sh` against the archived log | pass — success and `mpjpe_g` both MATCH |

Not yet executed end to end: `train.sh` has never been run as a script (it is a reconstruction of a
completed run), and `eval.sh` has not been run since it was written, because doing so would overwrite
the `phc_positions_for_eval.pkl` that the mesh-GPen results depend on. Run it with a fresh `EXP=`
if you want a live end-to-end confirmation.
