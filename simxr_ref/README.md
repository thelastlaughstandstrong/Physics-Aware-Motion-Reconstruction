# simxr_ref: SimXR evaluation instrumentation (historical record)

This directory originally held working copies of two files from
[SimXR](https://github.com/ZhengyiLuo/SimXR) (Luo et al., *Real-Time Simulated Avatar from
Head-Mounted Sensors*, CVPR 2024), patched so the SimXR **no-vision baseline** could be scored
with this repository's full metric set.

**SimXR declares no license, so those copies are not part of the public release.** They are kept
out of version control (see `.gitignore`); this README documents what they were and how to
reproduce them from an official SimXR checkout.

## Role in the experiments

Nothing in this repository imports or executes these files. They were applied once, by hand, to
a **separate SimXR checkout**, which produced the SimXR baseline's `phc_positions_for_eval.pkl`
dump. This repository's `scripts/evaluate_phc_metrics_quest.py` then scored that dump, giving the
MPJRE / MPJVE / jitter / foot-skate values in the SimXR column of `results_comparison.md`
(footnote 7).

| SimXR file (official revision) | Modification made for this project |
|---|---|
| `phc/env/tasks/humanoid_im.py` | In `post_physics_step`, also expose the simulated and reference rigid-body rotations (`body_rot`, `body_rot_gt`) in `self.extras`, next to the existing positions. MPJRE needs them |
| `phc/learning/im_amp_players.py` | Accumulate those rotations per evaluation batch alongside positions. Remove two unconditional `ipdb.set_trace()` breakpoints in the evaluation loop, which hang a non-interactive run. Write `phc_positions_for_eval.pkl` (positions, rotations, motion keys, success rate, termination history), then exit after the single evaluation pass |

The research algorithm, policy, observations, reward and success computation are **not**
changed. Only data is exported, and the interactive breakpoints are removed.

**Upstream revision used:** `39081b831aa1c1e52a4a974fb78a1ed93f7580c2` (SimXR `main`,
2025-05-11, "raw processing file").

## Reproducing the instrumentation

The modification is stored as an edit script,
[`scripts/simxr_novision/simxr_eval_instrumentation.json`](../scripts/simxr_novision/simxr_eval_instrumentation.json).
It holds line-range references into the official files plus this project's own added lines
(29 lines; no SimXR text). The applier checks SHA-256 before and after:

```bash
git clone https://github.com/ZhengyiLuo/SimXR.git /path/to/SimXR
git -C /path/to/SimXR checkout 39081b831aa1c1e52a4a974fb78a1ed93f7580c2
python scripts/simxr_novision/apply_eval_instrumentation.py --simxr /path/to/SimXR --check
python scripts/simxr_novision/apply_eval_instrumentation.py --simxr /path/to/SimXR
```

| File | SHA-256 of the official file | SHA-256 after instrumentation |
|---|---|---|
| `phc/env/tasks/humanoid_im.py` | `34ec3bf8bdcb7ec6b693afe1372b5169cb23105bb93712abe1eecc440959ddf7` | `9631b1fc50753096dd2f994ea739020aa01e6a597cdef4e25fa804a93a3ec69b` |
| `phc/learning/im_amp_players.py` | `ffb5f19774096e2b5ca4a6ae4c20950f7dfe20479525a09d4b061cbdd4f5d057` | `acd2a103c40e35a8b308ee59193f37294bc274dcdcdf653dddc1ec6251b66647` |

The result is byte-identical to the files used in the experiments; this was verified on
2026-11-23. Revert with `git -C /path/to/SimXR checkout -- phc/`. Then follow
[`scripts/simxr_novision/README.md`](../scripts/simxr_novision/README.md) to train or evaluate the
baseline in the SimXR checkout, and score the dump with:

```bash
python scripts/evaluate_phc_metrics_quest.py \
  --pos_dump <SimXR>/output/HumanoidIm/<exp_name>/phc_positions_for_eval.pkl
```

Your use of SimXR, its released checkpoints and its data is governed by its authors' terms.

## Historical notes (2026-07-27)

- The instrumented files were copied over the SimXR checkout on the original workstation. The
  peak (epoch 1,500) checkpoint of `simxr_quest_novision_lrdecay` was re-evaluated with
  `test=True im_eval=True` at `env.num_envs=64`, reduced from 512 because of GPU contention. This
  produced the dump.
- That re-run reproduced the original console numbers within noise: success 0.7155 vs. 0.717,
  MPJPE 112.36 vs. 112.5 mm. This supports the 64-env substitution.
- The unconditional `ipdb.set_trace()` had silently blocked this dump step in every earlier
  manual evaluation; metrics printed, then the process waited at a debugger prompt.
- The SimXR checkout, its checkpoints and the resulting dump lived only on the original
  workstation and **were not preserved** (see `docs/DATA_AND_ARTIFACTS.md`).
