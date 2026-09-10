"""Log SimXR's policy output before simulator application (diagnostic only; modifies no file).

Runs SimXR's own eval entry point with AMPPlayerContinuous.get_action patched at runtime, on the
SAME 56 ground-truth Quest clips used by the 49-clip FB analysis, so the reference motions are
identical across the two systems.

Streams recorded:
  simxr    the SimXR student's action, along ITS OWN rollout trajectory.
  teacher  the frozen PHC teacher (phc_prim_quest/Humanoid.pth -- md5-identical to the one the
           Finetuned Backbone distils from) queried on the observation the SimXR student consumed.
           COUNTERFACTUAL: what that teacher would command in SimXR's states.

The teacher stream is produced by the ENV itself, not by a second network I load: with
distill: True the task already computes kin_dict['gt_action'] every step for its own BC loss
(humanoid_ego_img.py:466, gt_action = self.encoder(full_obs)). Reading that buffer gives exactly
the target SimXR was trained against -- raw mu, unclipped, unsampled -- with no re-derivation.

Usage: log_simxr_actions.py <out.npz>
"""
import os, sys, runpy, importlib
import isaacgym  # noqa: F401  -- must precede torch
import numpy as np, torch

SIMXR = "/home/sia-irl/SimXR"
D = "/home/sia-irl/AvatarPoser/scripts/diag_simxr_ankle"
OUT = sys.argv[1]

sys.path.insert(0, SIMXR); sys.path.insert(0, f"{SIMXR}/phc")
os.chdir(SIMXR)

mods = []
for n in ("learning.amp_players", "phc.learning.amp_players"):
    try: mods.append(importlib.import_module(n))
    except Exception: pass
assert mods, "amp_players not importable"

LOG = {"simxr": [], "teacher": [], "dof_names": None, "keys": None, "num_steps": None}
_orig = mods[0].AMPPlayerContinuous.get_action


def patched(self, obs_dict, *a, **kw):
    task = self.env.task
    if not getattr(self, "_diag_ready", False):
        LOG["dof_names"] = list(task._dof_names)
        LOG["keys"] = [str(k) for k in task._motion_lib._motion_data_keys]
        ns = task._motion_lib.get_motion_num_steps()
        LOG["num_steps"] = ns.cpu().numpy() if hasattr(ns, "cpu") else np.asarray(ns)
        self._diag_ready = True
        print(f"[diag] dofs={len(LOG['dof_names'])} motions={len(LOG['keys'])} "
              f"distill={getattr(task,'distill',None)}", flush=True)
    act = _orig(self, obs_dict, *a, **kw)
    LOG["simxr"].append(act.detach().cpu().numpy().copy())
    # NOTE: kin_dict['gt_action'] is ZERO-INITIALISED at humanoid_ego_img.py:95 and only written
    # under `if self.distill and not flags.test and self.save_kin_info` (line 421). In an eval run
    # flags.test is True, so the buffer is never filled -- reading it yields all zeros, which looks
    # like a valid stream. Only keep it if it is actually non-zero.
    gt = task.kin_dict.get("gt_action") if getattr(task, "kin_dict", None) is not None else None
    if gt is not None and bool(gt.abs().max() > 0):
        LOG["teacher"].append(gt.detach().cpu().numpy().copy())
    return act


for m in mods:
    m.AMPPlayerContinuous.get_action = patched
print(f"[diag] patched AMPPlayerContinuous on {len(mods)} alias(es)", flush=True)

import atexit
def save():
    if not LOG["simxr"]:
        print("[diag] nothing logged", flush=True); return
    kw = dict(simxr=np.stack(LOG["simxr"], 0),
              dof_names=np.asarray(LOG["dof_names"] or []),
              keys=np.asarray(LOG["keys"] or []),
              num_steps=np.asarray(LOG["num_steps"] if LOG["num_steps"] is not None else []))
    if LOG["teacher"]:
        kw["teacher"] = np.stack(LOG["teacher"], 0)
    else:
        print("[diag] teacher stream NOT captured (env computes gt_action only when not test)",
              flush=True)
    np.savez(OUT, **kw)
    print(f"[diag] wrote {OUT}: simxr{kw['simxr'].shape}"
          + (f" teacher{kw['teacher'].shape}" if "teacher" in kw else " (NO teacher stream)"),
          flush=True)
atexit.register(save)

sys.argv = ["run_hydra.py",
            "exp_name=diag_simxr_ankle",
            "env=env_simxr_quest", "learning=im_simxr", "robot=quest_humanoid",
            f"env.motion_file={D}/gt_dir",
            "+env.pretrain=True", "env.cycle_motion=False",
            "epoch=1500", "test=True", "im_eval=True",
            "env.num_envs=56", "headless=True", "no_virtual_display=True", "no_log=True"]
runpy.run_path(f"{SIMXR}/phc/run_hydra.py", run_name="__main__")
