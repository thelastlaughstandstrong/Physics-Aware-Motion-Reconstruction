"""Evaluate ONE student checkpoint on the 90-clip validation subset (diagnostic; modifies nothing).

Runs the standard quest distillation eval rollout with DistillPlayerQuest.get_action patched at
runtime to record the student's action, so one pass yields both the physics dump (for post-sim
metrics) and the command stream (for ankle |da|). Nothing in phc/ is edited.

Writes:   <out_prefix>_actions.npz   student actions + motion keys + num_steps
          <out_prefix>_dump.pkl      copy of PHC's phc_positions_for_eval.pkl for this checkpoint

Usage: eval_checkpoint.py <exp_name> <student_ckpt_relpath> <out_prefix>
"""
import os, shutil, sys, runpy, importlib
import isaacgym  # noqa: F401  -- must precede torch
import numpy as np, torch

REPO = "/home/sia-irl/AvatarPoser"
D = f"{REPO}/scripts/ablation_footweight"
EXP, CKPT, OUT = sys.argv[1], sys.argv[2], sys.argv[3]
NUM_ENVS = int(os.environ.get("EVAL_NUM_ENVS", "90"))

sys.path.insert(0, f"{REPO}/phc"); sys.path.insert(0, f"{REPO}/phc/phc")
os.chdir(f"{REPO}/phc")

mods = []
for n in ("learning.distill_agent_quest", "phc.learning.distill_agent_quest"):
    try: mods.append(importlib.import_module(n))
    except Exception: pass
assert mods, "distill_agent_quest not importable"
_orig = mods[0].DistillPlayerQuest.get_action
LOG = {"a": [], "dof": None, "keys": None, "ns": None}


def patched(self, obs_dict, *a, **kw):
    if not getattr(self, "_diag_ready", False):
        t = self.env.task
        LOG["dof"] = list(t._dof_names)
        LOG["keys"] = [str(k) for k in t._motion_lib._motion_data_keys]
        ns = t._motion_lib.get_motion_num_steps()
        LOG["ns"] = ns.cpu().numpy() if hasattr(ns, "cpu") else np.asarray(ns)
        self._diag_ready = True
        print(f"[eval] {EXP} {os.path.basename(CKPT)} | motions={len(LOG['keys'])}", flush=True)
    out = _orig(self, obs_dict, *a, **kw)
    LOG["a"].append(out.detach().cpu().numpy().copy())
    return out


for m in mods:
    m.DistillPlayerQuest.get_action = patched

import atexit
def save():
    if LOG["a"]:
        np.savez(f"{OUT}_actions.npz", student=np.stack(LOG["a"], 0),
                 dof_names=np.asarray(LOG["dof"] or []), keys=np.asarray(LOG["keys"] or []),
                 num_steps=np.asarray(LOG["ns"] if LOG["ns"] is not None else []))
        print(f"[eval] actions -> {OUT}_actions.npz", flush=True)
    src = f"{REPO}/output/HumanoidIm/{EXP}/phc_positions_for_eval.pkl"
    if os.path.isfile(src):
        shutil.copy2(src, f"{OUT}_dump.pkl")
        print(f"[eval] dump    -> {OUT}_dump.pkl", flush=True)
atexit.register(save)

sys.argv = ["run_hydra.py", "learning=distill_quest", f"exp_name={EXP}",
            "env=env_quest_im_basic", "robot=quest_humanoid",
            f"env.motion_file={D}/val_dir",
            f"learning.params.config.student_checkpoint={CKPT}",
            f"env.num_envs={NUM_ENVS}", "env.cycle_motion=False", "env.min_length=5",
            "test=True", "im_eval=True", "headless=True", "no_virtual_display=True", "no_log=True"]
runpy.run_path(f"{REPO}/phc/phc/run_hydra.py", run_name="__main__")
