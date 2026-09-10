"""Batched multi-clip action logging (diagnostic only -- modifies no production file).

Two modes, deliberately kept distinct because the fairness of the whole analysis depends on it:

  fb     Finetuned Backbone (trackerr@380k) rolling out on GROUND-TRUTH Quest motions.
         Logs the student's control AND the frozen PHC teacher's mu queried on the SAME
         observation the student just consumed. The teacher stream is therefore COUNTERFACTUAL:
         "what the teacher would command in the states the student created". It is NOT the
         teacher's own trajectory.

  naive  The SAME frozen PHC weights (phc_prim_quest/Humanoid.pth) rolling out on AVATARPOSER
         PREDICTIONS -- the Naive PHC baseline. This is the teacher's OWN trajectory, its own
         visited states. Never mix this with the counterfactual stream above.

Runs with num_envs == n_clips so env i <-> motion i (humanoid_im.py:123 sets
_sampled_motion_ids = arange(num_envs) once and never reassigns it). Per-env valid frames are
sliced exactly as im_amp_players.py does it: arr[:(num_steps[idx]-1), idx]. The motion keys and
num_steps are read back FROM THE RUN and saved, so attribution never relies on assumed ordering.

Usage: log_actions_multi.py {fb|naive} <out.npz>
"""
import os, sys, runpy, importlib
import isaacgym  # noqa: F401  -- MUST precede torch
import numpy as np, torch

REPO = "/home/sia-irl/AvatarPoser"
D = f"{REPO}/scripts/diag_fb_multiclip"
MODE, OUT = sys.argv[1], sys.argv[2]
assert MODE in ("fb", "naive")
TEACHER = f"{REPO}/output/HumanoidIm/phc_prim_quest/Humanoid.pth"

sys.path.insert(0, f"{REPO}/phc"); sys.path.insert(0, f"{REPO}/phc/phc")
os.chdir(f"{REPO}/phc")

N_CLIPS = len(__import__("json").load(open(f"{D}/clip_order.json")))
LOG = {"student": [], "teacher": [], "dof_names": None, "keys": None, "num_steps": None}


def _grab_env_meta(self):
    try:
        t = self.env.task
        LOG["dof_names"] = list(t._dof_names)
        LOG["keys"] = [str(k) for k in t._motion_lib._motion_data_keys]
        ns = t._motion_lib.get_motion_num_steps()
        LOG["num_steps"] = ns.cpu().numpy() if hasattr(ns, "cpu") else np.asarray(ns)
    except Exception as e:
        print(f"[diag] meta grab failed: {e}", flush=True)


def make_patch(orig, with_teacher):
    def patched(self, obs_dict, *a, **kw):
        if not getattr(self, "_diag_ready", False):
            if with_teacher:
                ck = torch.load(TEACHER, map_location=self.device)
                self.model.load_state_dict(ck["model"] if "model" in ck else ck, strict=False)
                if getattr(self, "normalize_input", False) and "running_mean_std" in ck:
                    self.running_mean_std.load_state_dict(ck["running_mean_std"])
                self.model.eval()
            _grab_env_meta(self)
            self._diag_ready = True
            print(f"[diag] ready: dofs={len(LOG['dof_names'] or [])} "
                  f"motions={len(LOG['keys'] or [])}", flush=True)
        if with_teacher:
            obs = obs_dict["obs"]
            with torch.no_grad():
                t_mu = self.model({"obs": self.running_mean_std(obs) if self.normalize_input else obs,
                                   "is_train": False})["mus"]
            LOG["teacher"].append(t_mu.detach().cpu().numpy().copy())
        control = orig(self, obs_dict, *a, **kw)
        LOG["student"].append(control.detach().cpu().numpy().copy())
        return control
    return patched


if MODE == "fb":
    mods = []
    for n in ("learning.distill_agent_quest", "phc.learning.distill_agent_quest"):
        try: mods.append(importlib.import_module(n))
        except Exception: pass
    assert mods, "distill_agent_quest not importable"
    orig = mods[0].DistillPlayerQuest.get_action
    p = make_patch(orig, with_teacher=True)
    for m in mods: m.DistillPlayerQuest.get_action = p
    print(f"[diag] patched DistillPlayerQuest on {len(mods)} alias(es)", flush=True)
    argv = ["run_hydra.py", "learning=distill_quest", "exp_name=diag_mc_fb",
            "env=env_quest_im_basic", "robot=quest_humanoid",
            f"env.motion_file={D}/gt_subset.pkl",
            "learning.params.config.student_checkpoint="
            "output/HumanoidIm/quest_distill_trackerr/student/student_0380160.pth"]
else:
    mods = []
    for n in ("learning.im_amp_players", "phc.learning.im_amp_players"):
        try: mods.append(importlib.import_module(n))
        except Exception: pass
    assert mods, "im_amp_players not importable"
    orig = mods[0].IMAMPPlayerContinuous.get_action
    p = make_patch(orig, with_teacher=False)
    for m in mods: m.IMAMPPlayerContinuous.get_action = p
    print(f"[diag] patched IMAMPPlayerContinuous on {len(mods)} alias(es)", flush=True)
    argv = ["run_hydra.py", "learning=im_quest", "exp_name=diag_mc_naive",
            "env=env_quest_im_basic", "robot=quest_humanoid",
            # DIRECTORY mode, not a merged pkl: motion_lib_base.load_data() applies the
            # min_length / im_eval branches only in file mode, and both index 'pose_quat_global',
            # which the AvatarPoser-pred motions do not carry (they have pose_quat only).
            f"env.motion_file={D}/pred_dir", "has_eval=True", "epoch=-1"]

argv += [f"env.num_envs={N_CLIPS}", "env.cycle_motion=False", "test=True", "im_eval=True",
         "headless=True", "no_virtual_display=True"]

import atexit
def save():
    if not LOG["student"]:
        print("[diag] nothing logged", flush=True); return
    s = np.stack(LOG["student"], 0)                       # [T, num_envs, 72]
    kw = dict(student=s, dof_names=np.asarray(LOG["dof_names"] or []),
              keys=np.asarray(LOG["keys"] or []),
              num_steps=np.asarray(LOG["num_steps"] if LOG["num_steps"] is not None else []),
              mode=np.asarray([MODE]))
    if LOG["teacher"]:
        kw["teacher"] = np.stack(LOG["teacher"], 0)
    np.savez(OUT, **kw)
    print(f"[diag] wrote {OUT}: student{s.shape}"
          + (f" teacher{kw['teacher'].shape}" if "teacher" in kw else ""), flush=True)
atexit.register(save)

sys.argv = argv
runpy.run_path(f"{REPO}/phc/phc/run_hydra.py", run_name="__main__")
