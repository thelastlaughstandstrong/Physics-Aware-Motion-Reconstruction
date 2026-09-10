"""
Log FB-student control and PHC-teacher mu on the SAME observations, frame by frame.

Runs the normal student eval entry point (phc/run_hydra.py) with DistillPlayerQuest.get_action
monkey-patched at runtime -- no production file is modified. The patch additionally loads the
frozen teacher into the player's own `self.model` (the player stubs out restore()), so the
teacher's mu can be queried on exactly the observation the student just consumed.

Teacher label semantics are reproduced exactly as Stage E uses them:
    _query_teacher -> self.model({'obs': running_mean_std(obs), 'is_train': False})['mus']
i.e. the RAW, UNCLAMPED mu -- which is what the student was trained to imitate.

Usage:
    python scripts/diag_fb_policy/log_actions.py <clip_name> <out.npz>
"""
import os, sys, runpy
import isaacgym  # noqa: F401  -- MUST precede torch, or isaacgym refuses to import
import numpy as np, torch

REPO = "/home/sia-irl/AvatarPoser"
CLIP = sys.argv[1]
OUT = sys.argv[2]
TEACHER = f"{REPO}/output/HumanoidIm/phc_prim_quest/Humanoid.pth"

sys.path.insert(0, f"{REPO}/phc")
sys.path.insert(0, f"{REPO}/phc/phc")  # phc's internal `import learning.…`
os.chdir(f"{REPO}/phc")               # configs + relative asset paths resolve from here
# run_hydra.py does `from learning import distill_agent_quest` (bare package), NOT
# `phc.learning...`. Those are DIFFERENT sys.modules entries with different class objects, so
# patching the wrong spelling is a silent no-op. Patch every spelling that resolves.
import importlib
_MODS = []
for _name in ("learning.distill_agent_quest", "phc.learning.distill_agent_quest"):
    try:
        _MODS.append(importlib.import_module(_name))
    except Exception:
        pass
assert _MODS, "could not import distill_agent_quest under any name"
DQ = _MODS[0]

LOG = {"student": [], "teacher": [], "dof_names": None}
_orig = DQ.DistillPlayerQuest.get_action


def patched(self, obs_dict, is_determenistic=False):
    if not getattr(self, "_teach_ready", False):
        ck = torch.load(TEACHER, map_location=self.device)
        sd = ck["model"] if "model" in ck else ck
        self.model.load_state_dict(sd, strict=False)
        if getattr(self, "normalize_input", False) and "running_mean_std" in ck:
            self.running_mean_std.load_state_dict(ck["running_mean_std"])
        self.model.eval()
        self._teach_ready = True
        try:
            LOG["dof_names"] = list(self.env.task._dof_names)
        except Exception:
            LOG["dof_names"] = []
        print(f"[diag] teacher loaded; dof_names={len(LOG['dof_names'])}", flush=True)

    obs = obs_dict["obs"]
    with torch.no_grad():
        t_mu = self.model({"obs": self.running_mean_std(obs) if self.normalize_input else obs,
                           "is_train": False})["mus"]
    control = _orig(self, obs_dict, is_determenistic)
    LOG["student"].append(control.detach().cpu().numpy().copy())
    LOG["teacher"].append(t_mu.detach().cpu().numpy().copy())
    return control


for _m in _MODS:
    _m.DistillPlayerQuest.get_action = patched
print(f"[diag] patched get_action on {len(_MODS)} module alias(es)", flush=True)

import atexit
def save():
    if LOG["student"]:
        np.savez(OUT, student=np.concatenate(LOG["student"], 0),
                 teacher=np.concatenate(LOG["teacher"], 0),
                 dof_names=np.asarray(LOG["dof_names"]))
        print(f"[diag] wrote {OUT}: {len(LOG['student'])} frames", flush=True)
atexit.register(save)

os.chdir(f"{REPO}/phc")
sys.argv = ["run_hydra.py",
            "learning=distill_quest", "exp_name=quest_distill_trackerr",
            "env=env_quest_im_basic", "robot=quest_humanoid",
            f"env.motion_file={REPO}/data/quest2_synthetic/test_seg_motion/{CLIP}.pkl",
            "learning.params.config.student_checkpoint=output/HumanoidIm/quest_distill_trackerr/student/student_0380160.pth",
            "env.num_envs=1", "env.cycle_motion=False", "test=True", "im_eval=True",
            "headless=True", "no_virtual_display=True"]
runpy.run_path(f"{REPO}/phc/phc/run_hydra.py", run_name="__main__")
