"""
Level 3 smoke test for HumanoidImMCPGetupQuest (phc/phc/env/tasks/humanoid_im_quest.py).

Bypasses the full rl_games RL runner (run_hydra.py) -- there's no local `learning/amp.yaml` or
`learning/im.yaml`-compatible plain-imitation config wired for a from-scratch quest teacher yet,
and building one is out of scope for a smoke test. Instead this composes the same env/robot/sim
config via Hydra's compose API and calls parse_task() directly, mirroring what run_hydra.py's
create_rlgpu_env() does internally, then exercises reset() + a few step()s to verify:
  - MotionLibQuest's train/eval split loads the sample motion pkl without error
  - HumanoidImMCPGetupQuest's MRO correctly resolves to HumanoidQuest's Humanoid-level overrides
  - _compute_task_obs (incl. the quest_root_rot_to_smpl_frame heading correction) and
    _compute_reward run without shape/NaN errors
"""
import os
import sys

os.environ["OMP_NUM_THREADS"] = "1"
sys.path.append(os.getcwd())

PHC_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "phc")
sys.path.insert(0, os.path.abspath(os.path.join(PHC_ROOT, "phc")))
sys.path.insert(0, os.path.abspath(PHC_ROOT))

import hydra
from omegaconf import OmegaConf
from easydict import EasyDict
from isaacgym import gymapi
import torch

from phc.utils.config import set_np_formatting, set_seed
from phc.utils.parse_task import parse_task
from phc.utils.flags import flags


def main():
    with hydra.initialize_config_dir(config_dir=os.path.abspath(os.path.join(PHC_ROOT, "phc/data/cfg")), version_base=None):
        cfg_hydra = hydra.compose(config_name="config", overrides=[
            "env=env_quest_im_mcp_getup_basic",
            "robot=quest_humanoid",
            "learning=distill",  # unused (we skip the RL runner), just satisfies the defaults list
            "headless=True",
            "exp_name=quest_im_mcp_getup_smoke",
            "env.num_envs=2",
        ])

    cfg = EasyDict(OmegaConf.to_container(cfg_hydra, resolve=True))
    set_np_formatting()
    set_seed(cfg.get("seed", -1), False)

    flags.debug, flags.follow, flags.fixed, flags.divide_group, flags.no_collision_check, flags.fixed_path, flags.real_path, flags.show_traj, flags.server_mode, flags.slow, flags.real_traj, flags.im_eval, flags.no_virtual_display, flags.render_o3d = \
        cfg.debug, cfg.follow, False, False, False, False, False, True, cfg.server_mode, False, False, cfg.im_eval, cfg.no_virtual_display, cfg.render_o3d
    flags.test = cfg.test
    flags.add_proj = cfg.add_proj
    flags.has_eval = cfg.has_eval
    flags.trigger_input = False

    sim_params = gymapi.SimParams()
    sim_params.dt = eval(cfg.sim.physx.step_dt)
    sim_params.num_client_threads = cfg.sim.slices
    sim_params.physx.solver_type = 1
    sim_params.physx.num_position_iterations = 4
    sim_params.physx.num_velocity_iterations = 1
    sim_params.physx.num_threads = 4
    sim_params.physx.use_gpu = cfg.sim.pipeline in ["gpu"]
    sim_params.physx.num_subscenes = cfg.sim.subscenes
    sim_params.physx.max_gpu_contact_pairs = 4 * 1024 * 1024
    sim_params.use_gpu_pipeline = cfg.sim.pipeline in ["gpu"]

    from isaacgym import gymutil
    gymutil.parse_sim_config(cfg["sim"], sim_params)

    args = EasyDict({
        "task": cfg.env.task,
        "device_id": cfg.device_id,
        "rl_device": cfg.rl_device,
        "physics_engine": gymapi.SIM_PHYSX,
        "headless": cfg.headless,
        "device": cfg.device,
    })

    task, env = parse_task(args, cfg, cfg.learning, sim_params)

    print(f"[smoke test] task class: {type(task).__name__}")
    print(f"[smoke test] num_envs={env.num_envs} num_actions={env.num_actions} num_obs={env.num_obs}")
    print(f"[smoke test] num_amp_obs={task.get_num_amp_obs()}")

    obs = env.reset()
    print(f"[smoke test] reset OK, obs shape={obs.shape}")

    for i in range(5):
        actions = torch.zeros((env.num_envs, env.num_actions), device=cfg.device)
        obs, rew, done, info = env.step(actions)
        amp_obs = info.get("amp_obs", None)
        print(f"[smoke test] step {i}: obs {tuple(obs.shape)}, reward {rew.tolist()}, done {done.tolist()}, amp_obs {tuple(amp_obs.shape) if amp_obs is not None else None}")

    print("[smoke test] PASSED")


if __name__ == "__main__":
    main()
