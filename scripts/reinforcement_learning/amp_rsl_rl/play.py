#!/usr/bin/env python3

"""Play an AMP policy for project-local tasks."""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

from isaaclab.app import AppLauncher


PROJECT_ROOT = Path(__file__).resolve().parents[3]
SRC_DIR = PROJECT_ROOT / "src"
ISAACLAB_ROOT = PROJECT_ROOT.parent
LOGS_ROOT = PROJECT_ROOT / "logs"


parser = argparse.ArgumentParser(description="Play an AMP RL agent with amp-rsl-rl.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during play.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument("--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O.")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--agent",
    type=str,
    default="amp_rsl_rl_cfg_entry_point",
    help="Name of the AMP RL agent configuration entry point.",
)
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment.")
parser.add_argument("--checkpoint", type=str, default=None, help="Checkpoint file to load.")
parser.add_argument("--load_run", type=str, default=None, help="Run folder to load from.")
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time, if possible.")
parser.add_argument(
    "--print-command-interval",
    type=int,
    default=50,
    help="Print the base_velocity command for env 0 every N simulation steps. Set <= 0 to disable.",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

if args_cli.video:
    args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

for path in (SRC_DIR, ISAACLAB_ROOT / "rsl_rl"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import gymnasium as gym
import torch

from amp_rsl_rl.runners import AMPOnPolicyRunner

from isaaclab.envs import DirectMARLEnv
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

import roboot16_amp_project  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path, load_cfg_from_registry, parse_env_cfg


def _capture_base_velocity_cfg(command_cfg):
    cfg = {
        "rel_standing_envs": command_cfg.rel_standing_envs,
        "rel_heading_envs": command_cfg.rel_heading_envs,
        "heading_command": command_cfg.heading_command,
        "resampling_time_range": command_cfg.resampling_time_range,
        "ranges.lin_vel_x": command_cfg.ranges.lin_vel_x,
        "ranges.lin_vel_y": command_cfg.ranges.lin_vel_y,
        "ranges.ang_vel_z": command_cfg.ranges.ang_vel_z,
    }
    if hasattr(command_cfg.ranges, "heading"):
        cfg["ranges.heading"] = command_cfg.ranges.heading
    return cfg


def _set_nested_attr(obj, attr_path: str, value):
    target = obj
    attr_names = attr_path.split(".")
    for attr_name in attr_names[:-1]:
        target = getattr(target, attr_name)
    setattr(target, attr_names[-1], value)


def _apply_base_velocity_cfg(env, cfg_values: dict):
    command_term = env.unwrapped.command_manager.get_term("base_velocity")
    for attr_path, value in cfg_values.items():
        _set_nested_attr(env.unwrapped.cfg.commands.base_velocity, attr_path, value)
        _set_nested_attr(command_term.cfg, attr_path, value)
    command_term._resample(list(range(env.unwrapped.num_envs)))


def main():
    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric,
    )
    agent_cfg = load_cfg_from_registry(args_cli.task, args_cli.agent)
    if not isinstance(agent_cfg, dict):
        raise TypeError(f"Expected a dict config from '{args_cli.agent}', got {type(agent_cfg)}.")
    agent_cfg = dict(agent_cfg)

    if args_cli.seed is not None:
        agent_cfg["seed"] = args_cli.seed
        env_cfg.seed = args_cli.seed
    elif "seed" in agent_cfg:
        env_cfg.seed = agent_cfg["seed"]

    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    print(f"[DEBUG] roboot16_amp_project imported from: {roboot16_amp_project.__file__}")
    print(f"[DEBUG] env_cfg type: {type(env_cfg).__module__}.{type(env_cfg).__name__}")
    print(f"[DEBUG] task={args_cli.task}")
    print(
        "[DEBUG] base_velocity cfg: "
        f"resampling_time_range={env_cfg.commands.base_velocity.resampling_time_range}, "
        f"lin_vel_x={env_cfg.commands.base_velocity.ranges.lin_vel_x}, "
        f"lin_vel_y={env_cfg.commands.base_velocity.ranges.lin_vel_y}, "
        f"ang_vel_z={env_cfg.commands.base_velocity.ranges.ang_vel_z}, "
        f"rel_standing_envs={env_cfg.commands.base_velocity.rel_standing_envs}"
    )

    log_root_path = str((LOGS_ROOT / "amp_rsl_rl" / agent_cfg["experiment_name"]).resolve())
    print(f"[INFO] Loading experiment from directory: {log_root_path}")

    if args_cli.checkpoint:
        resume_path = args_cli.checkpoint
    else:
        load_run = args_cli.load_run if args_cli.load_run is not None else agent_cfg.get("load_run", ".*")
        load_checkpoint = agent_cfg.get("load_checkpoint", "model_.*.pt")
        resume_path = get_checkpoint_path(log_root_path, load_run, load_checkpoint)

    log_dir = os.path.dirname(resume_path)
    env_cfg.log_dir = log_dir

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
    if isinstance(env.unwrapped, DirectMARLEnv):
        from isaaclab.envs import multi_agent_to_single_agent

        env = multi_agent_to_single_agent(env)

    active_curriculum_terms = list(env.unwrapped.curriculum_manager.active_terms)
    runtime_command_term = env.unwrapped.command_manager.get_term("base_velocity")
    print(f"[DEBUG] runtime curriculum terms: {active_curriculum_terms}")
    print(
        "[DEBUG] runtime base_velocity term cfg: "
        f"resampling_time_range={runtime_command_term.cfg.resampling_time_range}, "
        f"lin_vel_x={runtime_command_term.cfg.ranges.lin_vel_x}, "
        f"lin_vel_y={runtime_command_term.cfg.ranges.lin_vel_y}, "
        f"ang_vel_z={runtime_command_term.cfg.ranges.ang_vel_z}, "
        f"rel_standing_envs={runtime_command_term.cfg.rel_standing_envs}"
    )
    initial_command = env.unwrapped.command_manager.get_command("base_velocity")[0].detach().cpu().tolist()
    print(f"[DEBUG] command right after gym.make env0={initial_command}")

    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "play"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.get("clip_actions"))
    play_base_velocity_cfg = _capture_base_velocity_cfg(env.unwrapped.cfg.commands.base_velocity)
    runner = AMPOnPolicyRunner(env, agent_cfg, log_dir=None, device=agent_cfg.get("device", "cuda:0"))

    print(f"[INFO]: Loading model checkpoint from: {resume_path}")
    runner.load(resume_path, load_optimizer=False)
    _apply_base_velocity_cfg(env, play_base_velocity_cfg)
    restored_command = env.unwrapped.command_manager.get_command("base_velocity")[0].detach().cpu().tolist()
    print(f"[DEBUG] command after restoring play base_velocity cfg env0={restored_command}")
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    dt = env.unwrapped.step_dt
    obs = env.get_observations()
    command_after_obs = env.unwrapped.command_manager.get_command("base_velocity")[0].detach().cpu().tolist()
    print(f"[DEBUG] command after get_observations env0={command_after_obs}")
    timestep = 0
    while simulation_app.is_running():
        start_time = time.time()
        with torch.inference_mode():
            if timestep == 0:
                command_before_step = env.unwrapped.command_manager.get_command("base_velocity")[0].detach().cpu().tolist()
                print(f"[DEBUG] command before first env.step env0={command_before_step}")
            actions = policy(obs)
            obs, _, dones, _ = env.step(actions)
            if timestep == 0:
                command_after_step = env.unwrapped.command_manager.get_command("base_velocity")[0].detach().cpu().tolist()
                print(f"[DEBUG] command after first env.step env0={command_after_step}")
            runner.alg.actor_critic.reset(dones)
        if args_cli.print_command_interval > 0 and timestep % args_cli.print_command_interval == 0:
            command = env.unwrapped.command_manager.get_command("base_velocity")[0].detach().cpu().tolist()
            print(f"[DEBUG] step={timestep} env0 base_velocity command={command}")
        if args_cli.video:
            timestep += 1
            if timestep == args_cli.video_length:
                break
        else:
            timestep += 1
        sleep_time = dt - (time.time() - start_time)
        if args_cli.real_time and sleep_time > 0:
            time.sleep(sleep_time)

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
