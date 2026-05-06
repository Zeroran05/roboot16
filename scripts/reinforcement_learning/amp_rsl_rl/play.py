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

    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "play"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.get("clip_actions"))
    runner = AMPOnPolicyRunner(env, agent_cfg, log_dir=None, device=agent_cfg.get("device", "cuda:0"))

    print(f"[INFO]: Loading model checkpoint from: {resume_path}")
    runner.load(resume_path, load_optimizer=False)
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    dt = env.unwrapped.step_dt
    obs = env.get_observations()
    timestep = 0
    while simulation_app.is_running():
        start_time = time.time()
        with torch.inference_mode():
            actions = policy(obs)
            obs, _, dones, _ = env.step(actions)
            runner.alg.actor_critic.reset(dones)
        if args_cli.video:
            timestep += 1
            if timestep == args_cli.video_length:
                break
        sleep_time = dt - (time.time() - start_time)
        if args_cli.real_time and sleep_time > 0:
            time.sleep(sleep_time)

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
