#!/usr/bin/env python3

"""Train an AMP RL agent for project-local tasks."""

from __future__ import annotations

import argparse
import os
import sys
import time
from datetime import datetime
from pathlib import Path

from isaaclab.app import AppLauncher


PROJECT_ROOT = Path(__file__).resolve().parents[3]
SRC_DIR = PROJECT_ROOT / "src"
ISAACLAB_ROOT = PROJECT_ROOT.parent
LOGS_ROOT = PROJECT_ROOT / "logs"


parser = argparse.ArgumentParser(description="Train an AMP RL agent with amp-rsl-rl.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument("--video_interval", type=int, default=2000, help="Interval between video recordings (in steps).")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--agent",
    type=str,
    default="amp_rsl_rl_cfg_entry_point",
    help="Name of the AMP RL agent configuration entry point.",
)
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment.")
parser.add_argument("--max_iterations", type=int, default=None, help="AMP policy training iterations.")
parser.add_argument("--experiment_name", type=str, default=None, help="Experiment folder name.")
parser.add_argument("--run_name", type=str, default=None, help="Run name suffix.")
parser.add_argument("--resume", action="store_true", default=False, help="Whether to resume from a checkpoint.")
parser.add_argument("--load_run", type=str, default=None, help="Run folder to resume from.")
parser.add_argument("--checkpoint", type=str, default=None, help="Checkpoint filename to resume from.")
parser.add_argument(
    "--logger", type=str, default=None, choices={"wandb", "tensorboard", "neptune"}, help="Logger module to use."
)
parser.add_argument(
    "--log_project_name", type=str, default=None, help="Project name for wandb or neptune if used."
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
import yaml

from amp_rsl_rl.runners import AMPOnPolicyRunner

from isaaclab.envs import DirectMARLEnv
from isaaclab.utils.io import dump_yaml
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

import roboot16_amp_project  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path, load_cfg_from_registry, parse_env_cfg


def _to_plain_dict(cfg):
    if isinstance(cfg, dict):
        return cfg
    if hasattr(cfg, "to_dict"):
        return cfg.to_dict()
    return cfg


def main():
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
    agent_cfg = load_cfg_from_registry(args_cli.task, args_cli.agent)
    if not isinstance(agent_cfg, dict):
        raise TypeError(
            f"Expected a plain dict from '{args_cli.agent}', but got {type(agent_cfg)}. "
            "Please keep amp_rsl_rl_cfg_entry_point registered to a dict-like config."
        )
    agent_cfg = dict(agent_cfg)

    if args_cli.seed is not None:
        agent_cfg["seed"] = args_cli.seed
        env_cfg.seed = args_cli.seed
    elif "seed" in agent_cfg:
        env_cfg.seed = agent_cfg["seed"]

    if args_cli.max_iterations is not None:
        agent_cfg["max_iterations"] = args_cli.max_iterations
    if args_cli.experiment_name is not None:
        agent_cfg["experiment_name"] = args_cli.experiment_name
    if args_cli.run_name is not None:
        agent_cfg["run_name"] = args_cli.run_name
    if args_cli.logger is not None:
        agent_cfg["logger"] = args_cli.logger
    if args_cli.log_project_name and agent_cfg.get("logger") in {"wandb", "neptune"}:
        agent_cfg["wandb_project"] = args_cli.log_project_name
        agent_cfg["neptune_project"] = args_cli.log_project_name
    if args_cli.resume:
        agent_cfg["resume"] = True
    if args_cli.load_run is not None:
        agent_cfg["load_run"] = args_cli.load_run
    if args_cli.checkpoint is not None:
        agent_cfg["load_checkpoint"] = args_cli.checkpoint

    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    log_root_path = str((LOGS_ROOT / "amp_rsl_rl" / agent_cfg["experiment_name"]).resolve())
    print(f"[INFO] Logging experiment in directory: {log_root_path}")
    log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    print(f"Exact experiment name requested from command line: {log_dir}")
    if agent_cfg.get("run_name"):
        log_dir += f"_{agent_cfg['run_name']}"
    log_dir = os.path.join(log_root_path, log_dir)

    env_cfg.log_dir = log_dir

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
    if isinstance(env.unwrapped, DirectMARLEnv):
        from isaaclab.envs import multi_agent_to_single_agent

        env = multi_agent_to_single_agent(env)

    resume_path = None
    if agent_cfg.get("resume", False):
        load_run = agent_cfg.get("load_run", ".*")
        load_checkpoint = agent_cfg.get("load_checkpoint", "model_.*.pt")
        resume_path = get_checkpoint_path(log_root_path, load_run, load_checkpoint)

    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "train"),
            "step_trigger": lambda step: step % args_cli.video_interval == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    start_time = time.time()
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.get("clip_actions"))
    runner = AMPOnPolicyRunner(env, agent_cfg, log_dir=log_dir, device=agent_cfg.get("device", "cuda:0"))
    runner.add_git_repo_to_log(__file__)

    if resume_path is not None:
        print(f"[INFO]: Loading model checkpoint from: {resume_path}")
        runner.load(resume_path)

    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    with open(os.path.join(log_dir, "params", "agent.yaml"), "w", encoding="utf-8") as f:
        yaml.safe_dump(_to_plain_dict(agent_cfg), f, allow_unicode=True, sort_keys=False)

    print(f"[INFO]: Environment setup complete in {time.time() - start_time:.2f} seconds.")
    runner.learn(num_learning_iterations=agent_cfg["max_iterations"], init_at_random_ep_len=False)

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
