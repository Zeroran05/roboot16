#!/usr/bin/env python3
# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Inspect the runtime joint/action order for a roboot16 IsaacLab task.

Example:
    ./isaaclab.sh -p roboot16/scripts/inspect_roboot16_action_order.py \
        --task Isaac-Roboot16-AMP-Flat-HighSpeed-Project-v0 --num_envs 1 --headless
"""

import argparse
import sys
from pathlib import Path

from isaaclab.app import AppLauncher


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"


parser = argparse.ArgumentParser(description="Inspect runtime joint and action order for a roboot16 task.")
parser.add_argument("--task", type=str, required=True, help="Gym task id to inspect.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of envs to create.")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import gymnasium as gym

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg

try:
    import roboot16_amp_project  # noqa: F401
except ImportError as exc:
    raise ImportError(
        f"无法导入项目本地任务包 'roboot16_amp_project'。\n"
        f"已尝试加入 src 路径: {SRC_DIR}\n"
        "请确认你是从 Isaac Lab 仓库根目录运行脚本，并且 roboot16/src/roboot16_amp_project 存在。"
    ) from exc


def _to_list(value):
    if isinstance(value, slice):
        return "slice(None)"
    if hasattr(value, "tolist"):
        return value.tolist()
    return list(value)


def _inspect_obs_descriptor(env, term_cfg):
    """Run the observation term once with inspect=True to populate its IO descriptor."""
    params = dict(term_cfg.params)
    term_cfg.func(env, **params, inspect=True)
    return getattr(term_cfg.func, "_descriptor", None)


def main():
    try:
        env_cfg = parse_env_cfg(
            args_cli.task,
            device=args_cli.device,
            num_envs=args_cli.num_envs,
            use_fabric=not args_cli.disable_fabric,
        )
    except gym.error.NameNotFound as exc:
        hint = ""
        if not args_cli.task.endswith("-v0"):
            hint = f"\n提示：你传入的是 `{args_cli.task}`，这个任务名看起来少了版本后缀，试试 `{args_cli.task}-v0`。"
        raise gym.error.NameNotFound(
            f"{exc}\n"
            "这个脚本会自动注册 roboot16/src 下的本地任务。"
            f"{hint}"
        ) from exc
    env = gym.make(args_cli.task, cfg=env_cfg)
    env.reset()

    robot = env.unwrapped.scene["robot"]
    action_manager = env.unwrapped.action_manager
    observation_manager = env.unwrapped.observation_manager
    term = action_manager.get_term("joint_pos")

    print("=" * 80)
    print(f"Task: {args_cli.task}")
    print(f"Robot num_joints: {robot.num_joints}")
    print(f"Action terms: {action_manager.active_terms}")
    print(f"Action term dims: {action_manager.action_term_dim}")
    print(f"Observation groups: {list(observation_manager.active_terms.keys())}")
    print(f"Observation group dims: {observation_manager.group_obs_dim}")
    print("-" * 80)
    print("robot.joint_names:")
    for i, name in enumerate(robot.joint_names):
        print(f"  {i:02d}: {name}")
    print("-" * 80)
    print("joint_pos term resolved _joint_names:")
    for i, name in enumerate(term._joint_names):
        print(f"  {i:02d}: {name}")
    print("-" * 80)
    print(f"joint_pos term _joint_ids: {_to_list(term._joint_ids)}")
    print(f"joint_pos scale: {term._scale}")
    print(f"joint_pos uses default offset shape: {tuple(term._offset.shape) if hasattr(term._offset, 'shape') else term._offset}")
    print("-" * 80)
    if "policy" in observation_manager.active_terms:
        print("policy observation terms (concatenation order):")
        term_names = observation_manager.active_terms["policy"]
        term_dims = observation_manager.group_obs_term_dim["policy"]
        term_cfgs = observation_manager._group_obs_term_cfgs["policy"]
        for i, (name, dims, cfg) in enumerate(zip(term_names, term_dims, term_cfgs, strict=True)):
            print(f"  [{i:02d}] {name}: shape={tuple(dims)}")
            descriptor = _inspect_obs_descriptor(env.unwrapped, cfg)
            joint_names = getattr(descriptor, "joint_names", None) if descriptor is not None else None
            if joint_names is not None:
                for j, joint_name in enumerate(joint_names):
                    print(f"       {j:02d}: {joint_name}")
            elif name in {"actions", "last_action"}:
                print("       uses action_manager.action order (same logical order as action terms)")
                for j, joint_name in enumerate(term._joint_names):
                    print(f"       {j:02d}: {joint_name}")
    else:
        print("No 'policy' observation group found on this task.")
    print("=" * 80)

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
