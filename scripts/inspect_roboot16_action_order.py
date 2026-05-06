#!/usr/bin/env python3
# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Inspect the runtime joint/action order for a roboot16 IsaacLab task.

Example:
    ./isaaclab.sh -p scripts/tools/inspect_roboot16_action_order.py \
        --task Isaac-Velocity-Flat-Roboot16-IdealPD-v0 --num_envs 1 --headless
"""

import argparse

from isaaclab.app import AppLauncher


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

import gymnasium as gym

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg


def _to_list(value):
    if isinstance(value, slice):
        return "slice(None)"
    if hasattr(value, "tolist"):
        return value.tolist()
    return list(value)


def main():
    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric,
    )
    env = gym.make(args_cli.task, cfg=env_cfg)
    env.reset()

    robot = env.unwrapped.scene["robot"]
    action_manager = env.unwrapped.action_manager
    term = action_manager.get_term("joint_pos")

    print("=" * 80)
    print(f"Task: {args_cli.task}")
    print(f"Robot num_joints: {robot.num_joints}")
    print(f"Action terms: {action_manager.active_terms}")
    print(f"Action term dims: {action_manager.action_term_dim}")
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
    print("=" * 80)

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
