# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Project-local Roboot16 flat locomotion task config."""


import isaaclab.envs.mdp as env_mdp
import torch
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import quat_apply_inverse, yaw_quat
from isaaclab_tasks.manager_based.locomotion.velocity.velocity_env_cfg import LocomotionVelocityRoughEnvCfg, RewardsCfg

from roboot16_amp_project.assets.robots import ROBOOT_CFG
from roboot16_amp_project.tasks.manager_based.locomotion.velocity import mdp


def _apply_step_schedule(env, env_ids, old_value, schedule, schedule_name="curriculum"):
    new_value = old_value
    for trigger_step, target_value in schedule:
        if env.common_step_counter >= trigger_step:
            new_value = target_value
        else:
            break
    if new_value == old_value:
        return env_mdp.modify_env_param.NO_CHANGE
    print(
        f"[Curriculum][{schedule_name}] step={env.common_step_counter}: "
        f"{old_value} -> {new_value}"
    )
    return new_value


def _set_nested_attr(obj, attr_path, value):
    target = obj
    parts = attr_path.split(".")
    for part in parts[:-1]:
        target = getattr(target, part)
    setattr(target, parts[-1], value)


def _get_command_cfg_value(command_term, attr_path):
    target = command_term.cfg
    for part in attr_path.split("."):
        target = getattr(target, part)
    return target


def _apply_command_term_schedule(
    env,
    env_ids,
    attr_path,
    schedule,
    schedule_name,
    command_name="base_velocity",
):
    command_term = env.command_manager.get_term(command_name)
    old_value = _get_command_cfg_value(command_term, attr_path)
    new_value = _apply_step_schedule(env, env_ids, old_value, schedule, schedule_name=schedule_name)
    if new_value is env_mdp.modify_env_param.NO_CHANGE:
        return old_value

    _set_nested_attr(env.cfg.commands.__dict__[command_name], attr_path, new_value)
    _set_nested_attr(command_term.cfg, attr_path, new_value)

    if isinstance(env_ids, slice):
        env_ids = list(range(env.num_envs))
    command_term._resample(env_ids)
    live_value = _get_command_cfg_value(command_term, attr_path)
    print(
        f"[Curriculum][{schedule_name}] live_{command_name}.{attr_path}={live_value}, "
        f"resampled_envs={len(env_ids)}"
    )
    return new_value


def biped_leg_lateral_separation_penalty(
    env,
    asset_cfg: SceneEntityCfg,
    min_knee_y_sep: float = 0.10,
    min_ankle_y_sep: float = 0.14,
):
    asset = env.scene[asset_cfg.name]
    body_pos_w = asset.data.body_pos_w[:, asset_cfg.body_ids, :]
    root_pos_w = asset.data.root_pos_w.unsqueeze(1)
    root_yaw_quat = yaw_quat(asset.data.root_quat_w).unsqueeze(1).expand(-1, len(asset_cfg.body_ids), -1)
    body_pos_yaw = quat_apply_inverse(root_yaw_quat, body_pos_w - root_pos_w)

    left_knee_y = body_pos_yaw[:, 0, 1]
    right_knee_y = body_pos_yaw[:, 1, 1]
    left_ankle_y = body_pos_yaw[:, 2, 1]
    right_ankle_y = body_pos_yaw[:, 3, 1]

    knee_sep = torch.abs(left_knee_y - right_knee_y)
    ankle_sep = torch.abs(left_ankle_y - right_ankle_y)
    knee_penalty = torch.clamp(min_knee_y_sep - knee_sep, min=0.0)
    ankle_penalty = torch.clamp(min_ankle_y_sep - ankle_sep, min=0.0)
    return knee_penalty + ankle_penalty


def inward_hip_roll_penalty(
    env,
    asset_cfg: SceneEntityCfg,
    inward_threshold: float = -0.02,
):
    asset = env.scene[asset_cfg.name]
    hip_roll_pos = asset.data.joint_pos[:, asset_cfg.joint_ids]
    return torch.sum(torch.clamp(inward_threshold - hip_roll_pos, min=0.0), dim=1)


def inward_hip_yaw_penalty(
    env,
    asset_cfg: SceneEntityCfg,
    inward_threshold: float = -0.02,
):
    asset = env.scene[asset_cfg.name]
    hip_yaw_pos = asset.data.joint_pos[:, asset_cfg.joint_ids]
    return torch.sum(torch.clamp(inward_threshold - hip_yaw_pos, min=0.0), dim=1)


def high_speed_outward_hip_yaw_reward(
    env,
    command_name: str,
    asset_cfg: SceneEntityCfg,
    speed_threshold: float = 2.0,
    target: float = 0.3,
    std: float = 0.12,
):
    asset = env.scene[asset_cfg.name]
    hip_yaw_pos = asset.data.joint_pos[:, asset_cfg.joint_ids]
    cmd = env.command_manager.get_command(command_name)
    forward_speed = torch.abs(cmd[:, 0])
    speed_mask = (forward_speed >= speed_threshold).float()
    yaw_error = torch.mean(torch.square(hip_yaw_pos - target), dim=1)
    reward = torch.exp(-yaw_error / (std**2))
    return reward * speed_mask


@configclass
class Roboot16Rewards(RewardsCfg):
    termination_penalty = RewTerm(func=mdp.is_terminated, weight=-200.0)
    track_lin_vel_xy_exp = RewTerm(
        func=mdp.track_lin_vel_xy_yaw_frame_exp,
        weight=3.0,
        params={"command_name": "base_velocity", "std": 0.4},
    )
    track_ang_vel_z_exp = RewTerm(
        func=mdp.track_ang_vel_z_world_exp, weight=1.4, params={"command_name": "base_velocity", "std": 0.5}
    )
    feet_air_time = RewTerm(
        func=mdp.feet_air_time_positive_biped,
        weight=0.5,
        params={
            "command_name": "base_velocity",
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names="(left|right)_ankle_roll_link"),
            "threshold": 0.4,
        },
    )
    feet_slide = RewTerm(
        func=mdp.feet_slide,
        weight=-0.15,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names="(left|right)_ankle_roll_link"),
            "asset_cfg": SceneEntityCfg("robot", body_names="(left|right)_ankle_roll_link"),
        },
    )
    dof_pos_limits = RewTerm(
        func=mdp.joint_pos_limits,
        weight=-1.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names="(left|right)_ankle_.*")},
    )
    joint_deviation_hip_roll = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.4,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=["(left|right)_hip_roll_joint"])},
    )
    joint_deviation_hip_yaw = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.05,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=["(left|right)_hip_yaw_joint"])},
    )
    joint_deviation_ankle_roll = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.2,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=["(left|right)_ankle_roll_joint"])},
    )
    leg_lateral_separation = RewTerm(
        func=biped_leg_lateral_separation_penalty,
        weight=-1.5,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                body_names=[
                    "left_knee_link",
                    "right_knee_link",
                    "left_ankle_roll_link",
                    "right_ankle_roll_link",
                ],
            ),
            "min_knee_y_sep": 0.08,
            "min_ankle_y_sep": 0.18,
        },
    )
    inward_hip_roll = RewTerm(
        func=inward_hip_roll_penalty,
        weight=-0.4,
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=["left_hip_roll_joint", "right_hip_roll_joint"]),
            "inward_threshold": -0.02,
        },
    )
    inward_hip_yaw = RewTerm(
        func=inward_hip_yaw_penalty,
        weight=-1.2,
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=["(left|right)_hip_yaw_joint"]),
            "inward_threshold": 0.2,
        },
    )
    high_speed_outward_hip_yaw = RewTerm(
        func=high_speed_outward_hip_yaw_reward,
        weight=0.9,
        params={
            "command_name": "base_velocity",
            "asset_cfg": SceneEntityCfg("robot", joint_names=["(left|right)_hip_yaw_joint"]),
            "speed_threshold": 1.5,
            "target": 0.3,
            "std": 0.12,
        },
    )


@configclass
class Roboot16CurriculumCfg:
    lin_vel_x_range = CurrTerm(
        func=_apply_command_term_schedule,
        params={
            "attr_path": "ranges.lin_vel_x",
            "schedule_name": "lin_vel_x",
            "schedule": [
                (0, (0.8, 1.8)),
                (7_200, (1.3, 2.3)),
                (16_800, (1.8, 2.8)),
            ],
        },
    )
    ang_vel_z_range = CurrTerm(
        func=_apply_command_term_schedule,
        params={
            "attr_path": "ranges.ang_vel_z",
            "schedule_name": "ang_vel_z",
            "schedule": [
                (0, (-0.3, 0.3)),
                (2_400, (-0.5, 0.5)),
                (4_800, (-0.6, 0.6)),
                (7_200, (-0.7, 0.7)),
            ],
        },
    )
    standing_env_ratio = CurrTerm(
        func=_apply_command_term_schedule,
        params={
            "attr_path": "rel_standing_envs",
            "schedule_name": "rel_standing_envs",
            "schedule": [
                (0, 0.08),
                (4_200, 0.04),
                (7_200, 0.02),
            ],
        },
    )


@configclass
class Roboot16FlatEnvCfg(LocomotionVelocityRoughEnvCfg):
    rewards: Roboot16Rewards = Roboot16Rewards()
    curriculum: Roboot16CurriculumCfg = Roboot16CurriculumCfg()

    def __post_init__(self):
        super().__post_init__()
        self.scene.robot = ROBOOT_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None
        self.scene.terrain.physics_material.static_friction = 1.1
        self.scene.terrain.physics_material.dynamic_friction = 1.0
        self.scene.height_scanner = None
        self.observations.policy.height_scan = None
        self.curriculum.terrain_levels = None

        if hasattr(self.events, "physics_material") and hasattr(self.events.physics_material, "params"):
            self.events.physics_material.params["static_friction_range"] = (0.7, 1.3)
            self.events.physics_material.params["dynamic_friction_range"] = (0.6, 1.2)
            self.events.physics_material.params["restitution_range"] = (0.0, 0.05)
        if hasattr(self.events, "reset_robot_joints") and hasattr(self.events.reset_robot_joints, "params"):
            self.events.reset_robot_joints.params["position_range"] = (0.95, 1.05)
        if hasattr(self.events, "reset_base"):
            self.events.reset_base.params = {
                "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "z": (0.00, 0.02), "yaw": (-0.35, 0.35)},
                "velocity_range": {
                    "x": (-0.2, 0.2),
                    "y": (-0.1, 0.1),
                    "z": (-0.1, 0.1),
                    "roll": (-0.15, 0.15),
                    "pitch": (-0.15, 0.15),
                    "yaw": (-0.3, 0.3),
                },
            }
        if hasattr(self.events, "add_base_mass") and hasattr(self.events.add_base_mass, "params"):
            self.events.add_base_mass.params["asset_cfg"].body_names = "base_link"
            self.events.add_base_mass.params["mass_distribution_params"] = (-2.0, 2.0)
        if hasattr(self.events, "base_com") and hasattr(self.events.base_com, "params"):
            self.events.base_com.params["asset_cfg"].body_names = "base_link"
            self.events.base_com.params["com_range"] = {"x": (-0.02, 0.02), "y": (-0.02, 0.02), "z": (-0.01, 0.01)}
        if hasattr(self.events, "base_external_force_torque") and hasattr(self.events.base_external_force_torque, "params"):
            self.events.base_external_force_torque.params["asset_cfg"].body_names = "base_link"
            self.events.base_external_force_torque.params["force_range"] = (-30.0, 30.0)
            self.events.base_external_force_torque.params["torque_range"] = (-5.0, 5.0)
        if hasattr(self.events, "push_robot") and hasattr(self.events.push_robot, "params"):
            self.events.push_robot.interval_range_s = (8.0, 14.0)
            self.events.push_robot.params["velocity_range"] = {"x": (-0.3, 0.3), "y": (-0.2, 0.2)}

        self.rewards.undesired_contacts.params["sensor_cfg"].body_names = "(left|right)_(hip_pitch|hip_roll|hip_yaw|knee)_link"
        self.rewards.undesired_contacts.weight = -1.5
        self.rewards.undesired_contacts.params["threshold"] = 0.2
        self.rewards.flat_orientation_l2.weight = -0.4
        self.rewards.dof_torques_l2.weight = 0.0
        self.rewards.action_rate_l2.weight = -0.005
        self.rewards.dof_acc_l2.weight = -1.0e-7

        self.commands.base_velocity.rel_standing_envs = 0.08
        self.commands.base_velocity.rel_heading_envs = 0.0
        self.commands.base_velocity.heading_command = False
        self.commands.base_velocity.ranges.lin_vel_x = (0.8, 1.5)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (-0.4, 0.4)
        self.commands.base_velocity.resampling_time_range = (12.0, 12.0)
        if hasattr(self.commands.base_velocity.ranges, "heading"):
            self.commands.base_velocity.ranges.heading = (0.0, 0.0)

        if hasattr(self.terminations, "base_contact") and "sensor_cfg" in self.terminations.base_contact.params:
            self.terminations.base_contact.params["sensor_cfg"].body_names = "base_link"
        if hasattr(self.scene.robot.spawn, "activate_contact_sensors"):
            self.scene.robot.spawn.activate_contact_sensors = True


@configclass
class Roboot16FlatEnvCfg_PLAY(Roboot16FlatEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.episode_length_s = 40.0
        self.observations.policy.enable_corruption = False
        self.events.base_external_force_torque = None
        self.events.push_robot = None
