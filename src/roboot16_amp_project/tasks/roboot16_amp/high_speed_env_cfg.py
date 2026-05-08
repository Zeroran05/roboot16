# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""High-speed AMP task config for the standalone project.

This variant keeps the base AMP environment parameters unchanged and only adds a
forward-speed curriculum using the same implementation pattern as the standard
Roboot16 velocity task.
"""

import isaaclab.envs.mdp as env_mdp
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.envs import ViewerCfg
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from roboot16_amp_project.tasks.roboot16.flat_env_cfg import (
    _apply_command_term_schedule,
    biped_leg_lateral_separation_penalty,
)
from roboot16_amp_project.tasks.manager_based.locomotion.velocity import mdp

from .flat_env_cfg import (
    Roboot16AmpFlatEnvCfg,
    Roboot16AmpRewards,
    feet_stumble_biped,
    feet_too_near_biped,
    inward_hip_yaw_reward,
    outward_hip_yaw_penalty,
)


PLAY_LIN_VEL_X = (3.5, 4.0)
PLAY_LIN_VEL_Y = (0.0, 0.0)
PLAY_ANG_VEL_Z = (0.0, 0.0)
PLAY_RESAMPLING_TIME_RANGE = (1.0e6, 1.0e6)


def _modify_value_by_schedule(env, env_ids, data, schedule, schedule_name="curriculum"):
    """Apply a step-based schedule to a scalar/tuple term setting."""
    new_value = data
    for trigger_step, target_value in schedule:
        if env.common_step_counter >= trigger_step:
            new_value = target_value
        else:
            break
    if new_value == data:
        return env_mdp.modify_term_cfg.NO_CHANGE
    print(
        f"[Curriculum][{schedule_name}] step={env.common_step_counter}: "
        f"{data} -> {new_value}"
    )
    return new_value


@configclass
class Roboot16AmpHighSpeedRewards(Roboot16AmpRewards):
    """Standalone reward config for the high-speed AMP task.

    The values currently match the base AMP task on purpose, but are duplicated here
    so future high-speed tuning can happen locally in this file.
    """

    termination_penalty = RewTerm(func=mdp.is_terminated, weight=-200.0)
    lin_vel_z_l2 = RewTerm(func=mdp.lin_vel_z_l2, weight=-0.5)
    track_lin_vel_xy_exp = RewTerm(
        func=mdp.track_lin_vel_xy_yaw_frame_exp,
        weight=1.8,
        params={"command_name": "base_velocity", "std": 0.4},
    )
    track_ang_vel_z_exp = RewTerm(
        func=mdp.track_ang_vel_z_world_exp,
        weight=1.2,
        params={"command_name": "base_velocity", "std": 0.4},
    )
    feet_air_time = RewTerm(
        func=mdp.feet_air_time_positive_biped,
        weight=0.6,
        params={
            "command_name": "base_velocity",
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names="(left|right)_ankle_roll_link"),
            "threshold": 0.4,
        },
    )
    feet_mode_time_symmetry = RewTerm(
        func=mdp.feet_mode_time_symmetry_penalty,
        weight=-0.5,
        params={
            "command_name": "base_velocity",
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names="(left|right)_ankle_roll_link"),
            "command_threshold": 0.2,
        },
    )
    feet_trajectory_symmetry = RewTerm(
        func=mdp.feet_trajectory_symmetry_penalty,
        weight=-0.5,
        params={
            "command_name": "base_velocity",
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names="(left|right)_ankle_roll_link"),
            "asset_cfg": SceneEntityCfg("robot", body_names=["left_ankle_roll_link", "right_ankle_roll_link"]),
            "command_threshold": 0.2,
            "phase_sigma": 0.12,
        },
    )
    feet_slide = RewTerm(
        func=mdp.feet_slide,
        weight=-0.9,
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
    joint_deviation_hip_roll_yaw = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.18,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=["(left|right)_hip_roll_joint", "(left|right)_hip_yaw_joint"],
            )
        },
    )
    joint_deviation_ankle_roll = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.20,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=["(left|right)_ankle_roll_joint"])},
    )
    feet_too_near = RewTerm(
        func=feet_too_near_biped,
        weight=-3.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=["left_ankle_roll_link", "right_ankle_roll_link"]),
            "threshold": 0.44,
        },
    )
    knee_lateral_separation = RewTerm(
        func=biped_leg_lateral_separation_penalty,
        weight=-3.0,
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
            "min_knee_y_sep": 0.42,
            "min_ankle_y_sep": 0.0,
        },
    )
    feet_stumble = RewTerm(
        func=feet_stumble_biped,
        weight=-1.5,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=["left_ankle_roll_link", "right_ankle_roll_link"]),
        },
    )
    # 惩罚yaw向外翻转过度，奖励yaw向内翻转，正为内转（有助于高速行走时的步态）
    outward_hip_yaw = RewTerm(
        func=outward_hip_yaw_penalty,
        weight=-1.0,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=["left_hip_yaw_joint", "right_hip_yaw_joint"],
            ),
            "outward_threshold": 0.12,
        },
    )
    inward_hip_yaw = RewTerm(
        func=inward_hip_yaw_reward,
        weight=0.0,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=["left_hip_yaw_joint", "right_hip_yaw_joint"],
            ),
            "target": 0.25,
            "std": 0.10,
        },
    )


@configclass
class Roboot16AmpHighSpeedCurriculumCfg:
    """Curriculum for progressively shifting AMP training toward higher speeds."""
    # 线速度、角速度命令课程
    # schedule的计数对应单个环境的步数
    # 而默认设置中一个iteration对应24步
    # 因此16_800：700iteration；28_800：1200iteration
    lin_vel_x_range = CurrTerm(
        func=_apply_command_term_schedule,
        params={
            "attr_path": "ranges.lin_vel_x",
            "schedule_name": "lin_vel_x",
            "schedule": [
                (0, (0.0, 1.0)),
                (16_800, (0.0, 1.5)),
                (28_800, (0.0, 2.0)),
                (40_800, (0.0, 2.5)),
                (52_800, (0.0, 3.0)),
                (64_800, (0.0, 3.5)),
                (76_800, (0.0, 4.0)),
                (88_800, (0.0, 4.5))
            ],
        },
    )
    # ang_vel_z_range = CurrTerm(
    #     func=_apply_command_term_schedule,
    #     params={
    #         "attr_path": "ranges.ang_vel_z",
    #         "schedule_name": "ang_vel_z",
    #         "schedule": [
    #             (0, (-0.2, 0.2))
    #         ],
    #     },
    # )
    standing_env_ratio = CurrTerm(
        func=_apply_command_term_schedule,
        params={
            "attr_path": "rel_standing_envs",
            "schedule_name": "rel_standing_envs",
            "schedule": [
                (0, 0.2),
                (16_800, 0.18),
                (28_800, 0.15),
                (40_800, 0.12),
                (52_800, 0.1)
            ],
        },
    )
    track_lin_vel_xy_weight = CurrTerm(
        func=env_mdp.modify_term_cfg,
        params={
            "address": "rewards.track_lin_vel_xy_exp.weight",
            "modify_fn": _modify_value_by_schedule,
            "modify_params": {
                "schedule_name": "track_lin_vel_xy_weight",
                "schedule": [
                    (0, 1.8),
                    (16_800, 2.2),
                    (28_800, 2.6),
                    (40_800, 3.0),
                    (52_800, 3.4),
                    (64_800, 3.8),
                    (72_800, 4.2)
                ],
            },
        },
    )
    track_lin_vel_xy_std = CurrTerm(
        func=env_mdp.modify_term_cfg,
        params={
            "address": "rewards.track_lin_vel_xy_exp.params.std",
            "modify_fn": _modify_value_by_schedule,
            "modify_params": {
                "schedule_name": "track_lin_vel_xy_std",
                "schedule": [
                    (0, 0.5),
                    (28_800, 0.45),
                    (52_800, 0.6),
                    (64_800, 0.65),
                    (77_600, 0.60),
                    (80_000, 0.55)
                ],
            },
        },
    )
    # track_ang_vel_z_weight = CurrTerm(
    #     func=env_mdp.modify_term_cfg,
    #     params={
    #         "address": "rewards.track_ang_vel_z_exp.weight",
    #         "modify_fn": _modify_value_by_schedule,
    #         "modify_params": {
    #             "schedule_name": "track_ang_vel_z_weight",
    #             "schedule": [
    #                 (0, 1.2)
    #             ],
    #         },
    #     },
    # )
    # track_ang_vel_z_std = CurrTerm(
    #     func=env_mdp.modify_term_cfg,
    #     params={
    #         "address": "rewards.track_ang_vel_z_exp.params.std",
    #         "modify_fn": _modify_value_by_schedule,
    #         "modify_params": {
    #             "schedule_name": "track_ang_vel_z_std",
    #             "schedule": [
    #                 (0, 0.5),
    #                 (16_800, 0.47),
    #                 (28_800, 0.44),
    #                 (40_800, 0.40),
    #             ],
    #         },
    #     },
    # )


@configclass
class Roboot16AmpHighSpeedEnvCfg(Roboot16AmpFlatEnvCfg):
    """AMP flat task with high-speed command curriculum."""

    rewards: Roboot16AmpHighSpeedRewards = Roboot16AmpHighSpeedRewards()
    curriculum: Roboot16AmpHighSpeedCurriculumCfg = Roboot16AmpHighSpeedCurriculumCfg()

    def __post_init__(self):
        super().__post_init__()
        # Keep the same base environment parameters as the standard AMP task, but
        # explicitly restate them here so high-speed tuning can stay local.
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None
        self.scene.terrain.physics_material.static_friction = 1.1
        self.scene.terrain.physics_material.dynamic_friction = 1.0
        self.scene.height_scanner = None
        self.observations.policy.height_scan = None
        self.curriculum.terrain_levels = None

        if hasattr(self.events, "physics_material") and hasattr(self.events.physics_material, "params"):
            self.events.physics_material.params["static_friction_range"] = (0.85, 1.15)
            self.events.physics_material.params["dynamic_friction_range"] = (0.75, 1.05)
            self.events.physics_material.params["restitution_range"] = (0.0, 0.02)
        if hasattr(self.events, "reset_robot_joints") and hasattr(self.events.reset_robot_joints, "params"):
            self.events.reset_robot_joints.params["position_range"] = (0.95, 1.05)
        if hasattr(self.events, "reset_base"):
            self.events.reset_base.params = {
                "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "z": (0.0, 0.02), "yaw": (-0.3, 0.3)},
                "velocity_range": {
                    "x": (-0.1, 0.1),
                    "y": (-0.05, 0.05),
                    "z": (-0.05, 0.05),
                    "roll": (-0.08, 0.08),
                    "pitch": (-0.08, 0.08),
                    "yaw": (-0.15, 0.15),
                },
            }
        if hasattr(self.events, "add_base_mass") and hasattr(self.events.add_base_mass, "params"):
            self.events.add_base_mass.params["asset_cfg"].body_names = "base_link"
            self.events.add_base_mass.params["mass_distribution_params"] = (-1.0, 1.0)
        if hasattr(self.events, "base_com") and hasattr(self.events.base_com, "params"):
            self.events.base_com.params["asset_cfg"].body_names = "base_link"
            self.events.base_com.params["com_range"] = {"x": (-0.01, 0.01), "y": (-0.01, 0.01), "z": (-0.005, 0.005)}
        if hasattr(self.events, "base_external_force_torque") and hasattr(self.events.base_external_force_torque, "params"):
            self.events.base_external_force_torque.params["asset_cfg"].body_names = "base_link"
            self.events.base_external_force_torque.params["force_range"] = (-12.0, 12.0)
            self.events.base_external_force_torque.params["torque_range"] = (-2.0, 2.0)
        if hasattr(self.events, "push_robot") and hasattr(self.events.push_robot, "params"):
            self.events.push_robot.interval_range_s = (12.0, 18.0)
            self.events.push_robot.params["velocity_range"] = {"x": (-0.15, 0.15), "y": (-0.1, 0.1)}

        self.rewards.undesired_contacts.params["sensor_cfg"].body_names = "(left|right)_(hip_pitch|hip_roll|hip_yaw|knee)_link"
        self.rewards.undesired_contacts.weight = -1.0
        self.rewards.undesired_contacts.params["threshold"] = 0.5
        self.rewards.flat_orientation_l2.weight = -1.0
        self.rewards.dof_torques_l2.weight = 0.0
        self.rewards.action_rate_l2.weight = -0.007
        self.rewards.dof_acc_l2.weight = -1.0e-7

        # These are the task-local initial command settings. The curriculum above
        # will progressively overwrite the tracked fields during training.
        self.commands.base_velocity.rel_standing_envs = 0.2
        self.commands.base_velocity.rel_heading_envs = 0.0
        self.commands.base_velocity.heading_command = False
        self.commands.base_velocity.ranges.lin_vel_x = (0.0, 1.0)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (-0.2, 0.2)
        if hasattr(self.commands.base_velocity.ranges, "heading"):
            self.commands.base_velocity.ranges.heading = (0.0, 0.0)

        if hasattr(self.terminations, "base_contact") and "sensor_cfg" in self.terminations.base_contact.params:
            self.terminations.base_contact.params["sensor_cfg"].body_names = "base_link"
        if hasattr(self.scene.robot.spawn, "activate_contact_sensors"):
            self.scene.robot.spawn.activate_contact_sensors = True


@configclass
class Roboot16AmpHighSpeedEnvCfg_PLAY(Roboot16AmpHighSpeedEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.episode_length_s = 40.0
        self.observations.policy.enable_corruption = False
        self.viewer = ViewerCfg(
            eye=(-5, 3.2, 1.2),# （-3.2, 3.2, 1.2）
            lookat=(0.0, 0.0, 0.45),# (0.0, 0.0, 0.45)
            origin_type="asset_body",
            asset_name="robot",
            body_name="base_link",
        )

        # Disable training-time curriculum updates during evaluation/play.
        self.curriculum.lin_vel_x_range = None
        self.curriculum.ang_vel_z_range = None
        self.curriculum.standing_env_ratio = None
        self.curriculum.track_lin_vel_xy_weight = None
        self.curriculum.track_lin_vel_xy_std = None
        self.curriculum.track_ang_vel_z_weight = None

        # Disable training-time randomization/disturbance events for clean playback.
        self.events.physics_material = None
        self.events.add_base_mass = None
        self.events.base_com = None
        self.events.base_external_force_torque = None
        self.events.push_robot = None

        # Fixed, user-editable play commands.
        self.commands.base_velocity.rel_standing_envs = 0.0
        self.commands.base_velocity.rel_heading_envs = 0.0
        self.commands.base_velocity.heading_command = False
        self.commands.base_velocity.resampling_time_range = PLAY_RESAMPLING_TIME_RANGE
        self.commands.base_velocity.ranges.lin_vel_x = PLAY_LIN_VEL_X
        self.commands.base_velocity.ranges.lin_vel_y = PLAY_LIN_VEL_Y
        self.commands.base_velocity.ranges.ang_vel_z = PLAY_ANG_VEL_Z
        if hasattr(self.commands.base_velocity.ranges, "heading"):
            self.commands.base_velocity.ranges.heading = (0.0, 0.0)
