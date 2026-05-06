# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Roboot16 AMP-compatible flat velocity task config for the standalone project."""

import torch

from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor
from isaaclab.utils import configclass
from isaaclab.utils.math import quat_apply_inverse

from isaaclab_tasks.manager_based.locomotion.velocity.velocity_env_cfg import (
    LocomotionVelocityRoughEnvCfg,
    ObservationsCfg as BaseObservationsCfg,
    RewardsCfg,
)

from roboot16_amp_project.assets.robots import ROBOOT_CFG
from roboot16_amp_project.tasks.manager_based.locomotion.velocity import mdp


AMP_JOINT_NAMES = [
    "left_hip_pitch_joint",
    "right_hip_pitch_joint",
    "left_hip_roll_joint",
    "right_hip_roll_joint",
    "left_hip_yaw_joint",
    "right_hip_yaw_joint",
    "left_knee_joint",
    "right_knee_joint",
    "left_ankle_pitch_joint",
    "right_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_ankle_roll_joint",
]

AMP_FOOT_BODY_NAMES = [
    "left_ankle_roll_link",
    "right_ankle_roll_link",
]


def feet_too_near_biped(
    env,
    threshold: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    assert len(asset_cfg.body_ids) == 2
    asset = env.scene[asset_cfg.name]
    foot_pos = asset.data.body_pos_w[:, asset_cfg.body_ids, :]
    distance = torch.norm(foot_pos[:, 0] - foot_pos[:, 1], dim=-1)
    return torch.clamp(threshold - distance, min=0.0)


def feet_stumble_biped(env, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    return torch.any(
        torch.norm(contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, :2], dim=2)
        > 5 * torch.abs(contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, 2]),
        dim=1,
    )


def outward_hip_yaw_penalty(
    env,
    asset_cfg: SceneEntityCfg,
    outward_threshold: float = -0.08,
) -> torch.Tensor:
    asset = env.scene[asset_cfg.name]
    hip_yaw_pos = asset.data.joint_pos[:, asset_cfg.joint_ids]
    return torch.sum(torch.clamp(outward_threshold - hip_yaw_pos, min=0.0), dim=1)


def inward_hip_yaw_reward(
    env,
    asset_cfg: SceneEntityCfg,
    target: float = 0.10,
    std: float = 0.10,
) -> torch.Tensor:
    asset = env.scene[asset_cfg.name]
    hip_yaw_pos = asset.data.joint_pos[:, asset_cfg.joint_ids]
    yaw_error = torch.mean(torch.square(hip_yaw_pos - target), dim=1)
    return torch.exp(-yaw_error / (std**2))


def amp_body_pos_b(
    env,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    asset = env.scene[asset_cfg.name]
    body_pos_w = asset.data.body_pos_w[:, asset_cfg.body_ids, :]
    root_pos_w = asset.data.root_pos_w.unsqueeze(1)
    root_quat_w = asset.data.root_quat_w.unsqueeze(1).expand(-1, len(asset_cfg.body_ids), -1)
    body_pos_b = quat_apply_inverse(root_quat_w, body_pos_w - root_pos_w)
    return body_pos_b.reshape(body_pos_b.shape[0], -1)


@configclass
class Roboot16AmpObservationsCfg(BaseObservationsCfg):
    """Policy obs stay the same; AMP keeps Isaac Sim's observation ordering.

    The expert loader is responsible for remapping MuJoCo-exported joint order into this
    observation order before discriminator training.
    """

    @configclass
    class AmpCfg(ObsGroup):
        joint_pos = ObsTerm(
            func=mdp.joint_pos,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=AMP_JOINT_NAMES)},
        )
        joint_vel = ObsTerm(
            func=mdp.joint_vel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=AMP_JOINT_NAMES)},
        )
        foot_pos = ObsTerm(
            func=amp_body_pos_b,
            params={"asset_cfg": SceneEntityCfg("robot", body_names=AMP_FOOT_BODY_NAMES)},
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    amp: AmpCfg = AmpCfg()


@configclass
class Roboot16AmpRewards(RewardsCfg):
    termination_penalty = RewTerm(func=mdp.is_terminated, weight=-200.0)
    lin_vel_z_l2 = RewTerm(func=mdp.lin_vel_z_l2, weight=-0.5)
    track_lin_vel_xy_exp = RewTerm(
        func=mdp.track_lin_vel_xy_yaw_frame_exp,
        weight=1.8,
        params={"command_name": "base_velocity", "std": 0.5},
    )
    track_ang_vel_z_exp = RewTerm(
        func=mdp.track_ang_vel_z_world_exp,
        weight=1.2,
        params={"command_name": "base_velocity", "std": 0.5},
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
    feet_slide = RewTerm(
        func=mdp.feet_slide,
        weight=-0.25,
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
        weight=-0.2,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=["(left|right)_hip_roll_joint", "(left|right)_hip_yaw_joint"],
            )
        },
    )
    joint_deviation_ankle_roll = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.25,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=["(left|right)_ankle_roll_joint"])},
    )
    feet_too_near = RewTerm(
        func=feet_too_near_biped,
        weight=-3.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=["left_ankle_roll_link", "right_ankle_roll_link"]),
            "threshold": 0.32,
        },
    )
    feet_stumble = RewTerm(
        func=feet_stumble_biped,
        weight=-1.5,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=["left_ankle_roll_link", "right_ankle_roll_link"]),
        },
    )
    outward_hip_yaw = RewTerm(
        func=outward_hip_yaw_penalty,
        weight=-3,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=["left_hip_yaw_joint", "right_hip_yaw_joint"],
            ),
            "outward_threshold": 0.22,
        },
    )
    inward_hip_yaw = RewTerm(
        func=inward_hip_yaw_reward,
        weight=0.5,
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
class Roboot16AmpFlatEnvCfg(LocomotionVelocityRoughEnvCfg):
    observations: Roboot16AmpObservationsCfg = Roboot16AmpObservationsCfg()
    rewards: Roboot16AmpRewards = Roboot16AmpRewards()

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

        self.commands.base_velocity.rel_standing_envs = 0.2
        self.commands.base_velocity.rel_heading_envs = 0.0
        self.commands.base_velocity.heading_command = False
        self.commands.base_velocity.ranges.lin_vel_x = (0.4, 1.5)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (-0.6, 0.6)
        if hasattr(self.commands.base_velocity.ranges, "heading"):
            self.commands.base_velocity.ranges.heading = (0.0, 0.0)

        if hasattr(self.terminations, "base_contact") and "sensor_cfg" in self.terminations.base_contact.params:
            self.terminations.base_contact.params["sensor_cfg"].body_names = "base_link"
        if hasattr(self.scene.robot.spawn, "activate_contact_sensors"):
            self.scene.robot.spawn.activate_contact_sensors = True


@configclass
class Roboot16AmpFlatEnvCfg_PLAY(Roboot16AmpFlatEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.episode_length_s = 40.0
        self.observations.policy.enable_corruption = False
        self.events.base_external_force_torque = None
        self.events.push_robot = None
