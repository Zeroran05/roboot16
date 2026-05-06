"""Project-local Roboot16 asset configuration."""

from __future__ import annotations

from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg


ROBOT_USD_PATH = Path(__file__).resolve().parent / "usd" / "Roboot16_1.usd"


ROBOOT_CFG = ArticulationCfg(
    prim_path="{ENV_REGEX_NS}/Robot",
    spawn=sim_utils.UsdFileCfg(
        usd_path=str(ROBOT_USD_PATH),
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
            enable_gyroscopic_forces=True,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=4,
            solver_velocity_iteration_count=4,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.96),
        joint_pos={
            ".*_hip_yaw_joint": 0.0,
            ".*_hip_roll_joint": 0.0,
            ".*_hip_pitch_joint": -0.28,
            ".*_knee_joint": 0.58,
            ".*_ankle_pitch_joint": -0.32,
            ".*_ankle_roll_joint": 0.0,
        },
        joint_vel={".*": 0.0},
    ),
    actuators={
        "hips": ImplicitActuatorCfg(
            joint_names_expr=[
                "left_hip_pitch_joint",
                "right_hip_pitch_joint",
                "left_hip_roll_joint",
                "right_hip_roll_joint",
                "left_hip_yaw_joint",
                "right_hip_yaw_joint",
            ],
            effort_limit_sim=300,
            stiffness={
                ".*_hip_yaw_joint": 150.0,
                ".*_hip_roll_joint": 150.0,
                ".*_hip_pitch_joint": 200.0,
            },
            damping={
                ".*_hip_yaw_joint": 5.0,
                ".*_hip_roll_joint": 5.0,
                ".*_hip_pitch_joint": 5.0,
            },
        ),
        "knees": ImplicitActuatorCfg(
            joint_names_expr=["left_knee_joint", "right_knee_joint"],
            effort_limit_sim=300,
            stiffness={".*": 200.0},
            damping={".*": 5.0},
        ),
        "ankle_pitch": ImplicitActuatorCfg(
            joint_names_expr=["left_ankle_pitch_joint", "right_ankle_pitch_joint"],
            effort_limit_sim=100,
            stiffness={".*": 20.0},
            damping={".*": 4.0},
        ),
        "ankle_roll": ImplicitActuatorCfg(
            joint_names_expr=["left_ankle_roll_joint", "right_ankle_roll_joint"],
            effort_limit_sim=100,
            stiffness={".*": 20.0},
            damping={".*": 4.0},
        ),
    },
)
