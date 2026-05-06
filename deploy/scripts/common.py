from __future__ import annotations

from pathlib import Path

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
DEFAULT_MODEL = PROJECT_ROOT / "assets" / "Roboot1.6" / "xml" / "scene_1.xml"
DEFAULT_CHECKPOINT_DIR = PROJECT_ROOT / "logs" / "rsl_rl" / "roboot16_flat_project"
DEFAULT_EXPORT_POLICY = PROJECT_ROOT / "deploy" / "pre_train" / "roboot16" / "policy.pt"

ISAACLAB_JOINT_ORDER = [
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
]

DEFAULT_JOINT_POS = {
    "left_hip_yaw_joint": 0.0,
    "right_hip_yaw_joint": 0.0,
    "left_hip_roll_joint": 0.0,
    "right_hip_roll_joint": 0.0,
    "left_hip_pitch_joint": -0.28,
    "right_hip_pitch_joint": -0.28,
    "left_knee_joint": 0.58,
    "right_knee_joint": 0.58,
    "left_ankle_pitch_joint": -0.32,
    "right_ankle_pitch_joint": -0.32,
    "left_ankle_roll_joint": 0.0,
    "right_ankle_roll_joint": 0.0,
}

PD_GAINS = {
    "hip_yaw": (150.0, 5.0),
    "hip_roll": (150.0, 5.0),
    "hip_pitch": (200.0, 5.0),
    "knee": (200.0, 5.0),
    "ankle_pitch": (20.0, 4.0),
    "ankle_roll": (20.0, 4.0),
}


def default_checkpoint() -> Path:
    checkpoint_paths = sorted(DEFAULT_CHECKPOINT_DIR.glob("**/model_*.pt"), key=lambda path: path.stat().st_mtime)
    if not checkpoint_paths:
        raise FileNotFoundError(
            "No checkpoint found under "
            f"{DEFAULT_CHECKPOINT_DIR}. Please pass --checkpoint explicitly."
        )
    return checkpoint_paths[-1]


def joint_gain(joint_name: str) -> tuple[float, float]:
    for key, gains in PD_GAINS.items():
        if key in joint_name:
            return gains
    return 50.0, 2.0


def joint_names(model) -> list[str]:
    import mujoco

    names = []
    for jid in range(model.njnt):
        names.append(mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, jid) or f"joint_{jid}")
    return names


def body_names(model) -> list[str]:
    import mujoco

    names = []
    for bid in range(model.nbody):
        names.append(mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, bid) or f"body_{bid}")
    return names


def hinge_joint_ids(model) -> list[int]:
    import mujoco

    return [jid for jid in range(model.njnt) if model.jnt_type[jid] == mujoco.mjtJoint.mjJNT_HINGE]


def _ground_geom_ids(model) -> list[int]:
    import mujoco

    geom_ids = []
    for gid in range(model.ngeom):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, gid)
        if name == "ground":
            continue
        if model.geom_contype[gid] == 0 and model.geom_conaffinity[gid] == 0:
            continue
        body_id = model.geom_bodyid[gid]
        body_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id) or ""
        if "ankle" not in body_name and "foot" not in body_name:
            continue
        geom_ids.append(gid)
    return geom_ids


def auto_ground_base_height(model, data, clearance: float = 0.01) -> float:
    import mujoco

    root_qpos = model.jnt_qposadr[0]
    geom_ids = _ground_geom_ids(model)

    data.qpos[root_qpos : root_qpos + 7] = np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0])
    mujoco.mj_forward(model, data)

    lowest_z = min(float(data.geom_xpos[gid, 2] - model.geom_rbound[gid]) for gid in geom_ids)
    return clearance - lowest_z


def apply_standing_pose(model, data, base_height: float | None = None, clearance: float = 0.01) -> None:
    import mujoco

    data.qpos[:] = 0.0
    data.qvel[:] = 0.0

    for jid in hinge_joint_ids(model):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, jid)
        if name in DEFAULT_JOINT_POS:
            data.qpos[model.jnt_qposadr[jid]] = DEFAULT_JOINT_POS[name]

    if model.njnt > 0 and model.jnt_type[0] == mujoco.mjtJoint.mjJNT_FREE:
        root_qpos = model.jnt_qposadr[0]
        resolved_height = auto_ground_base_height(model, data, clearance=clearance) if base_height is None else base_height
        data.qpos[root_qpos : root_qpos + 7] = np.array([0.0, 0.0, resolved_height, 1.0, 0.0, 0.0, 0.0])

    mujoco.mj_forward(model, data)


def sync_model_qpos0(model, data) -> None:
    model.qpos0[:] = data.qpos[:]


def set_default_pose(model, data, base_height: float | None = None, clearance: float = 0.01):
    apply_standing_pose(model, data, base_height=base_height, clearance=clearance)
    sync_model_qpos0(model, data)


def apply_pd(
    model,
    data,
    targets: dict[str, float] | None = None,
    target_scale: float = 1.0,
    kp_scale: float = 1.0,
    kd_scale: float = 1.0,
    gain_overrides: dict[str, tuple[float, float]] | None = None,
):
    import mujoco

    targets = DEFAULT_JOINT_POS if targets is None else targets
    data.qfrc_applied[:] = 0.0
    for jid in hinge_joint_ids(model):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, jid)
        if name not in targets:
            continue
        qadr = model.jnt_qposadr[jid]
        dadr = model.jnt_dofadr[jid]
        kp, kd = gain_overrides.get(name, joint_gain(name)) if gain_overrides else joint_gain(name)
        q = data.qpos[qadr]
        qd = data.qvel[dadr]
        data.qfrc_applied[dadr] = (kp * kp_scale) * (target_scale * targets[name] - q) - (kd * kd_scale) * qd
