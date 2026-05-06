import argparse
import time
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np
import torch
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = Path(__file__).resolve().parent / "configs"

LEFT_IDS = np.array([0, 1, 2, 3, 4, 5], dtype=np.int32)
RIGHT_IDS = np.array([6, 7, 8, 9, 10, 11], dtype=np.int32)


def resolve_path(path_str: str) -> str:
    return str(Path(path_str.replace("{PROJECT_ROOT}", str(PROJECT_ROOT))).expanduser().resolve())


def build_joint_maps(model: mujoco.MjModel, joint_order: list[str] | None):
    """Resolve qpos / dof / actuator indices for the requested joint order."""
    num_actions = len(joint_order) if joint_order is not None else model.nu

    if joint_order is None:
        qpos_adrs = [model.jnt_qposadr[jid] for jid in range(1, num_actions + 1)]
        dof_adrs = [model.jnt_dofadr[jid] for jid in range(1, num_actions + 1)]
        act_ids = list(range(num_actions))
        return qpos_adrs, dof_adrs, act_ids

    joint_name_to_jid = {}
    for jid in range(model.njnt):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, jid)
        if name is not None:
            joint_name_to_jid[name] = jid

    joint_to_actuator = {}
    for aid in range(model.nu):
        trn_type = model.actuator_trntype[aid]
        if trn_type != mujoco.mjtTrn.mjTRN_JOINT:
            continue
        joint_to_actuator[model.actuator_trnid[aid, 0]] = aid

    qpos_adrs = []
    dof_adrs = []
    act_ids = []
    for name in joint_order:
        if name not in joint_name_to_jid:
            raise KeyError(f"Joint '{name}' not found in MuJoCo model.")
        jid = joint_name_to_jid[name]
        if jid not in joint_to_actuator:
            raise KeyError(f"Joint '{name}' has no joint actuator bound in MuJoCo model.")
        qpos_adrs.append(model.jnt_qposadr[jid])
        dof_adrs.append(model.jnt_dofadr[jid])
        act_ids.append(joint_to_actuator[jid])

    return qpos_adrs, dof_adrs, act_ids


def initialize_pose(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    qpos_adrs: list[int],
    default_angles: np.ndarray,
    auto_ground: bool = False,
    clearance: float = 0.01,
):
    """Set default pose and optionally auto-place the free base above the ground."""
    data.qpos[:] = model.qpos0.copy()
    data.qvel[:] = 0.0

    for qadr, angle in zip(qpos_adrs, default_angles):
        data.qpos[qadr] = float(angle)

    if auto_ground and model.njnt > 0 and model.jnt_type[0] == mujoco.mjtJoint.mjJNT_FREE:
        root_qadr = model.jnt_qposadr[0]
        data.qpos[root_qadr + 2] = 0.0
        mujoco.mj_forward(model, data)

        lowest_z = None
        for gid in range(model.ngeom):
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, gid) or ""
            if name in {"floor", "ground"}:
                continue
            if model.geom_contype[gid] == 0 and model.geom_conaffinity[gid] == 0:
                continue
            z = float(data.geom_xpos[gid, 2] - model.geom_rbound[gid])
            lowest_z = z if lowest_z is None else min(lowest_z, z)

        if lowest_z is not None:
            data.qpos[root_qadr + 2] = clearance - lowest_z

    mujoco.mj_forward(model, data)


def get_gravity_orientation(quaternion):
    qw = quaternion[0]
    qx = quaternion[1]
    qy = quaternion[2]
    qz = quaternion[3]

    gravity_orientation = np.zeros(3, dtype=np.float32)
    gravity_orientation[0] = 2 * (-qz * qx + qw * qy)
    gravity_orientation[1] = -2 * (qz * qy + qw * qx)
    gravity_orientation[2] = 1 - 2 * (qw * qw + qz * qz)
    return gravity_orientation


def quat_wxyz_to_rotmat(quaternion):
    qw = quaternion[0]
    qx = quaternion[1]
    qy = quaternion[2]
    qz = quaternion[3]

    xx, yy, zz = qx * qx, qy * qy, qz * qz
    xy, xz, yz = qx * qy, qx * qz, qy * qz
    wx, wy, wz = qw * qx, qw * qy, qw * qz

    return np.array(
        [
            [1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy)],
            [2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)],
            [2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy)],
        ],
        dtype=np.float32,
    )


def pd_control(target_q, q, kp, target_dq, dq, kd):
    """Calculate torque from position targets."""
    return (target_q - q) * kp + (target_dq - dq) * kd


def build_observation(
    config: dict,
    d: mujoco.MjData,
    default_angles: np.ndarray,
    last_action: np.ndarray,
    cmd: np.ndarray,
    counter: int,
    simulation_dt: float,
    qpos_adrs: list[int] | None = None,
    dof_adrs: list[int] | None = None,
) -> np.ndarray:
    num_actions = config["num_actions"]
    num_obs = config["num_obs"]
    observation_mode = config.get("observation_mode", "phase_humanoid")

    lin_vel_scale = config.get("lin_vel_scale", 1.0)
    ang_vel_scale = config["ang_vel_scale"]
    dof_pos_scale = config["dof_pos_scale"]
    dof_vel_scale = config["dof_vel_scale"]
    cmd_scale = np.array(config["cmd_scale"], dtype=np.float32)

    if qpos_adrs is None:
        qj = d.qpos[7:].astype(np.float32, copy=True)
    else:
        qj = d.qpos[qpos_adrs].astype(np.float32, copy=True)

    if dof_adrs is None:
        dqj = d.qvel[6:].astype(np.float32, copy=True)
    else:
        dqj = d.qvel[dof_adrs].astype(np.float32, copy=True)

    quat = d.qpos[3:7].astype(np.float32, copy=True)

    rot_wb = quat_wxyz_to_rotmat(quat)
    rot_bw = rot_wb.T
    base_lin_vel_w = d.qvel[:3].astype(np.float32, copy=True)
    base_lin_vel = (rot_bw @ base_lin_vel_w) * lin_vel_scale
    omega = d.qvel[3:6].astype(np.float32, copy=True) * ang_vel_scale
    gravity_orientation = get_gravity_orientation(quat)

    qj = (qj - default_angles) * dof_pos_scale
    dqj = dqj * dof_vel_scale

    obs = np.zeros(num_obs, dtype=np.float32)
    cursor = 0

    if observation_mode == "manager_based_velocity":
        obs[cursor : cursor + 3] = base_lin_vel
        cursor += 3

    obs[cursor : cursor + 3] = omega
    cursor += 3
    obs[cursor : cursor + 3] = gravity_orientation
    cursor += 3
    obs[cursor : cursor + 3] = cmd * cmd_scale
    cursor += 3
    obs[cursor : cursor + num_actions] = qj
    cursor += num_actions
    obs[cursor : cursor + num_actions] = dqj
    cursor += num_actions
    obs[cursor : cursor + num_actions] = last_action
    cursor += num_actions

    if observation_mode == "phase_humanoid":
        period = config.get("phase_period", 0.8)
        count = counter * simulation_dt
        phase = count % period / period
        obs[cursor : cursor + 2] = np.array([np.sin(2 * np.pi * phase), np.cos(2 * np.pi * phase)], dtype=np.float32)
        cursor += 2

    if cursor != num_obs:
        raise ValueError(
            f"Observation size mismatch for mode={observation_mode}: built {cursor}, expected {num_obs}. "
            "Please check yaml observation settings."
        )
    return obs


def foot_contact_flags(model: mujoco.MjModel, data: mujoco.MjData):
    left = False
    right = False
    for i in range(data.ncon):
        con = data.contact[i]
        g1 = int(con.geom1)
        g2 = int(con.geom2)
        n1 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g1) or ""
        n2 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g2) or ""
        names = (n1, n2)
        if any("left_foot_" in n for n in names):
            left = True
        if any("right_foot_" in n for n in names):
            right = True
    return left, right


def interpolate_command(
    cmd_init: np.ndarray,
    cmd_target: np.ndarray,
    sim_time: float,
    ramp_start_time: float,
    ramp_time: float,
) -> np.ndarray:
    """Linearly ramp the velocity command from cmd_init to cmd_target."""
    if ramp_time <= 0.0:
        return cmd_target.copy()
    if sim_time <= ramp_start_time:
        return cmd_init.copy()
    alpha = np.clip((sim_time - ramp_start_time) / ramp_time, 0.0, 1.0)
    return ((1.0 - alpha) * cmd_init + alpha * cmd_target).astype(np.float32)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config_file", type=str, help="config file name in the config folder")
    args = parser.parse_args()

    config_path = CONFIG_DIR / args.config_file
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    policy_path = resolve_path(config["policy_path"])
    xml_path = resolve_path(config["xml_path"])
    control_mode = config.get("control_mode", "torque_pd")

    simulation_duration = config["simulation_duration"]
    simulation_dt = config["simulation_dt"]
    control_decimation = config["control_decimation"]

    kps = np.array(config["kps"], dtype=np.float32) * float(config.get("kp_scale", 1.0))
    kds = np.array(config["kds"], dtype=np.float32) * float(config.get("kd_scale", 1.0))
    default_angles = np.array(config["default_angles"], dtype=np.float32)

    action_scale = config["action_scale"]
    num_actions = config["num_actions"]
    joint_order = config.get("joint_order")
    action_signs = np.array(config.get("action_signs", [1.0] * num_actions), dtype=np.float32)
    action_gains = np.array(config.get("action_gains", [1.0] * num_actions), dtype=np.float32)
    action_permutation = np.array(config.get("action_permutation", list(range(num_actions))), dtype=np.int32)
    cmd_init = np.array(config["cmd_init"], dtype=np.float32)
    cmd_target = np.array(config.get("cmd_target", config["cmd_init"]), dtype=np.float32)
    cmd_ramp_start_time = float(config.get("cmd_ramp_start_time", 0.0))
    cmd_ramp_time = float(config.get("cmd_ramp_time", 0.0))
    cmd = cmd_init.copy()

    zero_action = bool(config.get("zero_action", False))
    policy_start_time = float(config.get("policy_start_time", 0.0))
    init_auto_ground = bool(config.get("init_auto_ground", False))
    init_clearance = float(config.get("init_clearance", 0.01))
    action_alpha = float(config.get("action_alpha", 1.0))
    action_ramp_time = float(config.get("action_ramp_time", 0.0))
    debug_trace = bool(config.get("debug_trace", False))
    debug_trace_every = int(config.get("debug_trace_every", 10))
    show_tracking = bool(config.get("show_tracking", False))
    tracking_every = int(config.get("tracking_every", 10))
    camera_follow = bool(config.get("camera_follow", False))
    camera_lookat_height = float(config.get("camera_lookat_height", 0.75))
    camera_distance = float(config.get("camera_distance", 3.0))
    camera_azimuth = float(config.get("camera_azimuth", 135.0))
    camera_elevation = float(config.get("camera_elevation", -20.0))

    action = np.zeros(num_actions, dtype=np.float32)
    raw_action = np.zeros(num_actions, dtype=np.float32)
    last_policy_action = np.zeros(num_actions, dtype=np.float32)
    target_dof_pos = default_angles.copy()
    counter = 0
    tau = np.zeros(num_actions, dtype=np.float32)

    policy_start_step = int(round(policy_start_time / simulation_dt))
    if action_ramp_time > 0.0:
        action_ramp_steps = max(1, int(round(action_ramp_time / simulation_dt)))
    else:
        action_ramp_steps = 0

    m = mujoco.MjModel.from_xml_path(xml_path)
    d = mujoco.MjData(m)
    m.opt.timestep = simulation_dt
    qpos_adrs, dof_adrs, act_ids = build_joint_maps(m, joint_order)
    initialize_pose(m, d, qpos_adrs, default_angles, auto_ground=init_auto_ground, clearance=init_clearance)

    policy = torch.jit.load(policy_path, map_location="cpu")
    policy.eval()

    with mujoco.viewer.launch_passive(m, d) as viewer:
        if camera_follow:
            viewer.cam.distance = camera_distance
            viewer.cam.azimuth = camera_azimuth
            viewer.cam.elevation = camera_elevation
            viewer.cam.lookat[:] = np.array(
                [d.qpos[0], d.qpos[1], d.qpos[2] + camera_lookat_height], dtype=np.float64
            )
        start = time.time()
        while viewer.is_running() and time.time() - start < simulation_duration:
            step_start = time.time()
            sim_time = counter * simulation_dt
            cmd = interpolate_command(cmd_init, cmd_target, sim_time, cmd_ramp_start_time, cmd_ramp_time)

            q = d.qpos[qpos_adrs]
            dq = d.qvel[dof_adrs]

            d.ctrl[:] = 0.0
            if control_mode == "position":
                d.ctrl[act_ids] = target_dof_pos
                tau = np.array(d.actuator_force[act_ids], dtype=np.float32, copy=True)
            else:
                tau = pd_control(target_dof_pos, q, kps, np.zeros_like(kds), dq, kds)
                d.ctrl[act_ids] = tau
            mujoco.mj_step(m, d)

            if show_tracking and (counter % max(tracking_every, 1) == 0):
                quat = d.qpos[3:7].astype(np.float32, copy=True)
                rot_wb = quat_wxyz_to_rotmat(quat)
                rot_bw = rot_wb.T
                base_lin_vel_w = d.qvel[:3].astype(np.float32, copy=True)
                base_lin_vel_b = rot_bw @ base_lin_vel_w
                base_ang_vel_b = d.qvel[3:6].astype(np.float32, copy=True)
                err = cmd - np.array(
                    [base_lin_vel_b[0], base_lin_vel_b[1], base_ang_vel_b[2]], dtype=np.float32
                )
                print(
                    (
                        f"\rt={sim_time:6.3f}  "
                        f"cmd[vx vy wz]=[{cmd[0]:5.2f} {cmd[1]:5.2f} {cmd[2]:5.2f}]  "
                        f"act[vx vy wz]=[{base_lin_vel_b[0]:5.2f} {base_lin_vel_b[1]:5.2f} {base_ang_vel_b[2]:5.2f}]  "
                        f"err=[{err[0]:+5.2f} {err[1]:+5.2f} {err[2]:+5.2f}]"
                    ),
                    end="",
                    flush=True,
                )

            counter += 1
            if counter % control_decimation == 0:
                if zero_action or counter < policy_start_step:
                    raw_action = np.zeros(num_actions, dtype=np.float32)
                else:
                    obs = build_observation(
                        config,
                        d,
                        default_angles,
                        last_policy_action,
                        cmd,
                        counter,
                        simulation_dt,
                        qpos_adrs=qpos_adrs,
                        dof_adrs=dof_adrs,
                    )
                    obs_tensor = torch.from_numpy(obs).unsqueeze(0)
                    raw_action = policy(obs_tensor).detach().cpu().numpy().squeeze().astype(np.float32)
                    raw_action = raw_action[action_permutation]
                    raw_action = raw_action * action_signs
                    raw_action = raw_action * action_gains

                last_policy_action = raw_action.copy()

                ramp = 1.0
                if action_ramp_steps > 0 and counter >= policy_start_step:
                    ramp_progress = min(counter - policy_start_step + control_decimation, action_ramp_steps)
                    ramp = np.clip(ramp_progress / action_ramp_steps, 0.0, 1.0)

                desired_action = raw_action * ramp
                action = (1.0 - action_alpha) * action + action_alpha * desired_action
                target_dof_pos = action * action_scale + default_angles

                if debug_trace and ((counter // control_decimation) % max(debug_trace_every, 1) == 0):
                    left_contact, right_contact = foot_contact_flags(m, d)
                    left_action = raw_action[LEFT_IDS]
                    right_action = raw_action[RIGHT_IDS]
                    left_tau = tau[LEFT_IDS]
                    right_tau = tau[RIGHT_IDS]
                    print(
                        "[trace]",
                        f"t={sim_time:.3f}",
                        f"cmd={np.round(cmd, 3).tolist()}",
                        f"Lact={np.round(left_action, 3).tolist()}",
                        f"Ract={np.round(right_action, 3).tolist()}",
                        f"Ltau_norm={float(np.linalg.norm(left_tau)):.2f}",
                        f"Rtau_norm={float(np.linalg.norm(right_tau)):.2f}",
                        f"Lcontact={left_contact}",
                        f"Rcontact={right_contact}",
                    )

            if camera_follow:
                viewer.cam.lookat[:] = np.array(
                    [d.qpos[0], d.qpos[1], d.qpos[2] + camera_lookat_height], dtype=np.float64
                )
            viewer.sync()

            time_until_next_step = m.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)

        if show_tracking:
            print()


if __name__ == "__main__":
    main()
