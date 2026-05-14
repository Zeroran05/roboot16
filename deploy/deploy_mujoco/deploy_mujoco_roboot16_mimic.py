from __future__ import annotations

import argparse
import time
from collections.abc import Sequence
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np
import onnx

from deploy_mujoco_roboot16 import build_joint_maps, initialize_pose, quat_wxyz_to_rotmat


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
DEFAULT_XML = PROJECT_ROOT / "assets" / "Roboot1.6" / "xml" / "scene_1.xml"


def _parse_csv_list(value: str) -> list[str]:
    if value == "":
        return []
    return [item.strip() for item in value.split(",")]


def _parse_float_list(value: str) -> list[float]:
    return [float(item) for item in _parse_csv_list(value)]


def _metadata_dict(onnx_path: Path) -> dict[str, str]:
    model = onnx.load(str(onnx_path))
    return {entry.key: entry.value for entry in model.metadata_props}


def _load_policy_spec(onnx_path: Path) -> dict:
    metadata = _metadata_dict(onnx_path)

    required_keys = [
        "joint_names",
        "joint_stiffness",
        "joint_damping",
        "default_joint_pos",
        "observation_names",
        "observation_history_lengths",
        "action_scale",
        "anchor_body_name",
        "body_names",
        "command_names",
    ]
    missing = [key for key in required_keys if key not in metadata]
    if missing:
        raise KeyError(
            f"ONNX metadata is missing required keys: {missing}. "
            "Please export the model with the mimic/tracking exporter."
        )

    spec = {
        "joint_names": _parse_csv_list(metadata["joint_names"]),
        "joint_stiffness": np.asarray(_parse_float_list(metadata["joint_stiffness"]), dtype=np.float32),
        "joint_damping": np.asarray(_parse_float_list(metadata["joint_damping"]), dtype=np.float32),
        "default_joint_pos": np.asarray(_parse_float_list(metadata["default_joint_pos"]), dtype=np.float32),
        "observation_names": _parse_csv_list(metadata["observation_names"]),
        "observation_history_lengths": [int(float(x)) for x in _parse_csv_list(metadata["observation_history_lengths"])],
        "action_scale": np.asarray(_parse_float_list(metadata["action_scale"]), dtype=np.float32),
        "anchor_body_name": metadata["anchor_body_name"],
        "body_names": _parse_csv_list(metadata["body_names"]),
        "command_names": _parse_csv_list(metadata["command_names"]),
        "run_path": metadata.get("run_path", "unknown"),
    }
    return spec


def _make_ort_session(onnx_path: Path):
    try:
        import onnxruntime as ort
    except ImportError as exc:
        raise ImportError(
            "onnxruntime is required to run the mimic deploy script. "
            "Install it in the current environment first."
        ) from exc

    providers = ["CPUExecutionProvider"]
    return ort.InferenceSession(str(onnx_path), providers=providers)


def _assert_supported_observation_history(history_lengths: Sequence[int]) -> None:
    unsupported = [length for length in history_lengths if length != 1]
    if unsupported:
        raise NotImplementedError(
            "This deploy script currently supports only history_length == 1 for every observation term. "
            f"Found: {list(history_lengths)}"
        )


def _get_base_obs(data: mujoco.MjData) -> tuple[np.ndarray, np.ndarray]:
    quat = data.qpos[3:7].astype(np.float32, copy=True)
    rot_wb = quat_wxyz_to_rotmat(quat)
    rot_bw = rot_wb.T
    base_lin_vel_w = data.qvel[:3].astype(np.float32, copy=True)
    base_lin_vel_b = rot_bw @ base_lin_vel_w
    base_ang_vel_b = data.qvel[3:6].astype(np.float32, copy=True)
    return base_lin_vel_b, base_ang_vel_b


def _compute_motion_anchor_terms(
    robot_anchor_pos_w: np.ndarray,
    robot_anchor_quat_w: np.ndarray,
    ref_anchor_pos_w: np.ndarray,
    ref_anchor_quat_w: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    rot_robot_w = quat_wxyz_to_rotmat(robot_anchor_quat_w)
    rot_ref_w = quat_wxyz_to_rotmat(ref_anchor_quat_w)
    delta_pos_w = ref_anchor_pos_w - robot_anchor_pos_w
    motion_anchor_pos_b = rot_robot_w.T @ delta_pos_w
    relative_rot = rot_robot_w.T @ rot_ref_w
    motion_anchor_ori_b = relative_rot[:, :2].reshape(-1).astype(np.float32)
    return motion_anchor_pos_b.astype(np.float32), motion_anchor_ori_b


def _build_policy_observation(
    observation_names: Sequence[str],
    data: mujoco.MjData,
    qpos_adrs: Sequence[int],
    dof_adrs: Sequence[int],
    default_joint_pos: np.ndarray,
    last_action: np.ndarray,
    ref_joint_pos: np.ndarray,
    ref_joint_vel: np.ndarray,
    ref_body_pos_w: np.ndarray,
    ref_body_quat_w: np.ndarray,
    anchor_body_index: int,
) -> np.ndarray:
    qj = data.qpos[qpos_adrs].astype(np.float32, copy=True)
    dqj = data.qvel[dof_adrs].astype(np.float32, copy=True)
    base_lin_vel_b, base_ang_vel_b = _get_base_obs(data)

    robot_anchor_pos_w = data.qpos[:3].astype(np.float32, copy=True)
    robot_anchor_quat_w = data.qpos[3:7].astype(np.float32, copy=True)
    ref_anchor_pos_w = ref_body_pos_w[anchor_body_index].astype(np.float32, copy=True)
    ref_anchor_quat_w = ref_body_quat_w[anchor_body_index].astype(np.float32, copy=True)
    motion_anchor_pos_b, motion_anchor_ori_b = _compute_motion_anchor_terms(
        robot_anchor_pos_w,
        robot_anchor_quat_w,
        ref_anchor_pos_w,
        ref_anchor_quat_w,
    )

    term_values = {
        "command": np.concatenate([ref_joint_pos, ref_joint_vel], axis=0).astype(np.float32),
        "motion_anchor_pos_b": motion_anchor_pos_b,
        "motion_anchor_ori_b": motion_anchor_ori_b,
        "base_lin_vel": base_lin_vel_b.astype(np.float32),
        "base_ang_vel": base_ang_vel_b.astype(np.float32),
        "joint_pos": (qj - default_joint_pos).astype(np.float32),
        "joint_vel": dqj.astype(np.float32),
        "actions": last_action.astype(np.float32),
    }

    obs_terms: list[np.ndarray] = []
    for name in observation_names:
        if name not in term_values:
            raise KeyError(
                f"Unsupported observation term '{name}' in ONNX metadata. "
                f"Supported terms are: {sorted(term_values)}"
            )
        obs_terms.append(term_values[name].reshape(-1))
    return np.concatenate(obs_terms, axis=0).astype(np.float32)


def _run_reference_pass(session, obs_dim: int, time_step: int):
    dummy_obs = np.zeros((1, obs_dim), dtype=np.float32)
    time_step_input = np.array([[time_step]], dtype=np.float32)
    outputs = session.run(
        ["actions", "joint_pos", "joint_vel", "body_pos_w", "body_quat_w", "body_lin_vel_w", "body_ang_vel_w"],
        {"obs": dummy_obs, "time_step": time_step_input},
    )
    _, joint_pos, joint_vel, body_pos_w, body_quat_w, body_lin_vel_w, body_ang_vel_w = outputs
    return (
        joint_pos[0].astype(np.float32),
        joint_vel[0].astype(np.float32),
        body_pos_w[0].astype(np.float32),
        body_quat_w[0].astype(np.float32),
        body_lin_vel_w[0].astype(np.float32),
        body_ang_vel_w[0].astype(np.float32),
    )


def _run_policy_pass(session, obs: np.ndarray, time_step: int) -> np.ndarray:
    outputs = session.run(
        ["actions"],
        {"obs": obs[None, :].astype(np.float32), "time_step": np.array([[time_step]], dtype=np.float32)},
    )
    return outputs[0][0].astype(np.float32)


def _update_camera_follow(viewer, data: mujoco.MjData, lookat_height: float) -> None:
    viewer.cam.lookat[:] = np.array(
        [data.qpos[0], data.qpos[1], data.qpos[2] + lookat_height],
        dtype=np.float64,
    )


def _joint_name_to_ids(model: mujoco.MjModel) -> dict[str, tuple[int, int, int]]:
    """Map joint name -> (joint_id, qpos_adr, dof_adr)."""
    mapping: dict[str, tuple[int, int, int]] = {}
    for jid in range(model.njnt):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, jid)
        if name is None:
            continue
        mapping[name] = (jid, model.jnt_qposadr[jid], model.jnt_dofadr[jid])
    return mapping


def _actuator_summary(model: mujoco.MjModel, joint_names: Sequence[str]) -> list[dict]:
    joint_map = _joint_name_to_ids(model)
    joint_to_actuator = {}
    for aid in range(model.nu):
        if model.actuator_trntype[aid] != mujoco.mjtTrn.mjTRN_JOINT:
            continue
        joint_to_actuator[model.actuator_trnid[aid, 0]] = aid

    summary = []
    for joint_name in joint_names:
        if joint_name not in joint_map:
            raise KeyError(f"Joint '{joint_name}' not found in MuJoCo model.")
        jid, _, _ = joint_map[joint_name]
        if jid not in joint_to_actuator:
            raise KeyError(f"Joint '{joint_name}' has no actuator in MuJoCo model.")
        aid = joint_to_actuator[jid]
        summary.append(
            {
                "joint_name": joint_name,
                "actuator_id": aid,
                "kp": float(model.actuator_gainprm[aid, 0]),
                "kv": float(model.actuator_biasprm[aid, 2] * -1.0),
                "forcerange": (
                    float(model.actuator_forcerange[aid, 0]),
                    float(model.actuator_forcerange[aid, 1]),
                ),
                "ctrlrange": (
                    float(model.actuator_ctrlrange[aid, 0]),
                    float(model.actuator_ctrlrange[aid, 1]),
                ),
            }
        )
    return summary


def _print_alignment_report(model: mujoco.MjModel, spec: dict) -> None:
    actuator_rows = _actuator_summary(model, spec["joint_names"])
    print("[align] MuJoCo actuator alignment report")
    mismatches = 0
    for i, row in enumerate(actuator_rows):
        expected_kp = float(spec["joint_stiffness"][i])
        expected_kv = float(spec["joint_damping"][i])
        force_limit = max(abs(row["forcerange"][0]), abs(row["forcerange"][1]))
        expected_force = expected_kp * float(spec["action_scale"][i]) * 4.0
        kp_ok = np.isclose(row["kp"], expected_kp, atol=1e-4)
        kv_ok = np.isclose(row["kv"], expected_kv, atol=1e-4)
        force_ok = np.isclose(force_limit, expected_force, atol=1e-3)
        status = "OK" if kp_ok and kv_ok and force_ok else "WARN"
        if status != "OK":
            mismatches += 1
        print(
            f"[align:{status}] {row['joint_name']}: "
            f"kp={row['kp']:.3f} expected={expected_kp:.3f}, "
            f"kv={row['kv']:.3f} expected={expected_kv:.3f}, "
            f"|force|={force_limit:.3f} expected~={expected_force:.3f}"
        )
    if mismatches == 0:
        print("[align] All joint actuators match exported training parameters closely.")
    else:
        print(f"[align] Found {mismatches} actuator mismatches. Review the WARN lines above.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Deploy a mimic/tracking ONNX policy on Roboot16 MuJoCo.")
    parser.add_argument("--onnx", type=Path, required=True, help="Path to exported policy.onnx.")
    parser.add_argument("--model", type=Path, default=DEFAULT_XML, help="MuJoCo XML model path.")
    parser.add_argument("--simulation-dt", type=float, default=0.005, help="MuJoCo simulation time-step.")
    parser.add_argument("--control-decimation", type=int, default=4, help="Run the policy every N sim steps.")
    parser.add_argument("--seconds", type=float, default=None, help="Optional wall-clock run duration.")
    parser.add_argument("--base-height", type=float, default=None, help="Initial floating-base height.")
    parser.add_argument("--init-auto-ground", action="store_true", help="Auto-place the robot slightly above the ground.")
    parser.add_argument("--init-clearance", type=float, default=0.01, help="Clearance used with auto-ground placement.")
    parser.add_argument("--camera-follow", action="store_true", help="Keep the viewer camera centered on the robot.")
    parser.add_argument("--camera-distance", type=float, default=3.0, help="Viewer camera distance when following.")
    parser.add_argument("--camera-azimuth", type=float, default=135.0, help="Viewer camera azimuth when following.")
    parser.add_argument("--camera-elevation", type=float, default=-10.0, help="Viewer camera elevation when following.")
    parser.add_argument("--camera-lookat-height", type=float, default=0.7, help="Look-at height offset for follow camera.")
    parser.add_argument("--time-step-start", type=int, default=0, help="Initial reference motion frame index.")
    parser.add_argument("--motion-frames", type=int, default=None, help="Optional total number of motion frames for looping.")
    parser.add_argument("--loop-motion", action="store_true", help="Loop the reference motion using --motion-frames.")
    parser.add_argument("--no-viewer", action="store_true", help="Run headless.")
    parser.add_argument("--print-every", type=int, default=50, help="Print a short debug line every N policy steps.")
    args = parser.parse_args()

    spec = _load_policy_spec(args.onnx)
    _assert_supported_observation_history(spec["observation_history_lengths"])
    if "motion" not in spec["command_names"]:
        raise ValueError(
            f"The exported ONNX does not advertise a 'motion' command. Found: {spec['command_names']}"
        )

    session = _make_ort_session(args.onnx)
    input_names = {item.name: item for item in session.get_inputs()}
    if "obs" not in input_names or "time_step" not in input_names:
        raise ValueError("Expected ONNX inputs named 'obs' and 'time_step'.")
    obs_dim = int(input_names["obs"].shape[1])

    if len(spec["observation_names"]) == 0:
        raise ValueError("The ONNX metadata did not contain any observation names.")

    model = mujoco.MjModel.from_xml_path(str(args.model))
    data = mujoco.MjData(model)
    model.opt.timestep = args.simulation_dt

    joint_names = spec["joint_names"]
    qpos_adrs, dof_adrs, act_ids = build_joint_maps(model, joint_names)
    default_joint_pos = spec["default_joint_pos"]
    if len(default_joint_pos) != len(qpos_adrs):
        raise ValueError(
            f"default_joint_pos length {len(default_joint_pos)} does not match policy joints {len(qpos_adrs)}."
        )
    if len(spec["action_scale"]) != len(qpos_adrs):
        raise ValueError(f"action_scale length {len(spec['action_scale'])} does not match policy joints {len(qpos_adrs)}.")
    _print_alignment_report(model, spec)

    if args.base_height is None and args.init_auto_ground:
        initialize_pose(
            model,
            data,
            qpos_adrs,
            default_joint_pos,
            auto_ground=True,
            clearance=args.init_clearance,
        )
    else:
        initialize_pose(model, data, qpos_adrs, default_joint_pos, auto_ground=False)
        if args.base_height is not None and model.njnt > 0 and model.jnt_type[0] == mujoco.mjtJoint.mjJNT_FREE:
            root_qadr = model.jnt_qposadr[0]
            data.qpos[root_qadr + 2] = args.base_height
            mujoco.mj_forward(model, data)

    anchor_body_name = spec["anchor_body_name"]
    if anchor_body_name not in spec["body_names"]:
        raise KeyError(
            f"Anchor body '{anchor_body_name}' is not present in exported body_names={spec['body_names']}"
        )
    anchor_body_index = spec["body_names"].index(anchor_body_name)

    last_action = np.zeros(len(joint_names), dtype=np.float32)
    raw_action = np.zeros(len(joint_names), dtype=np.float32)
    target_dof_pos = default_joint_pos.copy()
    step_count = 0
    policy_step_count = 0

    def resolve_time_step(policy_index: int) -> int:
        time_step = args.time_step_start + policy_index
        if args.loop_motion and args.motion_frames is not None and args.motion_frames > 0:
            time_step = args.time_step_start + (policy_index % args.motion_frames)
        return time_step

    def one_step() -> None:
        nonlocal step_count, policy_step_count, last_action, raw_action, target_dof_pos

        if step_count % args.control_decimation == 0:
            time_step = resolve_time_step(policy_step_count)
            (
                ref_joint_pos,
                ref_joint_vel,
                ref_body_pos_w,
                ref_body_quat_w,
                _ref_body_lin_vel_w,
                _ref_body_ang_vel_w,
            ) = _run_reference_pass(session, obs_dim, time_step)
            obs = _build_policy_observation(
                spec["observation_names"],
                data,
                qpos_adrs,
                dof_adrs,
                default_joint_pos,
                last_action,
                ref_joint_pos,
                ref_joint_vel,
                ref_body_pos_w,
                ref_body_quat_w,
                anchor_body_index,
            )
            if obs.shape[0] != obs_dim:
                raise ValueError(f"Built observation dim {obs.shape[0]} does not match ONNX input dim {obs_dim}.")
            raw_action = _run_policy_pass(session, obs, time_step)
            last_action = raw_action.copy()
            target_dof_pos = default_joint_pos + spec["action_scale"] * raw_action

            if args.print_every > 0 and policy_step_count % args.print_every == 0:
                base_lin_vel_b, base_ang_vel_b = _get_base_obs(data)
                print(
                    "[mimic]",
                    f"policy_step={policy_step_count}",
                    f"time_step={time_step}",
                    f"base_lin_vel={np.round(base_lin_vel_b, 3).tolist()}",
                    f"base_ang_vel={np.round(base_ang_vel_b, 3).tolist()}",
                    f"action_norm={float(np.linalg.norm(raw_action)):.3f}",
                )

            policy_step_count += 1

        q = data.qpos[qpos_adrs]
        data.ctrl[:] = 0.0
        data.ctrl[act_ids] = target_dof_pos

        mujoco.mj_step(model, data)
        step_count += 1

    print(f"Loaded ONNX: {args.onnx}")
    print(f"Loaded MuJoCo model: {args.model}")
    print(f"Export run path: {spec['run_path']}")
    print(f"Joint count: {len(joint_names)}")
    print(f"Observation terms: {spec['observation_names']}")
    print(f"Anchor body: {anchor_body_name}")
    print(f"Simulation dt={args.simulation_dt}, control_decimation={args.control_decimation}")

    if args.no_viewer:
        if args.seconds is None:
            raise ValueError("--seconds is required in --no-viewer mode.")
        num_steps = int(round(args.seconds / args.simulation_dt))
        for _ in range(num_steps):
            one_step()
        print(f"Finished headless mimic rollout. base_pos={data.qpos[0:3].tolist()}")
        return

    with mujoco.viewer.launch_passive(model, data) as viewer:
        if args.camera_follow:
            viewer.cam.distance = args.camera_distance
            viewer.cam.azimuth = args.camera_azimuth
            viewer.cam.elevation = args.camera_elevation
            _update_camera_follow(viewer, data, args.camera_lookat_height)

        start_t = time.time()
        while viewer.is_running() and (args.seconds is None or time.time() - start_t < args.seconds):
            step_start = time.time()
            one_step()
            if args.camera_follow:
                _update_camera_follow(viewer, data, args.camera_lookat_height)
            viewer.sync()

            time_until_next_step = model.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)


if __name__ == "__main__":
    main()
