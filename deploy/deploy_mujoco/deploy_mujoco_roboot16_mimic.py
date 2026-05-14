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


def _format_named_vector(names: Sequence[str], values: Sequence[float], decimals: int = 4) -> str:
    return ", ".join(f"{name}={float(value):.{decimals}f}" for name, value in zip(names, values, strict=True))


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


def _load_external_motion(npz_path: Path) -> dict:
    if not npz_path.is_file():
        raise FileNotFoundError(f"Motion npz not found: {npz_path}")
    data = np.load(npz_path)
    required_keys = [
        "joint_pos",
        "joint_vel",
        "body_pos_w",
        "body_quat_w",
        "body_lin_vel_w",
        "body_ang_vel_w",
    ]
    missing = [key for key in required_keys if key not in data]
    if missing:
        raise KeyError(f"Motion npz is missing required arrays: {missing}")
    motion = {key: np.asarray(data[key], dtype=np.float32) for key in required_keys}
    motion["fps"] = float(np.asarray(data["fps"]).reshape(-1)[0]) if "fps" in data else None
    motion["num_frames"] = int(motion["joint_pos"].shape[0])
    return motion


def _resolve_motion_body_indexes(
    model: mujoco.MjModel,
    motion: dict,
    body_names: Sequence[str],
) -> list[int]:
    motion_body_count = int(motion["body_pos_w"].shape[1])
    model_body_names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, bid) for bid in range(1, model.nbody)]
    name_to_motion_index = {name: idx for idx, name in enumerate(model_body_names) if name is not None}

    if motion_body_count != len(model_body_names):
        raise ValueError(
            "This deploy script currently expects the external motion npz to store all robot bodies "
            f"(excluding world). Found {motion_body_count} bodies, MuJoCo model has {len(model_body_names)}."
        )

    indexes: list[int] = []
    for name in body_names:
        if name not in name_to_motion_index:
            raise KeyError(f"Body '{name}' not found in MuJoCo model body list.")
        indexes.append(name_to_motion_index[name])
    return indexes


def _filter_motion_bodies(motion: dict, body_indexes: Sequence[int]) -> dict:
    filtered = dict(motion)
    for key in ("body_pos_w", "body_quat_w", "body_lin_vel_w", "body_ang_vel_w"):
        filtered[key] = motion[key][:, body_indexes].astype(np.float32, copy=False)
    return filtered


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
    try:
        quat = data.sensor("base_quat").data[[0, 1, 2, 3]].astype(np.float32, copy=True)
    except Exception:
        quat = data.qpos[3:7].astype(np.float32, copy=True)
    rot_wb = quat_wxyz_to_rotmat(quat)
    rot_bw = rot_wb.T
    base_lin_vel_w = data.qvel[:3].astype(np.float32, copy=True)
    base_lin_vel_b = rot_bw @ base_lin_vel_w
    try:
        base_ang_vel_b = data.sensor("base_gyro").data.astype(np.float32, copy=True)
    except Exception:
        base_ang_vel_w = data.qvel[3:6].astype(np.float32, copy=True)
        base_ang_vel_b = rot_bw @ base_ang_vel_w
    return base_lin_vel_b, base_ang_vel_b


def _quat_conjugate(quat_wxyz: np.ndarray) -> np.ndarray:
    return np.array([quat_wxyz[0], -quat_wxyz[1], -quat_wxyz[2], -quat_wxyz[3]], dtype=np.float32)


def _quat_multiply(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ],
        dtype=np.float32,
    )


def _get_body_pose(data: mujoco.MjData, body_id: int) -> tuple[np.ndarray, np.ndarray]:
    return (
        data.xpos[body_id].astype(np.float32, copy=True),
        data.xquat[body_id].astype(np.float32, copy=True),
    )


def _body_name_to_id(model: mujoco.MjModel, body_names: Sequence[str]) -> list[int]:
    body_ids: list[int] = []
    for name in body_names:
        body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
        if body_id < 0:
            raise KeyError(f"Body '{name}' not found in MuJoCo model.")
        body_ids.append(body_id)
    return body_ids


def _quat_distance(q1: np.ndarray, q2: np.ndarray) -> float:
    dot = float(np.clip(np.abs(np.dot(q1, q2)), -1.0, 1.0))
    return float(2.0 * np.arccos(dot))


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
    robot_anchor_body_id: int,
    anchor_body_index: int,
) -> np.ndarray:
    qj = data.qpos[qpos_adrs].astype(np.float32, copy=True)
    dqj = data.qvel[dof_adrs].astype(np.float32, copy=True)
    base_lin_vel_b, base_ang_vel_b = _get_base_obs(data)

    robot_anchor_pos_w, robot_anchor_quat_w = _get_body_pose(data, robot_anchor_body_id)
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


def _reference_frame_from_motion(motion: dict, time_step: int):
    frame_id = int(np.clip(time_step, 0, motion["num_frames"] - 1))
    return (
        motion["joint_pos"][frame_id].astype(np.float32),
        motion["joint_vel"][frame_id].astype(np.float32),
        motion["body_pos_w"][frame_id].astype(np.float32),
        motion["body_quat_w"][frame_id].astype(np.float32),
        motion["body_lin_vel_w"][frame_id].astype(np.float32),
        motion["body_ang_vel_w"][frame_id].astype(np.float32),
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
        kp_ok = np.isclose(row["kp"], expected_kp, atol=1e-4)
        kv_ok = np.isclose(row["kv"], expected_kv, atol=1e-4)
        status = "OK" if kp_ok and kv_ok else "WARN"
        if status != "OK":
            mismatches += 1
        print(
            f"[align:{status}] {row['joint_name']}: "
            f"kp={row['kp']:.3f} expected={expected_kp:.3f}, "
            f"kv={row['kv']:.3f} expected={expected_kv:.3f}, "
            f"|force|={force_limit:.3f}"
        )
    if mismatches == 0:
        print("[align] All joint actuator kp/kv values match exported training parameters closely.")
    else:
        print(f"[align] Found {mismatches} kp/kv mismatches. Review the WARN lines above.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Deploy a mimic/tracking ONNX policy on Roboot16 MuJoCo.")
    parser.add_argument("--onnx", type=Path, required=True, help="Path to exported policy.onnx.")
    parser.add_argument("--motion-npz", type=Path, required=True, help="Path to external mimic motion .npz reference.")
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
    parser.add_argument(
        "--default-joint-pos-override",
        type=str,
        default=None,
        help="Optional comma-separated default joint position override in ONNX joint_names order.",
    )
    args = parser.parse_args()

    spec = _load_policy_spec(args.onnx)
    motion = _load_external_motion(args.motion_npz)
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
    default_joint_pos = spec["default_joint_pos"].copy()
    onnx_default_joint_pos = spec["default_joint_pos"].copy()
    if args.default_joint_pos_override is not None:
        override = np.asarray(_parse_float_list(args.default_joint_pos_override), dtype=np.float32)
        if len(override) != len(default_joint_pos):
            raise ValueError(
                "default_joint_pos override length "
                f"{len(override)} does not match policy joints {len(default_joint_pos)}."
            )
        default_joint_pos = override
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

    motion_body_indexes = _resolve_motion_body_indexes(model, motion, spec["body_names"])
    motion = _filter_motion_bodies(motion, motion_body_indexes)
    tracked_body_ids = _body_name_to_id(model, spec["body_names"])
    anchor_body_name = spec["anchor_body_name"]
    if anchor_body_name not in spec["body_names"]:
        raise KeyError(f"Anchor body '{anchor_body_name}' is not present in metadata body_names: {spec['body_names']}")
    anchor_body_index = spec["body_names"].index(anchor_body_name)
    robot_anchor_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, anchor_body_name)
    if robot_anchor_body_id < 0:
        raise KeyError(f"Anchor body '{anchor_body_name}' not found in MuJoCo model.")

    last_action = np.zeros(len(joint_names), dtype=np.float32)
    raw_action = np.zeros(len(joint_names), dtype=np.float32)
    target_dof_pos = default_joint_pos.copy()
    step_count = 0
    policy_step_count = 0

    initial_time_step = args.time_step_start
    (
        init_ref_joint_pos,
        init_ref_joint_vel,
        init_ref_body_pos_w,
        init_ref_body_quat_w,
        init_ref_body_lin_vel_w,
        init_ref_body_ang_vel_w,
    ) = _reference_frame_from_motion(motion, initial_time_step)

    # Mimic training resets the robot close to the sampled reference frame.
    data.qpos[qpos_adrs] = init_ref_joint_pos
    data.qvel[dof_adrs] = init_ref_joint_vel
    if model.njnt > 0 and model.jnt_type[0] == mujoco.mjtJoint.mjJNT_FREE:
        root_qadr = model.jnt_qposadr[0]
        data.qpos[root_qadr : root_qadr + 3] = init_ref_body_pos_w[anchor_body_index]
        data.qpos[root_qadr + 3 : root_qadr + 7] = init_ref_body_quat_w[anchor_body_index]
        data.qvel[0:3] = init_ref_body_lin_vel_w[anchor_body_index]
        data.qvel[3:6] = init_ref_body_ang_vel_w[anchor_body_index]
    mujoco.mj_forward(model, data)
    init_robot_anchor_pos_w, init_robot_anchor_quat_w = _get_body_pose(data, robot_anchor_body_id)
    init_ref_anchor_pos_w = init_ref_body_pos_w[anchor_body_index].astype(np.float32, copy=True)
    init_ref_anchor_quat_w = init_ref_body_quat_w[anchor_body_index].astype(np.float32, copy=True)
    init_anchor_pos_b, init_anchor_ori_b = _compute_motion_anchor_terms(
        init_robot_anchor_pos_w,
        init_robot_anchor_quat_w,
        init_ref_anchor_pos_w,
        init_ref_anchor_quat_w,
    )
    init_body_pos_errors: list[float] = []
    init_body_rot_errors: list[float] = []
    for body_id, ref_pos, ref_quat in zip(tracked_body_ids, init_ref_body_pos_w, init_ref_body_quat_w, strict=True):
        robot_pos, robot_quat = _get_body_pose(data, body_id)
        init_body_pos_errors.append(float(np.linalg.norm(robot_pos - ref_pos)))
        init_body_rot_errors.append(_quat_distance(robot_quat, ref_quat))

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
            ) = _reference_frame_from_motion(motion, time_step)
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
                robot_anchor_body_id,
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
    print(f"Loaded motion npz: {args.motion_npz}")
    print(f"Loaded MuJoCo model: {args.model}")
    print(f"Export run path: {spec['run_path']}")
    print(f"Joint count: {len(joint_names)}")
    print(f"ONNX default_joint_pos: {_format_named_vector(joint_names, onnx_default_joint_pos)}")
    if args.default_joint_pos_override is not None:
        print(f"Override default_joint_pos: {_format_named_vector(joint_names, default_joint_pos)}")
    print(f"Observation terms: {spec['observation_names']}")
    print(f"Anchor body: {anchor_body_name}")
    print(f"Simulation dt={args.simulation_dt}, control_decimation={args.control_decimation}")
    print(
        "Initial anchor error:",
        f"pos_b={np.round(init_anchor_pos_b, 4).tolist()}",
        f"ori6d={np.round(init_anchor_ori_b, 4).tolist()}",
    )
    print(
        "Initial tracked-body error:",
        f"pos_mean={float(np.mean(init_body_pos_errors)):.4f}",
        f"pos_max={float(np.max(init_body_pos_errors)):.4f}",
        f"rot_mean={float(np.mean(init_body_rot_errors)):.4f}",
        f"rot_max={float(np.max(init_body_rot_errors)):.4f}",
    )
    if motion["fps"] is not None:
        print(f"Motion frames={motion['num_frames']}, motion fps={motion['fps']}")
    else:
        print(f"Motion frames={motion['num_frames']}")

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
