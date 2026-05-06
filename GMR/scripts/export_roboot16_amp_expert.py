import argparse
import json
import pathlib
import pickle
import sys
import time

import mujoco as mj
import mujoco.viewer as mjv
import numpy as np
import numpy.core as numpy_core
import numpy.core.multiarray as numpy_core_multiarray

sys.modules.setdefault("numpy._core", numpy_core)
sys.modules.setdefault("numpy._core.multiarray", numpy_core_multiarray)

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


# 关节位置和速度都是绝对值；脚位置是在root下的局部坐标；


# Ordering contract used by this script:
# 1. MuJoCo defines the roboot16 joint order through qpos/qvel indexing after the free joint.
# 2. The default ROBOOT16_DOF_NAMES below has been verified against:
#    - model.jnt_qposadr -> qpos[7:]
#    - model.dof_jntid   -> qvel[6:] / dof order
#    - model.actuator_trnid -> actuator-to-joint order
# 3. GMR motion PKLs are assumed to store dof_pos/dof_vel in that same MuJoCo order.
# 4. This exporter reorders frames into AMP expert layout by selecting joints from that verified order,
#    and computes root-local foot positions from MuJoCo forward kinematics at each frame.
ROBOOT16_DOF_NAMES = [
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
]

ROBOOT16_RIGHT_LEG = ROBOOT16_DOF_NAMES[:6]
ROBOOT16_LEFT_LEG = ROBOOT16_DOF_NAMES[6:]
ROBOOT16_FOOT_LINKS = ["left_ankle_roll_link", "right_ankle_roll_link"]
DEFAULT_ROBOOT16_XML = REPO_ROOT / "Roboot1.6" / "xml" / "roboot16_1.xml"


def quat_conjugate_wxyz(quat_wxyz):
    quat_wxyz = np.asarray(quat_wxyz, dtype=np.float32)
    return np.array(
        [quat_wxyz[0], -quat_wxyz[1], -quat_wxyz[2], -quat_wxyz[3]],
        dtype=np.float32,
    )


def quat_multiply_wxyz(q1, q2):
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


def rotate_vector_by_quat_wxyz(quat_wxyz, vec3):
    pure_quat = np.array([0.0, vec3[0], vec3[1], vec3[2]], dtype=np.float32)
    quat_inv = quat_conjugate_wxyz(quat_wxyz)
    rotated = quat_multiply_wxyz(quat_multiply_wxyz(quat_wxyz, pure_quat), quat_inv)
    return rotated[1:]


def world_to_root_local_positions(world_positions, root_pos, root_rot_wxyz):
    root_relative = np.asarray(world_positions, dtype=np.float32) - np.asarray(root_pos, dtype=np.float32)
    root_inv = quat_conjugate_wxyz(root_rot_wxyz)
    local_positions = np.zeros_like(root_relative, dtype=np.float32)
    for idx, vec in enumerate(root_relative):
        local_positions[idx] = rotate_vector_by_quat_wxyz(root_inv, vec)
    return local_positions


def load_motion_data(input_path):
    with open(input_path, "rb") as f:
        motion_data = pickle.load(f)
    return motion_data


def resolve_robot_xml_path(robot_name, robot_xml_path_text):
    if robot_xml_path_text:
        return pathlib.Path(robot_xml_path_text).resolve()

    if robot_name == "roboot16":
        return DEFAULT_ROBOOT16_XML.resolve()

    raise ValueError(f"Unsupported robot xml resolution for robot={robot_name!r}")


def resolve_dof_names(motion_data):
    dof_names = motion_data.get("dof_names")
    if dof_names is not None:
        return list(dof_names)
    return list(ROBOOT16_DOF_NAMES)


def build_joint_index_map(dof_names):
    return {name: idx for idx, name in enumerate(dof_names)}


def gather_leg_indices(dof_names):
    index_map = build_joint_index_map(dof_names)
    missing = [name for name in ROBOOT16_DOF_NAMES if name not in index_map]
    if missing:
        raise KeyError(f"Missing joints in motion data: {missing}")
    right_leg_ids = [index_map[name] for name in ROBOOT16_RIGHT_LEG]
    left_leg_ids = [index_map[name] for name in ROBOOT16_LEFT_LEG]
    return right_leg_ids, left_leg_ids


def ensure_dof_vel(motion_data, fps):
    dof_vel = motion_data.get("dof_vel")
    if dof_vel is not None:
        return np.asarray(dof_vel, dtype=np.float32)

    dof_pos = np.asarray(motion_data["dof_pos"], dtype=np.float32)
    dt = 1.0 / float(fps)
    vel = np.zeros_like(dof_pos, dtype=np.float32)
    if len(dof_pos) > 1:
        vel[:-1] = (dof_pos[1:] - dof_pos[:-1]) / dt
        vel[-1] = vel[-2]
    return vel


def extract_root_rot_wxyz(root_rot_xyzw):
    root_rot_xyzw = np.asarray(root_rot_xyzw, dtype=np.float32)
    return root_rot_xyzw[[3, 0, 1, 2]]


def compute_root_local_foot_positions(model, data, root_pos, root_rot_wxyz, dof_pos, foot_links):
    data.qpos[:3] = root_pos
    data.qpos[3:7] = root_rot_wxyz
    data.qpos[7:] = dof_pos
    mj.mj_forward(model, data)

    foot_world = []
    for link_name in foot_links:
        body_id = model.body(link_name).id
        foot_world.append(data.xpos[body_id].copy())
    foot_world = np.asarray(foot_world, dtype=np.float32)
    foot_local = world_to_root_local_positions(foot_world, root_pos, root_rot_wxyz)
    return foot_local, foot_world


def playback_motion(model, data, motion_data, playback_fps):
    root_pos = np.asarray(motion_data["root_pos"], dtype=np.float32)
    root_rot = np.asarray(motion_data["root_rot"], dtype=np.float32)
    dof_pos = np.asarray(motion_data["dof_pos"], dtype=np.float32)
    dt = 1.0 / float(playback_fps if playback_fps > 0 else motion_data["fps"])

    with mjv.launch_passive(
        model=model,
        data=data,
        show_left_ui=False,
        show_right_ui=False,
    ) as viewer:
        for frame_idx in range(len(dof_pos)):
            data.qpos[:3] = root_pos[frame_idx]
            data.qpos[3:7] = extract_root_rot_wxyz(root_rot[frame_idx])
            data.qpos[7:] = dof_pos[frame_idx]
            mj.mj_forward(model, data)
            viewer.sync()
            time.sleep(dt)


def build_amp_expert_frames(motion_data, right_leg_ids, left_leg_ids, model, data, foot_links):
    root_pos = np.asarray(motion_data["root_pos"], dtype=np.float32)
    root_rot = np.asarray(motion_data["root_rot"], dtype=np.float32)
    dof_pos = np.asarray(motion_data["dof_pos"], dtype=np.float32)
    dof_vel = ensure_dof_vel(motion_data, motion_data["fps"])

    frames = []
    foot_pos_local_all = []
    foot_pos_world_all = []

    for frame_idx in range(len(dof_pos)):
        root_rot_wxyz = extract_root_rot_wxyz(root_rot[frame_idx])
        foot_local, foot_world = compute_root_local_foot_positions(
            model,
            data,
            root_pos[frame_idx],
            root_rot_wxyz,
            dof_pos[frame_idx],
            foot_links,
        )
        right_leg_q = dof_pos[frame_idx, right_leg_ids]
        left_leg_q = dof_pos[frame_idx, left_leg_ids]
        right_leg_dq = dof_vel[frame_idx, right_leg_ids]
        left_leg_dq = dof_vel[frame_idx, left_leg_ids]

        frame = np.concatenate(
            [
                right_leg_q,
                left_leg_q,
                right_leg_dq,
                left_leg_dq,
                foot_local[0],
                foot_local[1],
            ],
            axis=0,
        ).astype(np.float32)
        frames.append(frame)
        foot_pos_local_all.append(foot_local)
        foot_pos_world_all.append(foot_world)

    return (
        np.asarray(frames, dtype=np.float32),
        np.asarray(foot_pos_local_all, dtype=np.float32),
        np.asarray(foot_pos_world_all, dtype=np.float32),
    )


def save_amp_expert_txt(output_path, frames, fps):
    output_path = pathlib.Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame_duration = 1.0 / float(fps)

    frames_list = [[float(x) for x in frame] for frame in frames]
    payload = {
        "LoopMode": "Wrap",
        "FrameDuration": round(frame_duration, 6),
        "EnableCycleOffsetPosition": True,
        "EnableCycleOffsetRotation": True,
        "MotionWeight": 0.5,
        "CoordinateFrame": "root_local_for_end_effectors",
        "FrameLayout": [
            "right_leg_q",
            "left_leg_q",
            "right_leg_dq",
            "left_leg_dq",
            "left_foot_pos_local",
            "right_foot_pos_local",
        ],
        "FrameLayoutDims": [6, 6, 6, 6, 3, 3],
        "JointOrder": {
            "right_leg_q": ROBOOT16_RIGHT_LEG,
            "left_leg_q": ROBOOT16_LEFT_LEG,
            "right_leg_dq": ROBOOT16_RIGHT_LEG,
            "left_leg_dq": ROBOOT16_LEFT_LEG,
        },
        "EndEffectorOrder": [
            "left_ankle_roll_link",
            "right_ankle_roll_link",
        ],
        "Frames": frames_list,
    }
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=True, indent=2)


def save_debug_pkl(output_path, motion_data, foot_links, foot_local, foot_world):
    debug_path = pathlib.Path(output_path)
    debug_payload = dict(motion_data)
    debug_payload["foot_link_body_list"] = list(foot_links)
    debug_payload["foot_link_body_pos"] = foot_local
    debug_payload["foot_link_body_pos_world"] = foot_world
    debug_payload["local_body_pos"] = foot_local
    debug_payload["link_body_list"] = list(foot_links)
    meta = dict(debug_payload.get("meta", {}))
    meta["foot_link_body_pos_space"] = "root_local"
    meta["foot_link_body_pos_world_space"] = "world"
    meta["amp_expert_frame_layout"] = (
        "right_leg_q(6), left_leg_q(6), right_leg_dq(6), left_leg_dq(6), "
        "left_foot_pos_local(3), right_foot_pos_local(3)"
    )
    debug_payload["meta"] = meta
    with debug_path.open("wb") as f:
        pickle.dump(debug_payload, f)


def main():
    parser = argparse.ArgumentParser(
        description="Replay a roboot16 GMR PKL in MuJoCo and export TianKung-style AMP expert frames."
    )
    parser.add_argument("--input_pkl", required=True, help="Input robot motion pkl")
    parser.add_argument("--output_txt", required=True, help="Output AMP expert txt/json path")
    parser.add_argument(
        "--output_debug_pkl",
        default="",
        help="Optional output pkl path with root-local/world foot positions added",
    )
    parser.add_argument("--robot", default="roboot16", choices=["roboot16"])
    parser.add_argument(
        "--robot_xml",
        default="",
        help="Optional path to the MuJoCo XML used for FK/export. Defaults to Roboot1.6/xml/roboot16_1.xml",
    )
    parser.add_argument(
        "--foot_links",
        default=",".join(ROBOOT16_FOOT_LINKS),
        help="Comma-separated foot body names in left,right order",
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Open a MuJoCo viewer and replay the motion while exporting.",
    )
    parser.add_argument(
        "--playback_fps",
        type=float,
        default=0.0,
        help="Playback FPS for visualization. Defaults to the motion fps.",
    )
    args = parser.parse_args()

    motion_data = load_motion_data(args.input_pkl)
    dof_names = resolve_dof_names(motion_data)
    right_leg_ids, left_leg_ids = gather_leg_indices(dof_names)
    foot_links = [item.strip() for item in args.foot_links.split(",") if item.strip()]
    if len(foot_links) != 2:
        raise ValueError("Expected exactly two foot links in left,right order")

    robot_xml_path = resolve_robot_xml_path(args.robot, args.robot_xml)
    model = mj.MjModel.from_xml_path(str(robot_xml_path))
    data = mj.MjData(model)

    if args.visualize:
        playback_motion(model, data, motion_data, args.playback_fps)

    frames, foot_local, foot_world = build_amp_expert_frames(
        motion_data,
        right_leg_ids,
        left_leg_ids,
        model,
        data,
        foot_links,
    )

    save_amp_expert_txt(args.output_txt, frames, motion_data["fps"])
    print(f"Saved AMP expert txt to {args.output_txt}")
    print(f"Robot XML: {robot_xml_path}")
    print(f"Frames: {frames.shape[0]}")
    print(f"Frame dim: {frames.shape[1]}")
    print("Frame layout: right_leg_q(6), left_leg_q(6), right_leg_dq(6), left_leg_dq(6), left_foot_pos_local(3), right_foot_pos_local(3)")

    if args.output_debug_pkl:
        save_debug_pkl(args.output_debug_pkl, motion_data, foot_links, foot_local, foot_world)
        print(f"Saved debug pkl to {args.output_debug_pkl}")


if __name__ == "__main__":
    main()
