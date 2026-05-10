import argparse
import json
from pathlib import Path

import mujoco as mj
import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[2]
DEFAULT_MODEL = REPO_ROOT / "assets" / "Roboot1.6" / "xml" / "scene_1.xml"
DEFAULT_OUTPUT = REPO_ROOT / "GMR" / "retargeting_data" / "roboot16" / "amp_expert_txt" / "stand" / "stand_current_2s_60f.txt"

ROBOOT16_RIGHT_LEG = [
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
]
ROBOOT16_LEFT_LEG = [
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
]
ROBOOT16_FOOT_LINKS = ["left_ankle_roll_link", "right_ankle_roll_link"]
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


def quat_conjugate_wxyz(quat_wxyz):
    quat_wxyz = np.asarray(quat_wxyz, dtype=np.float32)
    return np.array([quat_wxyz[0], -quat_wxyz[1], -quat_wxyz[2], -quat_wxyz[3]], dtype=np.float32)


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


def hinge_joint_ids(model):
    return [jid for jid in range(model.njnt) if model.jnt_type[jid] == mj.mjtJoint.mjJNT_HINGE]


def auto_ground_base_height(model, data, clearance=0.01):
    data.qpos[:] = 0.0
    data.qvel[:] = 0.0
    data.qpos[:7] = np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    mj.mj_forward(model, data)

    lowest_z = None
    for gid in range(model.ngeom):
        name = mj.mj_id2name(model, mj.mjtObj.mjOBJ_GEOM, gid) or ""
        if name in {"floor", "ground"}:
            continue
        if model.geom_contype[gid] == 0 and model.geom_conaffinity[gid] == 0:
            continue
        z = float(data.geom_xpos[gid, 2] - model.geom_rbound[gid])
        lowest_z = z if lowest_z is None else min(lowest_z, z)

    if lowest_z is None:
        return clearance
    return clearance - lowest_z


def set_standing_pose(model, data, clearance=0.01):
    data.qpos[:] = 0.0
    data.qvel[:] = 0.0

    for jid in hinge_joint_ids(model):
        name = mj.mj_id2name(model, mj.mjtObj.mjOBJ_JOINT, jid)
        if name in DEFAULT_JOINT_POS:
            data.qpos[model.jnt_qposadr[jid]] = DEFAULT_JOINT_POS[name]

    if model.njnt > 0 and model.jnt_type[0] == mj.mjtJoint.mjJNT_FREE:
        base_height = auto_ground_base_height(model, data, clearance=clearance)
        data.qpos[:7] = np.array([0.0, 0.0, base_height, 1.0, 0.0, 0.0, 0.0], dtype=np.float64)

    mj.mj_forward(model, data)


def joint_index_map(model):
    mapping = {}
    for jid in hinge_joint_ids(model):
        name = mj.mj_id2name(model, mj.mjtObj.mjOBJ_JOINT, jid)
        mapping[name] = (model.jnt_qposadr[jid], model.jnt_dofadr[jid])
    return mapping


def compute_foot_local_positions(model, data):
    root_pos = np.asarray(data.qpos[:3], dtype=np.float32)
    root_rot_wxyz = np.asarray(data.qpos[3:7], dtype=np.float32)
    foot_world = []
    for link_name in ROBOOT16_FOOT_LINKS:
        body_id = model.body(link_name).id
        foot_world.append(data.xpos[body_id].copy())
    foot_world = np.asarray(foot_world, dtype=np.float32)
    return world_to_root_local_positions(foot_world, root_pos, root_rot_wxyz)


def build_frame(model, data, index_map):
    right_leg_q = np.array([data.qpos[index_map[name][0]] for name in ROBOOT16_RIGHT_LEG], dtype=np.float32)
    left_leg_q = np.array([data.qpos[index_map[name][0]] for name in ROBOOT16_LEFT_LEG], dtype=np.float32)
    right_leg_dq = np.array([data.qvel[index_map[name][1]] for name in ROBOOT16_RIGHT_LEG], dtype=np.float32)
    left_leg_dq = np.array([data.qvel[index_map[name][1]] for name in ROBOOT16_LEFT_LEG], dtype=np.float32)
    foot_local = compute_foot_local_positions(model, data)
    return np.concatenate(
        [right_leg_q, left_leg_q, right_leg_dq, left_leg_dq, foot_local[0], foot_local[1]],
        axis=0,
    ).astype(np.float32)


def save_amp_expert_txt(output_path, frames, fps):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "LoopMode": "Wrap",
        "FrameDuration": round(1.0 / float(fps), 6),
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
        "EndEffectorOrder": ROBOOT16_FOOT_LINKS,
        "Frames": [[float(x) for x in frame] for frame in frames],
    }
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=True, indent=2)


def main():
    parser = argparse.ArgumentParser(
        description="Record roboot16 current MuJoCo state in AMP-expert txt format."
    )
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL, help="MuJoCo XML model path.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Output txt/json path.")
    parser.add_argument("--fps", type=int, default=30, help="Output FPS.")
    parser.add_argument("--seconds", type=float, default=2.0, help="Recorded duration in seconds.")
    parser.add_argument(
        "--step",
        action="store_true",
        help="Advance MuJoCo while recording. By default, repeat the current state for all frames.",
    )
    parser.add_argument(
        "--set-standing-pose",
        action="store_true",
        help="Initialize the robot to the default standing pose before recording.",
    )
    args = parser.parse_args()

    model = mj.MjModel.from_xml_path(str(args.model))
    data = mj.MjData(model)

    if args.set_standing_pose:
        set_standing_pose(model, data)
    else:
        mj.mj_forward(model, data)

    fps = int(args.fps)
    num_frames = int(round(float(args.seconds) * fps))
    if num_frames <= 0:
        raise ValueError("seconds * fps must be positive.")

    index_map = joint_index_map(model)
    frames = []
    for _ in range(num_frames):
        mj.mj_forward(model, data)
        frames.append(build_frame(model, data, index_map))
        if args.step:
            target_time = data.time + 1.0 / float(fps)
            while data.time + 1e-9 < target_time:
                mj.mj_step(model, data)

    frames = np.asarray(frames, dtype=np.float32)
    save_amp_expert_txt(args.output, frames, fps)
    print(f"Saved {len(frames)} frames to {args.output}")
    print("Frame layout: right_leg_q(6), left_leg_q(6), right_leg_dq(6), left_leg_dq(6), left_foot_pos_local(3), right_foot_pos_local(3)")


if __name__ == "__main__":
    main()
