import argparse
import json
import pathlib
import sys

import mujoco as mj
import numpy as np
from scipy.spatial.transform import Rotation as R

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from general_motion_retargeting import RobotMotionViewer
from general_motion_retargeting.params import ROBOT_XML_DICT
from general_motion_retargeting.utils.lafan1 import load_bvh_file
from rich import print


def load_robot_init(path: pathlib.Path):
    with path.open("r", encoding="utf-8") as f:
        config = json.load(f)
    root_pos = np.asarray(config["root_pos"], dtype=np.float32).reshape(3)
    root_rot = np.asarray(config["root_rot"], dtype=np.float64).reshape(4)
    root_rot /= max(np.linalg.norm(root_rot), 1e-12)
    joints_dict = {k: float(v) for k, v in config["joints"].items()}
    if config.get("degrees", False):
        joints_dict = {k: float(np.deg2rad(v)) for k, v in joints_dict.items()}
    return root_pos, root_rot, joints_dict


def get_robot_dof_names(robot_name: str) -> list[str]:
    xml_path = pathlib.Path(ROBOT_XML_DICT[robot_name])
    model = mj.MjModel.from_xml_path(str(xml_path))
    dof_names = []
    for joint_id in range(1, model.njnt):
        joint_name = mj.mj_id2name(model, mj.mjtObj.mjOBJ_JOINT, joint_id)
        if joint_name is not None:
            dof_names.append(joint_name)
    return dof_names


def build_robot_dof_vector(robot_name: str, joints_dict: dict[str, float]) -> np.ndarray:
    dof_names = get_robot_dof_names(robot_name)
    dof_pos = np.zeros(len(dof_names), dtype=np.float32)
    assigned = 0
    for idx, joint_name in enumerate(dof_names):
        if joint_name in joints_dict:
            dof_pos[idx] = joints_dict[joint_name]
            assigned += 1
    print(f"[INFO] Robot init joints mapped: {assigned}/{len(dof_names)}")
    return dof_pos


def scale_human_data(
    human_data: dict, human_root_name: str, human_scale_table: dict, include_all_joints: bool = False
) -> dict:
    root_pos, root_quat = human_data[human_root_name]
    scaled_root_pos = float(human_scale_table[human_root_name]) * np.asarray(root_pos, dtype=np.float64)
    default_scale = float(human_scale_table[human_root_name])
    human_data_local = {}
    for body_name, (pos, rot) in human_data.items():
        if body_name == human_root_name:
            continue
        if body_name in human_scale_table:
            scale = float(human_scale_table[body_name])
        elif include_all_joints:
            scale = default_scale
        else:
            continue
        human_data_local[body_name] = (
            np.asarray(pos, dtype=np.float64) - np.asarray(root_pos, dtype=np.float64)
        ) * scale

    human_data_global = {
        human_root_name: (
            scaled_root_pos.astype(np.float32),
            np.asarray(root_quat, dtype=np.float32),
        )
    }
    for body_name, local_pos in human_data_local.items():
        human_data_global[body_name] = (
            (local_pos + scaled_root_pos).astype(np.float32),
            np.asarray(human_data[body_name][1], dtype=np.float32),
        )
    return human_data_global


def offset_human_data(human_data: dict, pos_offsets: dict[str, np.ndarray], rot_offsets: dict[str, R]) -> dict:
    offset_data = {}
    for body_name, (pos, quat) in human_data.items():
        pos = np.asarray(pos, dtype=np.float32)
        quat = np.asarray(quat, dtype=np.float64)
        if body_name not in pos_offsets or body_name not in rot_offsets:
            offset_data[body_name] = (pos, quat.astype(np.float32))
            continue

        updated_quat = (R.from_quat(quat, scalar_first=True) * rot_offsets[body_name]).as_quat(scalar_first=True)
        local_offset = pos_offsets[body_name]
        global_pos_offset = R.from_quat(updated_quat, scalar_first=True).apply(local_offset)
        offset_data[body_name] = ((pos + global_pos_offset).astype(np.float32), updated_quat.astype(np.float32))
    return offset_data


def build_mapping_table(
    ik_cfg: dict, table_name: str
) -> tuple[list[str], dict[str, str], dict[str, np.ndarray], dict[str, R]]:
    table = ik_cfg[table_name]
    robot_frames = []
    mapping = {}
    pos_offsets = {}
    rot_offsets = {}
    ground = float(ik_cfg.get("ground_height", 0.0)) * np.array([0.0, 0.0, 1.0], dtype=np.float64)
    for robot_link, entry in table.items():
        human_body = entry[0]
        robot_frames.append(robot_link)
        mapping[robot_link] = human_body
        pos_offsets[human_body] = np.asarray(entry[3], dtype=np.float64) - ground
        rot_offsets[human_body] = R.from_quat(np.asarray(entry[4], dtype=np.float64), scalar_first=True)
    return robot_frames, mapping, pos_offsets, rot_offsets


def filter_human_data_for_mapping(human_data: dict, mapping: dict[str, str], keep_root_name: str) -> dict:
    selected_names = set(mapping.values())
    selected_names.add(keep_root_name)
    return {name: value for name, value in human_data.items() if name in selected_names}


def get_bvh_fps(bvh_path: pathlib.Path) -> float:
    with bvh_path.open("r", encoding="utf-8") as f:
        for line in f:
            stripped = line.strip()
            if stripped.startswith("Frame Time"):
                frame_time = float(stripped.split()[-1])
                return 1.0 / frame_time
    return 30.0


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Visualize one BVH frame and the robot under a given IK config."
    )
    parser.add_argument("--bvh_file", required=True, type=pathlib.Path)
    parser.add_argument("--robot", required=True, type=str)
    parser.add_argument("--ik_config", required=True, type=pathlib.Path)
    parser.add_argument("--robot_qpos_init", required=True, type=pathlib.Path)
    parser.add_argument("--frame_idx", type=int, default=0, help="0-based BVH frame index to visualize.")
    parser.add_argument("--format", choices=["lafan1", "nokov"], default="lafan1")
    parser.add_argument("--table", choices=["ik_match_table1", "ik_match_table2"], default="ik_match_table2")
    parser.add_argument("--show_all_human", action="store_true", help="Show all scaled human joints, not only mapped ones.")
    parser.add_argument("--camera_follow", action="store_true")
    args = parser.parse_args()

    with args.ik_config.open("r", encoding="utf-8") as f:
        ik_cfg = json.load(f)

    human_frames, actual_human_height = load_bvh_file(str(args.bvh_file), format=args.format)
    if not human_frames:
        raise RuntimeError(f"No frames loaded from {args.bvh_file}")
    frame_idx = max(0, min(args.frame_idx, len(human_frames) - 1))
    human_frame = human_frames[frame_idx]

    ratio = actual_human_height / float(ik_cfg["human_height_assumption"])
    human_scale_table = {
        key: float(value) * ratio for key, value in ik_cfg["human_scale_table"].items()
    }
    human_root_name = ik_cfg["human_root_name"]
    scaled_human_data = scale_human_data(
        human_frame,
        human_root_name,
        human_scale_table,
        include_all_joints=args.show_all_human,
    )

    robot_root_pos_init, robot_root_rot_init, robot_joints_init = load_robot_init(args.robot_qpos_init)
    robot_dof_pos = build_robot_dof_vector(args.robot, robot_joints_init)

    robot_frames, mapping, pos_offsets, rot_offsets = build_mapping_table(ik_cfg, args.table)
    offset_scaled_human_data = offset_human_data(scaled_human_data, pos_offsets, rot_offsets)
    human_display_data = offset_scaled_human_data if args.show_all_human else filter_human_data_for_mapping(
        offset_scaled_human_data, mapping, human_root_name
    )

    fps = get_bvh_fps(args.bvh_file)
    viewer = RobotMotionViewer(
        robot_type=args.robot,
        motion_fps=fps,
        camera_follow=args.camera_follow,
        record_video=False,
    )

    robot_root_pos = np.asarray(offset_scaled_human_data[human_root_name][0], dtype=np.float32)
    viewer.viewer.cam.lookat[:] = robot_root_pos + np.array([0.0, 0.0, -0.4], dtype=np.float32)

    print(f"[INFO] Using BVH frame {frame_idx + 1}/{len(human_frames)}")
    print(f"[INFO] Human root: {human_root_name} | Robot root: {ik_cfg['robot_root_name']}")
    print(f"[INFO] Visualizing table: {args.table}")
    print("[INFO] Applying per-body offsets from selected match table.")
    print("[INFO] Robot link -> Human body mapping:")
    for robot_link, human_body in mapping.items():
        print(f"  {robot_link} <- {human_body}")

    try:
        while True:
            viewer.step(
                root_pos=robot_root_pos,
                root_rot=robot_root_rot_init,
                dof_pos=robot_dof_pos,
                human_motion_data=human_display_data,
                human_pos_offset=np.array([0.0, 0.0, 0.0], dtype=np.float32),
                show_human_body_name=True,
                robot_frames=robot_frames,
                show_robot_body_name=True,
                rate_limit=True,
                follow_camera=args.camera_follow,
            )
    finally:
        viewer.close()


if __name__ == "__main__":
    main()
