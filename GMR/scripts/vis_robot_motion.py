import argparse
import os
import pathlib
import pickle
import sys

import mujoco as mj
import numpy as np

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from general_motion_retargeting import RobotMotionViewer, load_robot_motion


paused = False
frame_idx = 0
num_frames = 0
motion_fps = 30.0
recorded_marks = []
motion_data = None
env = None
motion_root_pos = None
motion_root_rot = None
motion_dof_pos = None
tracked_links = []
augmented_output_path = ""

DEFAULT_FOOT_LINKS = {
    "roboot16": ["left_ankle_roll_link", "right_ankle_roll_link"],
    "unitree_g1": ["left_ankle_roll_link", "right_ankle_roll_link"],
    "unitree_g1_auto": ["left_ankle_roll_link", "right_ankle_roll_link"],
    "unitree_g1_with_hands": ["left_ankle_roll_link", "right_ankle_roll_link"],
}


def safe_key_char(keycode):
    try:
        return chr(keycode)
    except (OverflowError, ValueError):
        return ""


def clamp_frame_index(idx, total_frames):
    if total_frames <= 0:
        return 0
    return idx % total_frames


def format_time(seconds):
    total_millis = int(round(seconds * 1000.0))
    millis = total_millis % 1000
    total_seconds = total_millis // 1000
    seconds_part = total_seconds % 60
    total_minutes = total_seconds // 60
    minutes = total_minutes % 60
    hours = total_minutes // 60
    return f"{hours:02d}:{minutes:02d}:{seconds_part:02d}.{millis:03d}"


def current_time_seconds():
    if motion_fps <= 0:
        return 0.0
    return frame_idx / motion_fps


def print_current_status(prefix="Current"):
    if num_frames <= 0:
        return
    current_seconds = current_time_seconds()
    total_seconds = num_frames / motion_fps if motion_fps > 0 else 0.0
    print(
        f"{prefix}: frame {frame_idx + 1}/{num_frames} | "
        f"time {format_time(current_seconds)} / {format_time(total_seconds)} "
        f"({current_seconds:.3f}s / {total_seconds:.3f}s)"
    )


def keyboard_callback(keycode):
    global frame_idx, paused, recorded_marks

    key_char = safe_key_char(keycode)
    upper_char = key_char.upper()

    if upper_char == "R":
        frame_idx = 0
        print_current_status("Reset")
    elif key_char == " ":
        paused = not paused
        print_current_status("Paused" if paused else "Resumed")
    elif keycode == 256 or upper_char == "Q":
        print("Exit viewer")
        os._exit(0)
    elif keycode == 262:
        frame_idx = clamp_frame_index(frame_idx + 1, num_frames)
        paused = True
        print_current_status("Step forward")
    elif keycode == 263:
        frame_idx = clamp_frame_index(frame_idx - 1, num_frames)
        paused = True
        print_current_status("Step backward")
    elif upper_char == "M":
        mark_seconds = current_time_seconds()
        recorded_marks.append((frame_idx, mark_seconds))
        print(
            f"Mark {len(recorded_marks)}: frame {frame_idx + 1}/{num_frames} | "
            f"time {format_time(mark_seconds)} ({mark_seconds:.3f}s)"
        )
    elif upper_char == "C":
        recorded_marks.clear()
        print("Cleared all marks")
    elif upper_char == "P":
        if env is None or motion_data is None or motion_root_pos is None or motion_root_rot is None or motion_dof_pos is None:
            print("Motion data is not ready for saving")
        elif not tracked_links:
            print("No tracked links configured")
        else:
            save_augmented_motion(
                env,
                motion_data,
                motion_root_pos,
                motion_root_rot,
                motion_dof_pos,
                tracked_links,
                augmented_output_path,
            )
    elif upper_char == "H":
        print_controls()


def print_controls():
    print("Controls:")
    print("  Space: pause/resume")
    print("  Left / Right: single-frame step and pause")
    print("  R: reset to frame 1")
    print("  M: record current timestamp")
    print("  C: clear recorded timestamps")
    print("  P: save current motion with tracked foot link positions")
    print("  H: print controls")
    print("  Q / Esc: quit")


def parse_tracked_links(tracked_links_text, robot_type):
    if tracked_links_text:
        return [item.strip() for item in tracked_links_text.split(",") if item.strip()]
    if robot_type in DEFAULT_FOOT_LINKS:
        return DEFAULT_FOOT_LINKS[robot_type]
    return []


def get_default_augmented_path(robot_motion_path):
    input_path = pathlib.Path(robot_motion_path)
    return str(input_path.with_name(f"{input_path.stem}_with_foot_links.pkl"))


def collect_link_positions(env, root_pos, root_rot, dof_pos, tracked_links):
    env.data.qpos[:3] = root_pos
    env.data.qpos[3:7] = root_rot
    env.data.qpos[7:] = dof_pos
    mj.mj_forward(env.model, env.data)

    link_positions = []
    for link_name in tracked_links:
        body_id = env.model.body(link_name).id
        link_positions.append(env.data.xpos[body_id].copy())
    return np.asarray(link_positions, dtype=np.float32)


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


def save_augmented_motion(
    env,
    motion_data_dict,
    motion_root_pos,
    motion_root_rot,
    motion_dof_pos,
    tracked_links,
    output_path,
):
    if not tracked_links:
        raise ValueError("No tracked links configured for saving")

    num_motion_frames = len(motion_root_pos)
    tracked_link_positions_world = np.zeros((num_motion_frames, len(tracked_links), 3), dtype=np.float32)
    tracked_link_positions_local = np.zeros((num_motion_frames, len(tracked_links), 3), dtype=np.float32)

    for idx in range(num_motion_frames):
        tracked_link_positions_world[idx] = collect_link_positions(
            env,
            motion_root_pos[idx],
            motion_root_rot[idx],
            motion_dof_pos[idx],
            tracked_links,
        )
        tracked_link_positions_local[idx] = world_to_root_local_positions(
            tracked_link_positions_world[idx],
            motion_root_pos[idx],
            motion_root_rot[idx],
        )

    augmented_motion_data = dict(motion_data_dict)
    augmented_motion_data["foot_link_body_pos"] = tracked_link_positions_local
    augmented_motion_data["foot_link_body_list"] = list(tracked_links)
    augmented_motion_data["foot_link_body_pos_world"] = tracked_link_positions_world

    # Align with existing dataset field naming: local_body_pos is root-local.
    augmented_motion_data["local_body_pos"] = tracked_link_positions_local
    augmented_motion_data["link_body_list"] = list(tracked_links)

    meta = dict(augmented_motion_data.get("meta", {}))
    meta["foot_link_body_pos_space"] = "root_local"
    meta["foot_link_body_pos_world_space"] = "world"
    meta["foot_link_body_source"] = "vis_robot_motion"
    augmented_motion_data["meta"] = meta

    output_file = pathlib.Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with output_file.open("wb") as f:
        pickle.dump(augmented_motion_data, f)

    print(f"Saved augmented motion to {output_file}")
    print(f"Stored root-local foot positions for AMP in: foot_link_body_pos / local_body_pos")


def draw_capsule(viewer, start, end, radius, rgba, label=""):
    geom = viewer.user_scn.geoms[viewer.user_scn.ngeom]
    mj.mjv_initGeom(
        geom,
        type=mj.mjtGeom.mjGEOM_CAPSULE,
        size=np.array([radius, radius, radius]),
        pos=np.zeros(3),
        mat=np.eye(3).flatten(),
        rgba=np.array(rgba),
    )
    mj.mjv_connector(
        geom,
        type=mj.mjtGeom.mjGEOM_CAPSULE,
        width=radius,
        from_=start,
        to=end,
    )
    geom.label = label
    viewer.user_scn.ngeom += 1


def draw_sphere(viewer, pos, radius, rgba, label=""):
    geom = viewer.user_scn.geoms[viewer.user_scn.ngeom]
    mj.mjv_initGeom(
        geom,
        type=mj.mjtGeom.mjGEOM_SPHERE,
        size=np.array([radius, radius, radius]),
        pos=pos,
        mat=np.eye(3).flatten(),
        rgba=np.array(rgba),
    )
    geom.label = label
    viewer.user_scn.ngeom += 1


def draw_timeline(env, displayed_frame_idx, total_frames, fps):
    if total_frames <= 0 or fps <= 0:
        return

    root_pos = env.data.qpos[:3].copy()
    current_seconds = displayed_frame_idx / fps
    total_seconds = total_frames / fps
    progress = displayed_frame_idx / max(total_frames - 1, 1)

    bar_half_length = 0.45
    bar_height = 1.15
    start = root_pos + np.array([-bar_half_length, 0.0, bar_height], dtype=float)
    end = root_pos + np.array([bar_half_length, 0.0, bar_height], dtype=float)
    marker = start + (end - start) * progress

    draw_capsule(env.viewer, start, end, 0.012, [0.15, 0.75, 0.95, 0.85])
    draw_sphere(env.viewer, start, 0.02, [0.2, 0.9, 0.4, 0.95], "start")
    draw_sphere(env.viewer, end, 0.02, [0.95, 0.35, 0.35, 0.95], f"end {format_time(total_seconds)}")

    status_label = (
        f"frame {displayed_frame_idx + 1}/{total_frames} | "
        f"{format_time(current_seconds)} / {format_time(total_seconds)}"
    )
    draw_sphere(env.viewer, marker, 0.03, [1.0, 0.9, 0.2, 0.98], status_label)


def render_frame(env, root_pos, root_rot, dof_pos, displayed_frame_idx, total_frames, fps):
    env.data.qpos[:3] = root_pos
    env.data.qpos[3:7] = root_rot
    env.data.qpos[7:] = dof_pos

    mj.mj_forward(env.model, env.data)

    if env.camera_follow:
        env.viewer.cam.lookat = env.data.xpos[env.model.body(env.robot_base).id]
        env.viewer.cam.distance = env.viewer_cam_distance
        env.viewer.cam.elevation = -10

    env.viewer.user_scn.ngeom = 0
    draw_timeline(env, displayed_frame_idx, total_frames, fps)

    env.viewer.sync()
    env.rate_limiter.sleep()

    if env.record_video:
        env.renderer.update_scene(env.data, camera=env.viewer.cam)
        img = env.renderer.render()
        env.mp4_writer.append_data(img)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--robot", type=str, default="unitree_g1")
    parser.add_argument("--robot_motion_path", type=str, required=True)
    parser.add_argument("--record_video", action="store_true")
    parser.add_argument("--video_path", type=str, default="videos/example.mp4")
    parser.add_argument(
        "--tracked_links",
        type=str,
        default="",
        help="Comma-separated body names to save. Defaults to foot links for known robots.",
    )
    parser.add_argument(
        "--save_augmented_pkl",
        action="store_true",
        help="Compute tracked link positions for the whole motion and save them to a new pkl before playback.",
    )
    parser.add_argument(
        "--augmented_output_path",
        type=str,
        default="",
        help="Output path for the augmented pkl. Defaults to <input>_with_foot_links.pkl",
    )
    args = parser.parse_args()

    robot_type = args.robot
    robot_motion_path = args.robot_motion_path

    if not os.path.exists(robot_motion_path):
        raise FileNotFoundError(f"Motion file {robot_motion_path} not found")

    (
        motion_data,
        motion_fps_loaded,
        motion_root_pos,
        motion_root_rot,
        motion_dof_pos,
        motion_local_body_pos,
        motion_link_body_list,
    ) = load_robot_motion(robot_motion_path)

    motion_fps = float(motion_fps_loaded)
    num_frames = len(motion_root_pos)
    tracked_links = parse_tracked_links(args.tracked_links, robot_type)
    augmented_output_path = args.augmented_output_path or get_default_augmented_path(robot_motion_path)

    env = RobotMotionViewer(
        robot_type=robot_type,
        motion_fps=motion_fps,
        camera_follow=False,
        record_video=args.record_video,
        video_path=args.video_path,
        keyboard_callback=keyboard_callback,
    )

    total_seconds = num_frames / motion_fps if motion_fps > 0 else 0.0
    print(
        f"Loaded motion: {robot_motion_path} | "
        f"fps={motion_fps:.3f} | frames={num_frames} | duration={format_time(total_seconds)}"
    )
    if tracked_links:
        print(f"Tracked links: {tracked_links}")
    else:
        print("Tracked links: none")
    print_controls()

    if args.save_augmented_pkl:
        save_augmented_motion(
            env,
            motion_data,
            motion_root_pos,
            motion_root_rot,
            motion_dof_pos,
            tracked_links,
            augmented_output_path,
        )

    while True:
        frame_idx = clamp_frame_index(frame_idx, num_frames)
        displayed_frame_idx = frame_idx

        render_frame(
            env,
            motion_root_pos[displayed_frame_idx],
            motion_root_rot[displayed_frame_idx],
            motion_dof_pos[displayed_frame_idx],
            displayed_frame_idx,
            num_frames,
            motion_fps,
        )

        if not paused:
            frame_idx = clamp_frame_index(displayed_frame_idx + 1, num_frames)

    env.close()
