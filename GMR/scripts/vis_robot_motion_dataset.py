import pathlib
import sys

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from general_motion_retargeting import RobotMotionViewer, load_robot_motion
import argparse
import os
import numpy as np
from tqdm import tqdm

paused = False
motion_num = 0
motion_id = 0
current_motion_id = -1


def get_frame_root_velocity(motion_data, frame_idx):
    root_vel = motion_data.get("root_vel")
    if root_vel is not None:
        return np.asarray(root_vel[frame_idx], dtype=np.float32)

    root_pos = motion_data["motion_root_pos"]
    motion_fps = float(motion_data["motion_fps"])
    num_frames = len(root_pos)
    if num_frames <= 1 or motion_fps <= 0:
        return np.zeros(3, dtype=np.float32)

    next_idx = min(frame_idx + 1, num_frames - 1)
    prev_idx = max(frame_idx - 1, 0)
    if next_idx == frame_idx:
        delta = np.asarray(root_pos[frame_idx], dtype=np.float32) - np.asarray(root_pos[prev_idx], dtype=np.float32)
    else:
        delta = np.asarray(root_pos[next_idx], dtype=np.float32) - np.asarray(root_pos[frame_idx], dtype=np.float32)
    return delta * motion_fps


def format_speed_status(motion_data, frame_idx):
    vel = get_frame_root_velocity(motion_data, frame_idx)
    speed_xy = float(np.linalg.norm(vel[:2]))
    speed_xyz = float(np.linalg.norm(vel))
    return (
        f"vx={vel[0]:+.3f} vy={vel[1]:+.3f} vz={vel[2]:+.3f} m/s | "
        f"speed_xy={speed_xy:.3f} speed_xyz={speed_xyz:.3f} m/s"
    )


def compute_mean_speed_xy(motion_data):
    num_frames = len(motion_data["motion_root_pos"])
    if num_frames <= 0:
        return 0.0
    speeds_xy = [float(np.linalg.norm(get_frame_root_velocity(motion_data, idx)[:2])) for idx in range(num_frames)]
    return float(np.mean(speeds_xy)) if speeds_xy else 0.0


def print_controls():
    print("Controls:")
    print("  Space: pause/resume")
    print("  [: previous motion")
    print("  ]: next motion")


def keyboard_callback(keycode):
    global paused, motion_id, motion_num
    if chr(keycode) == ' ':
        paused = not paused
    if chr(keycode) == '[':
        motion_id = (motion_id - 1) % motion_num
    if chr(keycode) == ']':
        motion_id = (motion_id + 1) % motion_num

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--robot", type=str, default="unitree_g1")
    parser.add_argument("--robot_motion_folder", type=str, required=True)
    parser.add_argument("--camera_follow", action="store_true")
    parser.add_argument("--camera_lookat_height", type=float, default=0.25)
    parser.add_argument("--camera_distance", type=float, default=3.0)
    parser.add_argument("--camera_azimuth", type=float, default=135.0)
    parser.add_argument("--camera_elevation", type=float, default=-5.0)
    parser.add_argument("--record_video", action="store_true")
    parser.add_argument("--video_path", type=str, 
                        default="videos/example.mp4")
                        
    args = parser.parse_args()
    
    robot_type = args.robot
    robot_motion_folder = args.robot_motion_folder
    
    if not os.path.exists(robot_motion_folder):
        raise FileNotFoundError(f"Motion data dir {robot_motion_folder} does not exist.")
    
    motion_files = sorted(
        [
            str(path.relative_to(robot_motion_folder))
            for path in pathlib.Path(robot_motion_folder).rglob("*.pkl")
        ]
    )
    motion_num = len(motion_files)
    if motion_num == 0:
        raise FileNotFoundError(f"No .pkl motion files found under {robot_motion_folder}.")
    print(f"Found {motion_num} motion files in {robot_motion_folder}, loading...")
    motion_dataset = []
    for motion_file in tqdm(motion_files):
        motion_path = os.path.join(robot_motion_folder, motion_file)
        motion_data, motion_fps, motion_root_pos, motion_root_rot, motion_dof_pos, motion_local_body_pos, motion_link_body_list = load_robot_motion(motion_path)
        motion_dataset.append({
            "motion_file": motion_file,
            "motion_data": motion_data,
            "root_vel": motion_data.get("root_vel"),
            "motion_fps": motion_fps,
            "motion_root_pos": motion_root_pos,
            "motion_root_rot": motion_root_rot,
            "motion_dof_pos": motion_dof_pos,
            "motion_local_body_pos": motion_local_body_pos,
            "motion_link_body_list": motion_link_body_list,
        })
    print("Loading done.")
    print_controls()
    default_motion_fps = motion_dataset[0]["motion_fps"]
    
    env = RobotMotionViewer(robot_type=robot_type,
                            motion_fps=default_motion_fps,
                            camera_follow=args.camera_follow,
                            camera_lookat_height=args.camera_lookat_height,
                            camera_distance=args.camera_distance,
                            camera_azimuth=args.camera_azimuth,
                            camera_elevation=args.camera_elevation,
                            record_video=args.record_video, video_path=args.video_path, 
                            keyboard_callback=keyboard_callback)
    
    frame_idx = 0
    while True:
        # get current motion
        if current_motion_id != motion_id:
            current_motion_id = motion_id
            frame_idx = 0
            motion_data = motion_dataset[motion_id]
            motion_file = motion_data["motion_file"]
            motion_fps = motion_data["motion_fps"]
            motion_root_pos = motion_data["motion_root_pos"]
            motion_root_rot = motion_data["motion_root_rot"]
            motion_dof_pos = motion_data["motion_dof_pos"]
            print()
            print(
                f"[motion {motion_id + 1}/{motion_num}] "
                f"{motion_file} | fps={motion_fps:.3f} | frames={len(motion_root_pos)} | "
                f"mean_speed_xy={compute_mean_speed_xy(motion_data):.3f} m/s",
                flush=True,
            )
        
        
        if not paused:
            current_frame_idx = frame_idx
            env.step(motion_root_pos[frame_idx], 
                    motion_root_rot[frame_idx], 
                    motion_dof_pos[frame_idx], 
                    rate_limit=True,
                    follow_camera=args.camera_follow)
            status = format_speed_status(motion_data, current_frame_idx)
            current_time_s = current_frame_idx / motion_fps if motion_fps > 0 else 0.0
            total_time_s = len(motion_root_pos) / motion_fps if motion_fps > 0 else 0.0
            print(
                f"\rframe {current_frame_idx + 1}/{len(motion_root_pos)} | "
                f"time {current_time_s:.3f}/{total_time_s:.3f}s | {status}",
                end="",
                flush=True,
            )
            frame_idx += 1
            if frame_idx >= len(motion_root_pos):
                print(
                    f"\nsegment finished: {motion_file} | "
                    f"mean_speed_xy={compute_mean_speed_xy(motion_data):.3f} m/s",
                    flush=True,
                )
                frame_idx = 0
    env.close()
