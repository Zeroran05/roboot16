import argparse
import os
import pathlib
import sys
import time
from typing import Optional

import numpy as np
from rich import print
from scipy.spatial.transform import Rotation as R
from tqdm import tqdm

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from bvh_to_robot import (  # noqa: E402
    check_exported_root_velocities,
    get_bvh_frame_rate,
    get_robot_dof_names,
    resample_bvh_data,
)
from general_motion_retargeting import RobotMotionViewer  # noqa: E402
from general_motion_retargeting.motion_retarget_grounded import GroundedMotionRetargeting as GMR  # noqa: E402
from general_motion_retargeting.utils.lafan1 import load_bvh_file  # noqa: E402


def _parse_csv_names(value: Optional[str]) -> Optional[list[str]]:
    if value is None or value.strip() == "":
        return None
    return [item.strip() for item in value.split(",") if item.strip()]


def _parse_csv_ints(value: Optional[str]) -> Optional[tuple[int, ...]]:
    if value is None:
        return None
    value = value.strip()
    if value == "" or value.lower() in {"none", "all"}:
        return None
    return tuple(int(item.strip()) for item in value.split(",") if item.strip())


def _unpack_retarget_result(ret):
    if isinstance(ret, tuple):
        if len(ret) == 3:
            return ret
        if len(ret) == 2:
            qpos, qvel = ret
            return qpos, qvel, qvel
        raise RuntimeError(f"Unexpected retarget() return length: {len(ret)}")
    raise RuntimeError("retarget() should return a tuple")


def _export_motion(args, qpos_list, qvel_list, frame_dt_list, aligned_fps):
    import pickle

    qpos_arr = np.asarray(qpos_list)
    qvel_arr = np.asarray(qvel_list)
    frame_dt_arr = np.asarray(frame_dt_list, dtype=np.float32)

    try:
        check_exported_root_velocities(
            qpos_seq=qpos_arr,
            qvel_seq=qvel_arr,
            dt_list=frame_dt_arr,
            lin_tol=getattr(args, "lin_tol", 5e-3),
            ang_tol=getattr(args, "ang_tol", 5e-3),
            strict=getattr(args, "vel_check_strict", False),
        )
    except Exception as exc:
        print(f"[WARN] export-time velocity validation failed: {exc}")

    root_pos = qpos_arr[:, :3]
    root_rot_wxyz = qpos_arr[:, 3:7]
    root_rot_xyzw = root_rot_wxyz[:, [1, 2, 3, 0]]
    dof_pos = qpos_arr[:, 7:]

    root_vel_world = qvel_arr[:, 0:3]
    root_rot_vel = qvel_arr[:, 3:6]
    dof_vel = qvel_arr[:, 6:]

    length = min(
        root_pos.shape[0],
        root_rot_xyzw.shape[0],
        dof_pos.shape[0],
        root_vel_world.shape[0],
        root_rot_vel.shape[0],
        dof_vel.shape[0],
        frame_dt_arr.shape[0],
    )
    start = 5 if length > 5 else 0
    root_pos = root_pos[start:length].astype(np.float32)
    root_rot_wxyz = root_rot_wxyz[start:length].astype(np.float32)
    root_rot_xyzw = root_rot_xyzw[start:length].astype(np.float32)
    dof_pos = dof_pos[start:length].astype(np.float32)
    root_vel_world = root_vel_world[start:length].astype(np.float32)
    root_rot_vel = root_rot_vel[start:length].astype(np.float32)
    dof_vel = dof_vel[start:length].astype(np.float32)

    root_to_world = R.from_quat(root_rot_wxyz, scalar_first=True).as_matrix()
    world_to_root = np.transpose(root_to_world, (0, 2, 1))
    root_vel_body = np.einsum("tij,tj->ti", world_to_root, root_vel_world).astype(np.float32)

    export_data = {
        "motion_file": pathlib.Path(args.bvh_file).stem,
        "root_pos": root_pos,
        "root_vel": root_vel_world,
        "root_vel_body": root_vel_body,
        "root_rot": root_rot_xyzw,
        "root_rot_vel": root_rot_vel,
        "dof_pos": dof_pos,
        "dof_vel": dof_vel,
        "dof_names": get_robot_dof_names(args.robot),
        "local_body_pos": None,
        "link_body_list": None,
        "fps": float(aligned_fps),
        "meta": {
            "root_rot_convention": "xyzw",
            "root_ang_vel_space": "local",
            "root_lin_vel_space_world": "world",
            "root_lin_vel_space_body": "local",
            "retargeter": "GroundedMotionRetargeting",
            "ground_height": float(args.ground_height),
            "ground_clearance": float(args.ground_clearance),
            "foot_bodies": _parse_csv_names(args.foot_bodies),
            "foot_geom_groups": _parse_csv_ints(args.foot_geom_groups),
            "ground_mode": args.ground_mode,
            "contact_source": args.contact_source,
            "support_height_mode": args.support_height_mode,
            "calibrated_sole": not args.disable_calibrated_sole,
            "foot_flattening": not args.disable_foot_flattening,
            "sequence_foot_flattening": not args.disable_sequence_foot_flattening,
            "sequence_ground_smoothing": not args.disable_sequence_ground_smoothing,
            "sequence_ground_window": int(args.sequence_ground_window),
            "sequence_ground_passes": int(args.sequence_ground_passes),
            "sequence_ground_max_step_down": float(args.sequence_ground_max_step_down),
            "sequence_ground_max_step_up": float(args.sequence_ground_max_step_up),
            "stance_lock": not args.disable_stance_lock,
        },
    }

    save_dir = os.path.dirname(args.save_path)
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
    with open(args.save_path, "wb") as file:
        pickle.dump(export_data, file)
    print(f"Saved grounded motion to {args.save_path}")


def main():
    parser = argparse.ArgumentParser(description="BVH to robot retargeting with ground projection constraints.")
    parser.add_argument("--bvh_file", required=True, type=str, help="BVH motion file to load.")
    parser.add_argument("--format", choices=["lafan1", "nokov"], default="lafan1")
    parser.add_argument("--loop", default=False, action="store_true")
    parser.add_argument(
        "--robot",
        choices=[
            "unitree_g1",
            "unitree_g1_auto",
            "unitree_g1_with_hands",
            "booster_t1",
            "stanford_toddy",
            "fourier_n1",
            "engineai_pm01",
            "pal_talos",
            "roboot16",
        ],
        default="roboot16",
    )
    parser.add_argument("--record_video", action="store_true", default=False)
    parser.add_argument("--video_path", type=str, default="videos/grounded_retarget.mp4")
    parser.add_argument("--rate_limit", action="store_true", default=False)
    parser.add_argument("--headless", action="store_true", default=False)
    parser.add_argument("--save_path", default=None, help="Path to save the robot motion pkl.")
    parser.add_argument("--motion_fps", default=None, type=float)
    parser.add_argument("--offset_to_ground", action="store_true", default=False)

    parser.add_argument("--ground_height", type=float, default=0.0)
    parser.add_argument("--ground_clearance", type=float, default=0.005)
    parser.add_argument(
        "--ground_mode",
        choices=["support", "prevent_penetration"],
        default="support",
        help="support also pulls detected stance feet toward the floor; prevent_penetration only lifts feet above it.",
    )
    parser.add_argument(
        "--foot_bodies",
        type=str,
        default="left_ankle_roll_link,right_ankle_roll_link",
        help="Comma-separated robot foot body names used for ground projection.",
    )
    parser.add_argument(
        "--foot_geom_groups",
        type=str,
        default="3",
        help="Comma-separated MuJoCo geom groups used as foot contact geometry. Use 'all' to include all geoms.",
    )
    parser.add_argument("--max_root_z_correction", type=float, default=0.25)
    parser.add_argument("--max_root_z_step_down", type=float, default=0.025)
    parser.add_argument("--max_root_z_step_up", type=float, default=0.08)
    parser.add_argument("--root_z_pull_down_alpha", type=float, default=0.45)
    parser.add_argument("--root_z_lift_alpha", type=float, default=1.0)
    parser.add_argument(
        "--contact_source",
        choices=["human", "robot", "hybrid"],
        default="hybrid",
        help="Where stance-foot contact is inferred from. PHUMA-style grounding usually wants human or hybrid.",
    )
    parser.add_argument(
        "--support_height_mode",
        choices=["lowest", "mean", "highest"],
        default="mean",
        help="Foot height representative used by support grounding. mean uses fore/rear/heel groups when available.",
    )
    parser.add_argument("--human_contact_height_threshold", type=float, default=0.06)
    parser.add_argument("--human_contact_speed_threshold", type=float, default=0.35)
    parser.add_argument("--disable_calibrated_sole", action="store_true", default=False)
    parser.add_argument("--disable_foot_flattening", action="store_true", default=False)
    parser.add_argument("--foot_flatten_iterations", type=int, default=2)
    parser.add_argument("--foot_flatten_step", type=float, default=0.035)
    parser.add_argument("--foot_flatten_tracking_weight", type=float, default=0.08)
    parser.add_argument("--disable_sequence_foot_flattening", action="store_true", default=False)
    parser.add_argument("--sequence_foot_flatten_contact_pre_roll", type=int, default=8)
    parser.add_argument("--sequence_foot_flatten_contact_post_roll", type=int, default=3)
    parser.add_argument("--disable_sequence_ground_smoothing", action="store_true", default=False)
    parser.add_argument("--sequence_ground_window", type=int, default=11)
    parser.add_argument("--sequence_ground_passes", type=int, default=2)
    parser.add_argument("--sequence_ground_max_step_down", type=float, default=0.006)
    parser.add_argument("--sequence_ground_max_step_up", type=float, default=0.04)
    parser.add_argument("--sequence_ground_contact_pre_roll", type=int, default=6)
    parser.add_argument("--sequence_ground_contact_post_roll", type=int, default=2)
    parser.add_argument("--disable_stance_lock", action="store_true", default=False)
    parser.add_argument("--stance_height_threshold", type=float, default=0.04)
    parser.add_argument("--stance_speed_threshold", type=float, default=0.35)
    parser.add_argument("--stance_xy_blend", type=float, default=0.75)
    parser.add_argument("--max_stance_xy_correction", type=float, default=0.04)
    parser.add_argument("--disable_velocity_recompute", action="store_true", default=False)

    parser.add_argument("--vel_check_strict", action="store_true", default=False)
    parser.add_argument("--lin_tol", type=float, default=5e-3)
    parser.add_argument("--ang_tol", type=float, default=5e-3)
    args = parser.parse_args()

    motion_frames, actual_human_height = load_bvh_file(args.bvh_file, format=args.format)
    src_fps = get_bvh_frame_rate(args.bvh_file)
    if src_fps is None:
        src_fps = args.motion_fps if args.motion_fps is not None else 30.0
        print(f"[WARN] Could not detect frame rate from BVH file, using {src_fps:.3f} Hz")
    else:
        print(f"[INFO] Detected source FPS from BVH file: {src_fps:.3f} Hz")

    target_fps = args.motion_fps if args.motion_fps is not None else src_fps
    if abs(src_fps - target_fps) > 1e-6:
        motion_frames = resample_bvh_data(motion_frames, src_fps, target_fps)
        aligned_fps = target_fps
        print(f"[INFO] Resampled to {len(motion_frames)} frames at {aligned_fps:.3f} Hz")
    else:
        aligned_fps = src_fps
        print(f"[INFO] Using original {len(motion_frames)} frames at {aligned_fps:.3f} Hz")

    retargeter = GMR(
        src_human=f"bvh_{args.format}",
        tgt_robot=args.robot,
        actual_human_height=actual_human_height,
        ground_height=args.ground_height,
        ground_clearance=args.ground_clearance,
        foot_bodies=_parse_csv_names(args.foot_bodies),
        foot_geom_groups=_parse_csv_ints(args.foot_geom_groups),
        ground_mode=args.ground_mode,
        max_root_z_correction=args.max_root_z_correction,
        max_root_z_step_down=args.max_root_z_step_down,
        max_root_z_step_up=args.max_root_z_step_up,
        root_z_pull_down_alpha=args.root_z_pull_down_alpha,
        root_z_lift_alpha=args.root_z_lift_alpha,
        contact_source=args.contact_source,
        support_height_mode=args.support_height_mode,
        use_calibrated_sole=not args.disable_calibrated_sole,
        human_contact_height_threshold=args.human_contact_height_threshold,
        human_contact_speed_threshold=args.human_contact_speed_threshold,
        enable_foot_flattening=not args.disable_foot_flattening,
        foot_flatten_iterations=args.foot_flatten_iterations,
        foot_flatten_step=args.foot_flatten_step,
        foot_flatten_tracking_weight=args.foot_flatten_tracking_weight,
        enable_stance_lock=not args.disable_stance_lock,
        stance_height_threshold=args.stance_height_threshold,
        stance_speed_threshold=args.stance_speed_threshold,
        stance_xy_blend=args.stance_xy_blend,
        max_stance_xy_correction=args.max_stance_xy_correction,
        recompute_velocity=not args.disable_velocity_recompute,
    )

    viewer_xml_override = None
    if args.robot == "roboot16":
        candidate_scene = REPO_ROOT / "Roboot1.6" / "xml" / "scene_1.xml"
        if candidate_scene.exists():
            viewer_xml_override = str(candidate_scene)

    robot_motion_viewer = None
    if not args.headless:
        robot_motion_viewer = RobotMotionViewer(
            robot_type=args.robot,
            motion_fps=aligned_fps,
            transparent_robot=0,
            record_video=args.record_video,
            video_path=args.video_path,
            xml_path_override=viewer_xml_override,
        )

    qpos_list = []
    qvel_list = []
    stance_feet_list = []
    frame_dt_list = [0.0]
    target_dt = 1.0 / float(aligned_fps)
    desired_dt = target_dt
    next_frame_time = time.perf_counter()
    frame_index = 0

    pbar = tqdm(total=len(motion_frames), desc="Grounded retargeting")
    try:
        while True:
            if args.rate_limit:
                now = time.perf_counter()
                if now < next_frame_time:
                    time.sleep(next_frame_time - now)
                    next_frame_time += desired_dt
                else:
                    missed = int((now - next_frame_time) // desired_dt) + 1
                    next_frame_time += missed * desired_dt

            frame = motion_frames[frame_index]
            qpos, _qvel_last, qvel = _unpack_retarget_result(
                retargeter.retarget(
                    frame,
                    offset_to_ground=args.offset_to_ground,
                    frame_dt_target=target_dt,
                )
            )
            dt_this = getattr(retargeter, "last_frame_dt", None)
            if not (isinstance(dt_this, (float, np.floating)) and np.isfinite(dt_this) and dt_this > 0.0):
                dt_this = target_dt

            if robot_motion_viewer is not None:
                robot_motion_viewer.step(
                    root_pos=qpos[:3],
                    root_rot=qpos[3:7],
                    dof_pos=qpos[7:],
                    human_motion_data=retargeter.scaled_human_data,
                    human_pos_offset=np.array([0.0, 0.0, 0.0]),
                    show_human_body_name=False,
                    robot_frames=retargeter.ik_match_table1.keys(),
                    show_robot_body_name=False,
                    rate_limit=args.rate_limit,
                )

            qpos_list.append(qpos)
            qvel_list.append(qvel)
            stance_feet_list.append(set(retargeter.get_ground_report().get("stance_feet", [])))
            frame_dt_list.append(float(dt_this))
            pbar.update(1)

            if args.loop:
                frame_index = (frame_index + 1) % len(motion_frames)
            else:
                frame_index += 1
                if frame_index >= len(motion_frames):
                    break
    finally:
        pbar.close()
        try:
            if robot_motion_viewer is not None and getattr(robot_motion_viewer, "stop_recording", None) and args.record_video:
                robot_motion_viewer.stop_recording()
        except Exception as exc:
            print(f"[WARN] stop_recording failed: {exc}")
        try:
            if robot_motion_viewer is not None and getattr(robot_motion_viewer, "close", None):
                robot_motion_viewer.close()
            elif robot_motion_viewer is not None and getattr(robot_motion_viewer, "destroy", None):
                robot_motion_viewer.destroy()
        except Exception as exc:
            print(f"[WARN] viewer close failed: {exc}")
        time.sleep(0.05)

    print(f"[ground] stats: {retargeter.ground_stats}")
    post_processed = False
    if qpos_list and not args.disable_sequence_foot_flattening:
        qpos_arr = retargeter.flatten_feet_sequence(
            qpos_list,
            stance_feet_seq=stance_feet_list,
            contact_pre_roll=args.sequence_foot_flatten_contact_pre_roll,
            contact_post_roll=args.sequence_foot_flatten_contact_post_roll,
        )
        qpos_list = [qpos.copy() for qpos in qpos_arr]
        post_processed = True
        print(
            "[ground] applied sequence foot flattening: "
            f"pre_roll={args.sequence_foot_flatten_contact_pre_roll}, "
            f"post_roll={args.sequence_foot_flatten_contact_post_roll}"
        )
    if qpos_list and not args.disable_sequence_ground_smoothing:
        qpos_arr = retargeter.smooth_root_z_sequence(
            qpos_list,
            stance_feet_seq=stance_feet_list,
            window=args.sequence_ground_window,
            passes=args.sequence_ground_passes,
            max_step_down=args.sequence_ground_max_step_down,
            max_step_up=args.sequence_ground_max_step_up,
            contact_pre_roll=args.sequence_ground_contact_pre_roll,
            contact_post_roll=args.sequence_ground_contact_post_roll,
        )
        qvel_arr = retargeter.recompute_qvel_sequence(qpos_arr, frame_dt_list)
        qpos_list = [qpos.copy() for qpos in qpos_arr]
        qvel_list = [qvel.copy() for qvel in qvel_arr]
        post_processed = False
        print(
            "[ground] applied sequence root-z smoothing: "
            f"window={args.sequence_ground_window}, passes={args.sequence_ground_passes}, "
            f"max_step_down={args.sequence_ground_max_step_down}"
        )
    if post_processed:
        qvel_arr = retargeter.recompute_qvel_sequence(qpos_list, frame_dt_list)
        qvel_list = [qvel.copy() for qvel in qvel_arr]
    if args.save_path is not None:
        _export_motion(args, qpos_list, qvel_list, frame_dt_list, aligned_fps)


if __name__ == "__main__":
    main()
