#!/usr/bin/env python3

''' 
txt回放；固定root；
./isaaclab.sh -p roboot16_amp_project/scripts/visualize_amp_replay_isaacsim.py \
--txt /root/isaaclab/roboot16_amp_project/data/amp_expert/run1_1.txt

结合txt和pkl回放；root使用pkl轨迹，关节和脚位置使用txt数据；验证两者的对齐情况；
./isaaclab.sh -p roboot16_amp_project/scripts/visualize_amp_replay_isaacsim.py \
--txt /root/isaaclab/roboot16_amp_project/data/amp_expert/run1_2.txt \
--pkl /root/isaaclab/roboot16_amp_project/data/segments/run1_2.pkl \
--loop \
--playback-speed 0.3

'''
from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
import traceback
from dataclasses import dataclass
from pathlib import Path

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(
    description="Replay AMP txt/pkl data in Isaac Sim and validate the reordered observations."
)
parser.add_argument("--txt", type=Path, default=None, help="Path to the TianKung-style 30D AMP txt/json file.")
parser.add_argument("--pkl", type=Path, default=None, help="Path to the source PKL motion file.")
parser.add_argument(
    "--play-source",
    type=str,
    choices=("txt", "pkl"),
    default=None,
    help="Which source to replay when both --txt and --pkl are provided. Defaults to txt if available.",
)
parser.add_argument("--start-frame", type=int, default=0, help="Starting frame index.")
parser.add_argument("--max-frames", type=int, default=0, help="Maximum number of frames to replay. 0 means all.")
parser.add_argument("--frame-stride", type=int, default=1, help="Advance this many source frames per rendered frame.")
parser.add_argument("--playback-speed", type=float, default=1.0, help="Playback speed multiplier.")
parser.add_argument("--report-every", type=int, default=10, help="Print one validation report every N replayed frames.")
parser.add_argument("--loop", action="store_true", help="Loop the clip continuously.")
parser.add_argument("--compare-only", action="store_true", help="Only print summary stats, do not open playback loop.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ISAACLAB_ROOT = PROJECT_ROOT.parent
SRC_DIR = PROJECT_ROOT / "src"
for path in (SRC_DIR, ISAACLAB_ROOT / "source", ISAACLAB_ROOT / "source" / "isaaclab_tasks"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import numpy as np
import torch
import numpy.core as numpy_core
import numpy.core.multiarray as numpy_core_multiarray

sys.modules.setdefault("numpy._core", numpy_core)
sys.modules.setdefault("numpy._core.multiarray", numpy_core_multiarray)

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.sim import SimulationContext
from isaaclab.utils.math import quat_apply_inverse

from roboot16_amp_project.assets.robots import ROBOOT_CFG
from isaaclab_tasks.manager_based.locomotion.velocity.config.roboot16_amp.flat_env_cfg import (
    AMP_FOOT_BODY_NAMES,
    AMP_JOINT_NAMES,
)


MUJOCO_DOF_NAMES = [
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


@dataclass
class ReplayClip:
    source_name: str
    frame_duration: float
    joint_positions: np.ndarray
    joint_velocities: np.ndarray
    foot_positions_local: np.ndarray | None
    root_pose_wxyz: np.ndarray | None

    def __len__(self) -> int:
        return int(self.joint_positions.shape[0])


def _reorder_block(block: np.ndarray, dataset_joint_names: list[str], target_joint_names: list[str]) -> np.ndarray:
    idx_map: list[int] = []
    for joint_name in target_joint_names:
        if joint_name not in dataset_joint_names:
            raise KeyError(f"Joint '{joint_name}' is missing from dataset ordering: {dataset_joint_names}")
        idx_map.append(dataset_joint_names.index(joint_name))
    return block[:, idx_map]


def load_txt_clip(txt_path: Path, target_joint_names: list[str]) -> ReplayClip:
    payload = json.loads(txt_path.read_text(encoding="utf-8"))
    frame_layout = payload["FrameLayout"]
    frame_layout_dims = payload["FrameLayoutDims"]
    joint_order = payload["JointOrder"]
    frames = np.asarray(payload["Frames"], dtype=np.float32)

    cursor = 0
    blocks: dict[str, np.ndarray] = {}
    for name, width in zip(frame_layout, frame_layout_dims):
        width = int(width)
        blocks[name] = frames[:, cursor : cursor + width]
        cursor += width

    dataset_joint_names = list(joint_order["right_leg_q"]) + list(joint_order["left_leg_q"])
    joint_positions = np.concatenate([blocks["right_leg_q"], blocks["left_leg_q"]], axis=1)
    joint_velocities = np.concatenate([blocks["right_leg_dq"], blocks["left_leg_dq"]], axis=1)
    foot_positions_local = np.concatenate(
        [blocks["left_foot_pos_local"], blocks["right_foot_pos_local"]],
        axis=1,
    )

    joint_positions = _reorder_block(joint_positions, dataset_joint_names, target_joint_names)
    joint_velocities = _reorder_block(joint_velocities, dataset_joint_names, target_joint_names)

    return ReplayClip(
        source_name=f"txt:{txt_path.name}",
        frame_duration=float(payload["FrameDuration"]),
        joint_positions=joint_positions,
        joint_velocities=joint_velocities,
        foot_positions_local=foot_positions_local,
        root_pose_wxyz=None,
    )


def load_pkl_clip(pkl_path: Path, target_joint_names: list[str]) -> ReplayClip:
    with pkl_path.open("rb") as f:
        motion_data = pickle.load(f)

    dataset_joint_names = list(motion_data.get("dof_names", MUJOCO_DOF_NAMES))
    joint_positions = np.asarray(motion_data["dof_pos"], dtype=np.float32)
    joint_velocities = motion_data.get("dof_vel")
    if joint_velocities is None:
        dt = 1.0 / float(motion_data["fps"])
        joint_velocities = np.zeros_like(joint_positions, dtype=np.float32)
        if len(joint_positions) > 1:
            joint_velocities[:-1] = (joint_positions[1:] - joint_positions[:-1]) / dt
            joint_velocities[-1] = joint_velocities[-2]
    else:
        joint_velocities = np.asarray(joint_velocities, dtype=np.float32)

    joint_positions = _reorder_block(joint_positions, dataset_joint_names, target_joint_names)
    joint_velocities = _reorder_block(joint_velocities, dataset_joint_names, target_joint_names)

    root_pose_wxyz = None
    if "root_pos" in motion_data and "root_rot" in motion_data:
        root_pos = np.asarray(motion_data["root_pos"], dtype=np.float32)
        root_rot_xyzw = np.asarray(motion_data["root_rot"], dtype=np.float32)
        root_rot_wxyz = root_rot_xyzw[:, [3, 0, 1, 2]]
        root_pose_wxyz = np.concatenate([root_pos, root_rot_wxyz], axis=1)

    return ReplayClip(
        source_name=f"pkl:{pkl_path.name}",
        frame_duration=1.0 / float(motion_data["fps"]),
        joint_positions=joint_positions,
        joint_velocities=joint_velocities,
        foot_positions_local=None,
        root_pose_wxyz=root_pose_wxyz,
    )


def compute_root_local_feet(robot: Articulation, foot_body_ids: list[int]) -> torch.Tensor:
    body_pos_w = robot.data.body_pos_w[:, foot_body_ids, :]
    root_pos_w = robot.data.root_pos_w.unsqueeze(1)
    root_quat_w = robot.data.root_quat_w.unsqueeze(1).expand(-1, len(foot_body_ids), -1)
    foot_pos_b = quat_apply_inverse(root_quat_w, body_pos_w - root_pos_w)
    return foot_pos_b.reshape(foot_pos_b.shape[0], -1)


def report_clip_alignment(txt_clip: ReplayClip | None, pkl_clip: ReplayClip | None) -> None:
    if txt_clip is None or pkl_clip is None:
        return
    common = min(len(txt_clip), len(pkl_clip))
    if common <= 0:
        print("[WARN] txt/pkl overlap is empty; skipping clip-to-clip comparison.", flush=True)
        return
    joint_pos_diff = np.abs(txt_clip.joint_positions[:common] - pkl_clip.joint_positions[:common])
    joint_vel_diff = np.abs(txt_clip.joint_velocities[:common] - pkl_clip.joint_velocities[:common])
    max_q = float(joint_pos_diff.max())
    max_dq = float(joint_vel_diff.max())
    mean_q = float(joint_pos_diff.mean())
    mean_dq = float(joint_vel_diff.mean())
    print("[INFO] txt vs pkl reordered comparison", flush=True)
    print(f"       frames={common} | max|dqpos|={max_q:.6f} | mean|dqpos|={mean_q:.6f}", flush=True)
    print(f"       frames={common} | max|dvel| ={max_dq:.6f} | mean|dvel| ={mean_dq:.6f}", flush=True)


def main() -> None:
    if args_cli.txt is None and args_cli.pkl is None:
        raise ValueError("At least one of --txt or --pkl must be provided.")

    txt_clip = load_txt_clip(args_cli.txt, AMP_JOINT_NAMES) if args_cli.txt is not None else None
    pkl_clip = load_pkl_clip(args_cli.pkl, AMP_JOINT_NAMES) if args_cli.pkl is not None else None
    report_clip_alignment(txt_clip, pkl_clip)

    play_source = args_cli.play_source
    if play_source is None:
        play_source = "txt" if txt_clip is not None else "pkl"

    clip = txt_clip if play_source == "txt" else pkl_clip
    if clip is None:
        raise ValueError(f"Requested --play-source {play_source!r}, but that input was not provided.")

    # If both sources are available, prefer the PKL root trajectory so the robot moves in
    # world coordinates while still validating the AMP joint/foot quantities from the txt clip.
    root_motion_clip = pkl_clip if pkl_clip is not None and pkl_clip.root_pose_wxyz is not None else clip

    replay_frame_count = len(clip)
    if root_motion_clip.root_pose_wxyz is not None:
        replay_frame_count = min(replay_frame_count, len(root_motion_clip.root_pose_wxyz))
    if replay_frame_count <= 0:
        raise ValueError("Replay frame count resolved to zero.")

    print(f"[INFO] Replaying source: {clip.source_name}", flush=True)
    print(
        f"[INFO] Replay frames: {replay_frame_count} "
        f"| source_frames={len(clip)} | frame_duration={clip.frame_duration:.6f}s",
        flush=True,
    )
    if pkl_clip is not None and pkl_clip.root_pose_wxyz is not None:
        print("[INFO] Root motion source: pkl root trajectory", flush=True)
    else:
        print("[INFO] Root motion source: fixed default root pose", flush=True)

    if args_cli.compare_only:
        return

    sim = SimulationContext(sim_utils.SimulationCfg(dt=1.0 / 120.0))
    sim.set_camera_view(eye=[2.8, 2.2, 1.8], target=[0.0, 0.0, 0.9])

    ground_cfg = sim_utils.GroundPlaneCfg()
    ground_cfg.func("/World/defaultGroundPlane", ground_cfg)
    light_cfg = sim_utils.DomeLightCfg(intensity=2500.0, color=(0.75, 0.75, 0.75))
    light_cfg.func("/World/Light", light_cfg)

    robot_cfg = ROBOOT_CFG.replace(prim_path="/World/Robot")
    if robot_cfg.spawn.rigid_props is not None:
        robot_cfg.spawn.rigid_props.disable_gravity = True
    robot_cfg.spawn.func("/World/Robot", robot_cfg.spawn)
    robot = Articulation(robot_cfg)

    sim.reset()
    sim_dt = sim.get_physics_dt()
    device = robot.device

    joint_ids, _ = robot.find_joints(AMP_JOINT_NAMES)
    foot_body_ids, _ = robot.find_bodies(AMP_FOOT_BODY_NAMES)
    default_root_state = robot.data.default_root_state.clone()
    zero_root_velocity = torch.zeros((1, 6), device=device)

    total_frames = replay_frame_count
    frame_idx = max(0, args_cli.start_frame)
    if frame_idx >= total_frames:
        raise ValueError(f"--start-frame {frame_idx} is out of range for clip length {total_frames}.")

    replayed = 0
    max_joint_pos_err = 0.0
    max_joint_vel_err = 0.0
    max_foot_err = 0.0

    while simulation_app.is_running():
        if args_cli.max_frames > 0 and replayed >= args_cli.max_frames:
            break

        if frame_idx >= total_frames:
            if not args_cli.loop:
                break
            frame_idx = 0

        frame_start = time.time()

        joint_pos = torch.tensor(clip.joint_positions[frame_idx], dtype=torch.float32, device=device).unsqueeze(0)
        joint_vel = torch.tensor(clip.joint_velocities[frame_idx], dtype=torch.float32, device=device).unsqueeze(0)

        if root_motion_clip.root_pose_wxyz is not None:
            root_pose = torch.tensor(
                root_motion_clip.root_pose_wxyz[frame_idx], dtype=torch.float32, device=device
            ).unsqueeze(0)
        else:
            root_pose = default_root_state[:, :7].clone()

        robot.write_root_pose_to_sim(root_pose)
        robot.write_root_velocity_to_sim(zero_root_velocity)
        robot.write_joint_state_to_sim(joint_pos, joint_vel)
        robot.update(0.0)

        observed_joint_pos = robot.data.joint_pos[:, joint_ids]
        observed_joint_vel = robot.data.joint_vel[:, joint_ids]
        observed_foot_pos = compute_root_local_feet(robot, foot_body_ids)

        q_err = torch.max(torch.abs(observed_joint_pos - joint_pos)).item()
        dq_err = torch.max(torch.abs(observed_joint_vel - joint_vel)).item()
        foot_err = 0.0
        if clip.foot_positions_local is not None:
            target_foot = torch.tensor(clip.foot_positions_local[frame_idx], dtype=torch.float32, device=device).unsqueeze(0)
            foot_err = torch.max(torch.abs(observed_foot_pos - target_foot)).item()

        max_joint_pos_err = max(max_joint_pos_err, q_err)
        max_joint_vel_err = max(max_joint_vel_err, dq_err)
        max_foot_err = max(max_foot_err, foot_err)

        if replayed % max(1, args_cli.report_every) == 0:
            msg = (
                f"[frame {frame_idx:05d}] "
                f"max|joint_pos_err|={q_err:.6e} "
                f"max|joint_vel_err|={dq_err:.6e}"
            )
            if clip.foot_positions_local is not None:
                msg += f" max|foot_local_err|={foot_err:.6e}"
            print(msg, flush=True)

        sim.step(render=True)
        robot.update(sim_dt)

        replayed += 1
        frame_idx += max(1, args_cli.frame_stride)

        sleep_s = (clip.frame_duration / max(args_cli.playback_speed, 1e-6)) - (time.time() - frame_start)
        if sleep_s > 0:
            time.sleep(sleep_s)

    print("[INFO] Replay finished.", flush=True)
    print(f"       max|joint_pos_err| = {max_joint_pos_err:.6e}", flush=True)
    print(f"       max|joint_vel_err| = {max_joint_vel_err:.6e}", flush=True)
    if clip.foot_positions_local is not None:
        print(f"       max|foot_local_err| = {max_foot_err:.6e}", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"[ERROR] visualize_amp_replay_isaacsim failed: {exc}", flush=True)
        traceback.print_exc()
        raise
    finally:
        simulation_app.close()
