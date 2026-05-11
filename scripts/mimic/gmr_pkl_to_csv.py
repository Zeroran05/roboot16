#!/usr/bin/env python3

"""Convert a GMR robot motion pickle into the CSV layout expected by mimic preprocessing.

CSV layout per frame:
    root_pos(xyz), root_quat(xyzw), dof_pos(...)
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import numpy as np


DEFAULT_DOF_ORDER = [
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


def _resolve_reorder_indices(dof_names: list[str], target_order: list[str]) -> list[int]:
    index_map = {name: idx for idx, name in enumerate(dof_names)}
    missing = [name for name in target_order if name not in index_map]
    if missing:
        raise KeyError(f"Missing joints in motion data: {missing}")
    return [index_map[name] for name in target_order]


def convert_pkl_to_csv(input_pkl: Path, output_csv: Path, target_fps: float | None) -> None:
    with input_pkl.open("rb") as f:
        motion_data = pickle.load(f)

    root_pos = np.asarray(motion_data["root_pos"], dtype=np.float32)
    root_rot = np.asarray(motion_data["root_rot"], dtype=np.float32)  # xyzw
    dof_pos = np.asarray(motion_data["dof_pos"], dtype=np.float32)
    fps = float(motion_data["fps"])

    dof_names = list(motion_data.get("dof_names") or DEFAULT_DOF_ORDER)
    reorder_indices = _resolve_reorder_indices(dof_names, DEFAULT_DOF_ORDER)
    dof_pos = dof_pos[:, reorder_indices]

    motion = np.concatenate([root_pos, root_rot, dof_pos], axis=1).astype(np.float32)

    if target_fps is not None and abs(fps - target_fps) > 1e-6:
        if target_fps <= 0:
            raise ValueError("--target_fps must be positive")
        downsample_factor = fps / target_fps
        indices = np.arange(0, motion.shape[0], downsample_factor).astype(int)
        motion = motion[np.clip(indices, 0, motion.shape[0] - 1)]
        fps = target_fps

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(output_csv, motion, delimiter=",")
    print(f"[INFO] Saved CSV: {output_csv}")
    print(f"[INFO] Frames: {motion.shape[0]}, Columns: {motion.shape[1]}, FPS tag used downstream: {fps}")


def main():
    parser = argparse.ArgumentParser(description="Convert one GMR pickle to mimic CSV.")
    parser.add_argument("--input_pkl", type=Path, required=True, help="Input GMR motion pickle.")
    parser.add_argument("--output_csv", type=Path, required=True, help="Output CSV file path.")
    parser.add_argument("--target_fps", type=float, default=None, help="Optional downsample target FPS.")
    args = parser.parse_args()

    convert_pkl_to_csv(args.input_pkl, args.output_csv, args.target_fps)


if __name__ == "__main__":
    main()
