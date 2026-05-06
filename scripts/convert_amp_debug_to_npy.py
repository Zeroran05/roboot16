#!/usr/bin/env python3
"""Convert Roboot16 AMP debug pickle files into amp-rsl-rl .npy datasets."""

from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path
from typing import Any

import numpy as np
import numpy.core
import numpy.core.multiarray
import numpy.core.numeric


DEFAULT_JOINTS = [
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


def register_numpy_compat_aliases() -> None:
    sys.modules.setdefault("numpy._core", numpy.core)
    sys.modules.setdefault("numpy._core.multiarray", numpy.core.multiarray)
    sys.modules.setdefault("numpy._core.numeric", numpy.core.numeric)


class NumpyCompatUnpickler(pickle.Unpickler):
    MODULE_ALIASES = {
        "numpy._core": "numpy.core",
        "numpy._core.multiarray": "numpy.core.multiarray",
        "numpy._core.numeric": "numpy.core.numeric",
    }

    def find_class(self, module: str, name: str) -> Any:
        module = self.MODULE_ALIASES.get(module, module)
        return super().find_class(module, name)


def require_array(data: dict[str, Any], key: str, width: int) -> np.ndarray:
    array = np.asarray(data[key], dtype=np.float32)
    if array.ndim != 2 or array.shape[1] != width:
        raise ValueError(f"Expected '{key}' to have shape (T, {width}), got {array.shape}.")
    return array


def optional_array(data: dict[str, Any], key: str, width: int | None = None) -> np.ndarray | None:
    value = data.get(key)
    if value is None:
        return None
    array = np.asarray(value, dtype=np.float32)
    if array.ndim != 2:
        raise ValueError(f"Expected optional key '{key}' to be 2D, got {array.shape}.")
    if width is not None and array.shape[1] != width:
        raise ValueError(f"Expected optional key '{key}' to have shape (T, {width}), got {array.shape}.")
    return array


def load_pickle(path: Path) -> dict[str, Any]:
    with path.open("rb") as file:
        data = NumpyCompatUnpickler(file).load()
    if not isinstance(data, dict):
        raise TypeError(f"Expected dict payload in {path}, got {type(data).__name__}.")
    return data


def build_amp_payload(data: dict[str, Any], joints_list: list[str], extra_body_names: list[str]) -> dict[str, Any]:
    joint_positions = require_array(data, "dof_pos", len(joints_list))
    root_position = require_array(data, "root_pos", 3)
    root_quaternion = require_array(data, "root_rot", 4)
    joint_velocities = optional_array(data, "dof_vel", len(joints_list))
    base_lin_velocities_local = optional_array(data, "root_vel_body", 3)
    base_ang_velocities_local = optional_array(data, "root_rot_vel", 3)

    frame_count = joint_positions.shape[0]
    fps = float(data["fps"])

    payload = {
        "joints_list": joints_list,
        "joint_positions": [frame.copy() for frame in joint_positions],
        "root_position": [frame.copy() for frame in root_position],
        "root_quaternion": [frame.copy() for frame in root_quaternion],
        "fps": fps,
    }
    if joint_velocities is not None:
        payload["joint_velocities"] = [frame.copy() for frame in joint_velocities]
        payload["amp_joint_velocities"] = [frame.copy() for frame in joint_velocities]
    if base_lin_velocities_local is not None:
        payload["base_lin_velocities_local"] = [frame.copy() for frame in base_lin_velocities_local]
        payload["amp_base_lin_velocities_local"] = [frame.copy() for frame in base_lin_velocities_local]
    if base_ang_velocities_local is not None:
        payload["base_ang_velocities_local"] = [frame.copy() for frame in base_ang_velocities_local]
        payload["amp_base_ang_velocities_local"] = [frame.copy() for frame in base_ang_velocities_local]

    local_body_pos = data.get("local_body_pos")
    link_body_list = data.get("link_body_list")
    if local_body_pos is not None and link_body_list is not None and extra_body_names:
        body_positions = np.asarray(local_body_pos, dtype=np.float32)
        link_names = list(link_body_list)
        selected = []
        amp_extra_names = []
        for body_name in extra_body_names:
            if body_name not in link_names:
                raise KeyError(f"Missing body '{body_name}' in link_body_list.")
            selected.append(link_names.index(body_name))
            amp_extra_names.extend(f"{body_name}_{axis}" for axis in ("x", "y", "z"))
        amp_extra = body_positions[:, selected, :].reshape(frame_count, -1)
        payload["amp_extra_observations"] = [frame.copy() for frame in amp_extra]
        payload["amp_extra_names"] = amp_extra_names

    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_pkl", type=Path, help="Input AMP debug pickle.")
    parser.add_argument("output_npy", type=Path, help="Output .npy path.")
    parser.add_argument(
        "--extra-body-names",
        nargs="*",
        default=["left_ankle_roll_link", "right_ankle_roll_link"],
        help="Body names to export as AMP extra observations.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    register_numpy_compat_aliases()
    payload = build_amp_payload(load_pickle(args.input_pkl), list(DEFAULT_JOINTS), list(args.extra_body_names))
    args.output_npy.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.output_npy, payload, allow_pickle=True)
    print(f"Converted {args.input_pkl} -> {args.output_npy}")


if __name__ == "__main__":
    main()
