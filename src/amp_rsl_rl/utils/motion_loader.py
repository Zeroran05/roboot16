# Copyright (c) 2025, Istituto Italiano di Tecnologia
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from pathlib import Path
from typing import List, Union, Tuple, Generator, Dict
from dataclasses import dataclass
import sys
import re

import torch
import numpy as np
from scipy.spatial.transform import Rotation, Slerp
from scipy.interpolate import interp1d

from .motion_txt import load_motion_txt, resolve_motion_dataset_path


_SPEED_TAG_RE = re.compile(r"^(stand|walk|run|sprint)\d+_(\d+p\d+)(?:_mirror(?:ed)?)?$")


def _infer_dataset_speed(dataset_name: str) -> tuple[str, float]:
    """Infer (group, speed) from a categorized dataset name like ``walk3_1p7_mirror``."""
    stem = Path(dataset_name).stem
    match = _SPEED_TAG_RE.match(stem)
    if match is None:
        raise ValueError(
            f"Unable to infer speed metadata from dataset name '{dataset_name}'. "
            "Expected names like 'walk3_1p7' or 'sprint2_3p2_mirror'."
        )
    group = match.group(1)
    speed = float(match.group(2).replace("p", "."))
    return group, speed


def _load_pickled_numpy_dict(dataset_path: Path) -> dict:
    """Load a pickled numpy dict across NumPy 1.x/2.x internal module renames."""
    try:
        return np.load(str(dataset_path), allow_pickle=True).item()
    except ModuleNotFoundError as exc:
        # Some datasets were serialized with NumPy 2.x and reference `numpy._core.*`.
        # Isaac Sim often ships an older NumPy that still exposes the same internals
        # through `numpy.core.*`, so we register a temporary import alias.
        if exc.name != "numpy._core":
            raise

        import numpy.core as numpy_core
        import numpy.core.multiarray as numpy_core_multiarray

        sys.modules.setdefault("numpy._core", numpy_core)
        sys.modules.setdefault("numpy._core.multiarray", numpy_core_multiarray)
        return np.load(str(dataset_path), allow_pickle=True).item()


def _as_optional_array(data: dict, key: str, width: int | None = None) -> np.ndarray | None:
    """Load an optional 2D array field from a serialized motion payload."""
    if key not in data or data[key] is None:
        return None
    array = np.asarray(data[key], dtype=np.float32)
    if array.ndim != 2:
        raise ValueError(f"Expected optional field '{key}' to be 2D, got shape {array.shape}.")
    if width is not None and array.shape[1] != width:
        raise ValueError(f"Expected optional field '{key}' to have width {width}, got shape {array.shape}.")
    return array


def _identity_quat_wxyz(num_frames: int) -> np.ndarray:
    """Return an identity quaternion sequence in IsaacLab's wxyz convention."""
    quat = np.zeros((num_frames, 4), dtype=np.float32)
    quat[:, 0] = 1.0
    return quat


def download_amp_dataset_from_hf(
    destination_dir: Path,
    robot_folder: str,
    files: list,
    repo_id: str = "ami-iit/amp-dataset",
) -> list:
    """
    Downloads AMP dataset files from Hugging Face and saves them to `destination_dir`.
    Ensures real file copies (not symlinks or hard links).

    Args:
        destination_dir (Path): Local directory to save the files.
        robot_folder (str): Folder in the Hugging Face dataset repo to pull from.
        files (list): List of filenames to download.
        repo_id (str): Hugging Face repository ID. Default is "ami-iit/amp-dataset".

    Returns:
        List[str]: List of dataset names (without .npy extension).
    """
    from huggingface_hub import hf_hub_download

    destination_dir.mkdir(parents=True, exist_ok=True)
    dataset_names = []

    for file in files:
        file_path = hf_hub_download(
            repo_id=repo_id,
            filename=f"{robot_folder}/{file}",
            repo_type="dataset",
            local_files_only=False,
        )
        local_copy = destination_dir / file
        # Deep copy to avoid symlinks
        with open(file_path, "rb") as src_file, open(local_copy, "wb") as dst_file:
            dst_file.write(src_file.read())
        dataset_names.append(file.replace(".npy", ""))

    return dataset_names


@dataclass
class MotionData:
    """
    Data class representing motion data for humanoid agents.

    This class stores joint positions and velocities, base velocities (both in local
    and mixed/world frames), and base orientation (as quaternion). It offers utilities
    for preparing data in AMP-compatible format, as well as environment reset states.

    Attributes:
        - joint_positions: shape (T, N)
        - joint_velocities: shape (T, N)
        - base_lin_velocities_mixed: linear velocity in world frame
        - base_ang_velocities_mixed: (currently zeros)
        - base_lin_velocities_local: linear velocity in local (body) frame
        - base_ang_velocities_local: (currently zeros)
        - base_quat: orientation quaternion as torch.Tensor in wxyz order

    Notes:
        - The quaternion is expected in the dataset as `xyzw` format (SciPy default),
          and it is converted internally to `wxyz` format to be compatible with IsaacLab conventions.
        - All data is converted to torch.Tensor on the specified device during initialization.
    """

    joint_positions: Union[torch.Tensor, np.ndarray]
    joint_velocities: Union[torch.Tensor, np.ndarray]
    base_lin_velocities_mixed: Union[torch.Tensor, np.ndarray]
    base_ang_velocities_mixed: Union[torch.Tensor, np.ndarray]
    base_lin_velocities_local: Union[torch.Tensor, np.ndarray]
    base_ang_velocities_local: Union[torch.Tensor, np.ndarray]
    base_quat: Union[Rotation, torch.Tensor]
    amp_joint_positions: Union[torch.Tensor, np.ndarray, None] = None
    amp_joint_velocities: Union[torch.Tensor, np.ndarray, None] = None
    amp_base_lin_velocities_local: Union[torch.Tensor, np.ndarray, None] = None
    amp_base_ang_velocities_local: Union[torch.Tensor, np.ndarray, None] = None
    amp_extra_observations: Union[torch.Tensor, np.ndarray, None] = None
    device: torch.device = torch.device("cpu")

    def __post_init__(self) -> None:
        # Convert numpy arrays (or SciPy Rotations) to torch tensors
        def to_tensor(x):
            return torch.tensor(x, device=self.device, dtype=torch.float32)

        if isinstance(self.joint_positions, np.ndarray):
            self.joint_positions = to_tensor(self.joint_positions)
        if isinstance(self.joint_velocities, np.ndarray):
            self.joint_velocities = to_tensor(self.joint_velocities)
        if isinstance(self.base_lin_velocities_mixed, np.ndarray):
            self.base_lin_velocities_mixed = to_tensor(self.base_lin_velocities_mixed)
        if isinstance(self.base_ang_velocities_mixed, np.ndarray):
            self.base_ang_velocities_mixed = to_tensor(self.base_ang_velocities_mixed)
        if isinstance(self.base_lin_velocities_local, np.ndarray):
            self.base_lin_velocities_local = to_tensor(self.base_lin_velocities_local)
        if isinstance(self.base_ang_velocities_local, np.ndarray):
            self.base_ang_velocities_local = to_tensor(self.base_ang_velocities_local)
        if isinstance(self.base_quat, np.ndarray):
            self.base_quat = to_tensor(self.base_quat)
        elif isinstance(self.base_quat, Rotation):
            quat_xyzw = self.base_quat.as_quat()  # (T,4) xyzw
            # convert to wxyz
            self.base_quat = torch.tensor(
                quat_xyzw[:, [3, 0, 1, 2]],
                device=self.device,
                dtype=torch.float32,
            )
        if isinstance(self.amp_joint_positions, np.ndarray):
            self.amp_joint_positions = to_tensor(self.amp_joint_positions)
        if isinstance(self.amp_joint_velocities, np.ndarray):
            self.amp_joint_velocities = to_tensor(self.amp_joint_velocities)
        if isinstance(self.amp_base_lin_velocities_local, np.ndarray):
            self.amp_base_lin_velocities_local = to_tensor(self.amp_base_lin_velocities_local)
        if isinstance(self.amp_base_ang_velocities_local, np.ndarray):
            self.amp_base_ang_velocities_local = to_tensor(self.amp_base_ang_velocities_local)
        if isinstance(self.amp_extra_observations, np.ndarray):
            self.amp_extra_observations = to_tensor(self.amp_extra_observations)

        if self.amp_joint_positions is None:
            self.amp_joint_positions = self.joint_positions
        if self.amp_joint_velocities is None:
            self.amp_joint_velocities = self.joint_velocities
        if self.amp_base_lin_velocities_local is None:
            self.amp_base_lin_velocities_local = self.base_lin_velocities_local
        if self.amp_base_ang_velocities_local is None:
            self.amp_base_ang_velocities_local = self.base_ang_velocities_local

    def __len__(self) -> int:
        return self.joint_positions.shape[0]

    def get_amp_dataset_obs(self, indices: torch.Tensor) -> torch.Tensor:
        """
        Returns the AMP observation tensor for given indices.

        Args:
            indices: indices of samples to retrieve

        Returns:
            Concatenated observation tensor
        """
        parts = [
            self.amp_joint_positions[indices],
            self.amp_joint_velocities[indices],
            self.amp_base_lin_velocities_local[indices],
            self.amp_base_ang_velocities_local[indices],
        ]
        if self.amp_extra_observations is not None:
            parts.append(self.amp_extra_observations[indices])
        return torch.cat(parts, dim=1)

    def get_state_for_reset(self, indices: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        """
        Returns the full state needed for environment reset.

        Args:
            indices: indices of samples to retrieve

        Returns:
            Tuple of (quat, joint_positions, joint_velocities, base_lin_velocities, base_ang_velocities)
        """
        return (
            self.base_quat[indices],
            self.joint_positions[indices],
            self.joint_velocities[indices],
            self.base_lin_velocities_local[indices],
            self.base_ang_velocities_local[indices],
        )

    def get_random_sample_for_reset(self, items: int = 1) -> Tuple[torch.Tensor, ...]:
        indices = torch.randint(0, len(self), (items,), device=self.device)
        return self.get_state_for_reset(indices)


class AMPLoader:
    """
    Loader and processor for humanoid motion capture datasets in AMP format.

    Responsibilities:
      - Loading `.npy` raw motion files or precomputed `.txt/.json` AMP expert files
      - Building a unified joint ordering across all datasets
      - Resampling trajectories to match the simulator's timestep
      - Computing derived quantities (velocities, local-frame motion)
      - Returning torch-friendly MotionData instances

    Dataset format:
        Raw `.npy` format:
          - "joints_list": List[str]
          - "joint_positions": List[np.ndarray]
          - "root_position": List[np.ndarray]
          - "root_quaternion": List[np.ndarray] (xyzw)
          - "fps": float (frames/sec)

        Precomputed `.txt/.json` expert format:
          - TianKung-style 30D AMP frames stored as JSON `.txt/.json`
          - Required metadata includes:
            - "FrameLayout"
            - "FrameLayoutDims"
            - "JointOrder"
            - "FrameDuration"
            - "Frames"

    Args:
        device: Target torch device ('cpu' or 'cuda')
        dataset_path_root: Directory containing the .npy motion files
        datasets: Dictionary mapping dataset names (without extension) to sampling weights (floats)
        simulation_dt: Timestep used by the simulator
        slow_down_factor: Integer factor to slow down original data
        expected_joint_names: (Optional) override for joint ordering
    """

    def __init__(
        self,
        device: str,
        dataset_path_root: Path,
        datasets: Dict[str, float],
        simulation_dt: float,
        slow_down_factor: int,
        expected_joint_names: Union[List[str], None] = None,
        speed_conditioning_tau: float = 0.35,
        stand_only_speed_threshold: float = 0.1,
    ) -> None:
        self.device = device
        self.speed_conditioning_tau = float(speed_conditioning_tau)
        self.stand_only_speed_threshold = float(stand_only_speed_threshold)
        if isinstance(dataset_path_root, str):
            dataset_path_root = Path(dataset_path_root)

        # ─── Parse dataset names and weights ───
        dataset_names = list(datasets.keys())
        dataset_weights = list(datasets.values())

        # ─── Build union of all joint names if not provided ───
        if expected_joint_names is None:
            joint_union: List[str] = []
            seen = set()
            for name in dataset_names:
                dataset_path = resolve_motion_dataset_path(dataset_path_root, name)
                if dataset_path.suffix == ".npy":
                    info = _load_pickled_numpy_dict(dataset_path)
                    joint_names = info["joints_list"]
                else:
                    info = load_motion_txt(dataset_path)
                    joint_order = info.get("JointOrder", {})
                    joint_names = list(joint_order.get("right_leg_q", [])) + list(joint_order.get("left_leg_q", []))
                    if not joint_names:
                        raise ValueError(
                            f"Unsupported expert txt format in '{dataset_path}'. "
                            "Expected TianKung-style JointOrder metadata."
                        )
                for j in joint_names:
                    if j not in seen:
                        seen.add(j)
                        joint_union.append(j)
            expected_joint_names = joint_union
        # ─────────────────────────────────────────────────────────

        # Load and process each dataset into MotionData
        self.motion_data: List[MotionData] = []
        self.dataset_groups: List[str] = []
        self.dataset_speeds: List[float] = []
        for dataset_name in dataset_names:
            dataset_path = resolve_motion_dataset_path(dataset_path_root, dataset_name)
            dataset_group, dataset_speed = _infer_dataset_speed(dataset_name)
            md = self.load_data(
                dataset_path,
                simulation_dt,
                slow_down_factor,
                expected_joint_names,
            )
            self.motion_data.append(md)
            self.dataset_groups.append(dataset_group)
            self.dataset_speeds.append(dataset_speed)

        # Normalize dataset-level sampling weights
        weights = torch.tensor(dataset_weights, dtype=torch.float32, device=self.device)
        self.dataset_weights = weights / weights.sum()

        # Precompute flat buffers for fast sampling
        obs_list, next_obs_list, reset_states = [], [], []
        self.clip_lengths: List[int] = []
        self.clip_start_indices: List[int] = []
        frame_cursor = 0
        for data, w in zip(self.motion_data, self.dataset_weights):
            T = len(data)
            self.clip_lengths.append(T)
            self.clip_start_indices.append(frame_cursor)
            idx = torch.arange(T, device=self.device)
            obs = data.get_amp_dataset_obs(idx)
            next_idx = torch.clamp(idx + 1, max=T - 1)
            next_obs = data.get_amp_dataset_obs(next_idx)

            obs_list.append(obs)
            next_obs_list.append(next_obs)

            quat, jp, jv, blv, bav = data.get_state_for_reset(idx)
            reset_states.append(torch.cat([quat, jp, jv, blv, bav], dim=1))
            frame_cursor += T

        self.all_obs = torch.cat(obs_list, dim=0)
        self.all_next_obs = torch.cat(next_obs_list, dim=0)
        self.all_states = torch.cat(reset_states, dim=0)

        # Build per-frame sampling weights: weight_i / length_i
        lengths = [len(d) for d in self.motion_data]
        per_frame = torch.cat(
            [
                torch.full((L,), w / L, device=self.device)
                for w, L in zip(self.dataset_weights, lengths)
            ]
        )
        self.per_frame_weights = per_frame / per_frame.sum()
        self.clip_speeds = torch.tensor(self.dataset_speeds, dtype=torch.float32, device=self.device)
        self.clip_is_stand = torch.tensor(
            [group == "stand" for group in self.dataset_groups],
            dtype=torch.bool,
            device=self.device,
        )
        self.clip_lengths_tensor = torch.tensor(self.clip_lengths, dtype=torch.long, device=self.device)
        self.clip_start_indices_tensor = torch.tensor(
            self.clip_start_indices, dtype=torch.long, device=self.device
        )

    def _resample_data_Rn(
        self,
        data: List[np.ndarray],
        original_keyframes,
        target_keyframes,
    ) -> np.ndarray:
        f = interp1d(original_keyframes, data, axis=0)
        return f(target_keyframes)

    def _resample_data_SO3(
        self,
        raw_quaternions: List[np.ndarray],
        original_keyframes,
        target_keyframes,
    ) -> Rotation:

        # the quaternion is expected in the dataset as `xyzw` format (SciPy default)
        tmp = Rotation.from_quat(raw_quaternions)
        slerp = Slerp(original_keyframes, tmp)
        return slerp(target_keyframes)

    def _compute_ang_vel(
        self,
        data: List[Rotation],
        dt: float,
        local: bool = False,
    ) -> np.ndarray:
        R_prev = data[:-1]
        R_next = data[1:]

        if local:
            # Exp = R_i⁻¹ · R_{i+1}
            rel = R_prev.inv() * R_next
        else:
            # Exp = R_{i+1} · R_i⁻¹
            rel = R_next * R_prev.inv()

        # Log-map to rotation vectors and divide by Δt
        rotvec = rel.as_rotvec() / dt

        return np.vstack((rotvec, rotvec[-1]))

    def _compute_raw_derivative(self, data: np.ndarray, dt: float) -> np.ndarray:
        d = (data[1:] - data[:-1]) / dt
        return np.vstack([d, d[-1:]])

    def load_data(
        self,
        dataset_path: Path,
        simulation_dt: float,
        slow_down_factor: int = 1,
        expected_joint_names: Union[List[str], None] = None,
    ) -> MotionData:
        """
        Loads and processes one motion dataset.

        Returns:
            MotionData instance
        """
        if dataset_path.suffix in {".txt", ".json"}:
            return self.load_expert_txt_data(dataset_path, simulation_dt, slow_down_factor, expected_joint_names)

        data = _load_pickled_numpy_dict(dataset_path)
        dataset_joint_names = data["joints_list"]

        # build index map for expected_joint_names
        idx_map: List[Union[int, None]] = []
        for j in expected_joint_names:
            if j in dataset_joint_names:
                idx_map.append(dataset_joint_names.index(j))
            else:
                idx_map.append(None)

        # reorder & fill joint positions
        jp_list: List[np.ndarray] = []
        for frame in data["joint_positions"]:
            arr = np.zeros((len(idx_map),), dtype=frame.dtype)
            for i, src_idx in enumerate(idx_map):
                if src_idx is not None:
                    arr[i] = frame[src_idx]
            jp_list.append(arr)

        joint_velocities_raw = _as_optional_array(data, "joint_velocities", len(dataset_joint_names))
        amp_joint_positions_raw = _as_optional_array(data, "amp_joint_positions", len(dataset_joint_names))
        amp_joint_velocities_raw = _as_optional_array(data, "amp_joint_velocities", len(dataset_joint_names))
        base_lin_vel_local_raw = _as_optional_array(data, "base_lin_velocities_local", 3)
        base_ang_vel_local_raw = _as_optional_array(data, "base_ang_velocities_local", 3)
        amp_base_lin_vel_local_raw = _as_optional_array(data, "amp_base_lin_velocities_local", 3)
        amp_base_ang_vel_local_raw = _as_optional_array(data, "amp_base_ang_velocities_local", 3)
        amp_extra_observations_raw = _as_optional_array(data, "amp_extra_observations")

        def reorder_joint_array(block: np.ndarray | None) -> np.ndarray | None:
            if block is None:
                return None
            out = np.zeros((block.shape[0], len(idx_map)), dtype=block.dtype)
            for target_idx, src_idx in enumerate(idx_map):
                if src_idx is not None:
                    out[:, target_idx] = block[:, src_idx]
            return out

        joint_velocities_raw = reorder_joint_array(joint_velocities_raw)
        amp_joint_positions_raw = reorder_joint_array(amp_joint_positions_raw)
        amp_joint_velocities_raw = reorder_joint_array(amp_joint_velocities_raw)

        dt = 1.0 / data["fps"] / float(slow_down_factor)
        T = len(jp_list)
        t_orig = np.linspace(0, T * dt, T)
        T_new = int(T * dt / simulation_dt)
        t_new = np.linspace(0, T * dt, T_new)

        resampled_joint_positions = self._resample_data_Rn(jp_list, t_orig, t_new)
        if joint_velocities_raw is not None:
            resampled_joint_velocities = self._resample_data_Rn(joint_velocities_raw, t_orig, t_new)
        else:
            resampled_joint_velocities = self._compute_raw_derivative(
                resampled_joint_positions, simulation_dt
            )

        resampled_base_positions = self._resample_data_Rn(
            data["root_position"], t_orig, t_new
        )
        resampled_base_orientations = self._resample_data_SO3(
            data["root_quaternion"], t_orig, t_new
        )

        resampled_base_lin_vel_mixed = self._compute_raw_derivative(
            resampled_base_positions, simulation_dt
        )

        resampled_base_ang_vel_mixed = self._compute_ang_vel(
            resampled_base_orientations, simulation_dt, local=False
        )

        if base_lin_vel_local_raw is not None:
            resampled_base_lin_vel_local = self._resample_data_Rn(base_lin_vel_local_raw, t_orig, t_new)
        else:
            resampled_base_lin_vel_local = np.stack(
                [
                    R.as_matrix().T @ v
                    for R, v in zip(
                        resampled_base_orientations, resampled_base_lin_vel_mixed
                    )
                ]
            )
        if base_ang_vel_local_raw is not None:
            resampled_base_ang_vel_local = self._resample_data_Rn(base_ang_vel_local_raw, t_orig, t_new)
        else:
            resampled_base_ang_vel_local = self._compute_ang_vel(
                resampled_base_orientations, simulation_dt, local=True
            )

        resampled_amp_joint_positions = (
            self._resample_data_Rn(amp_joint_positions_raw, t_orig, t_new)
            if amp_joint_positions_raw is not None
            else resampled_joint_positions
        )
        resampled_amp_joint_velocities = (
            self._resample_data_Rn(amp_joint_velocities_raw, t_orig, t_new)
            if amp_joint_velocities_raw is not None
            else resampled_joint_velocities
        )
        resampled_amp_base_lin_vel_local = (
            self._resample_data_Rn(amp_base_lin_vel_local_raw, t_orig, t_new)
            if amp_base_lin_vel_local_raw is not None
            else resampled_base_lin_vel_local
        )
        resampled_amp_base_ang_vel_local = (
            self._resample_data_Rn(amp_base_ang_vel_local_raw, t_orig, t_new)
            if amp_base_ang_vel_local_raw is not None
            else resampled_base_ang_vel_local
        )
        resampled_amp_extra_observations = (
            self._resample_data_Rn(amp_extra_observations_raw, t_orig, t_new)
            if amp_extra_observations_raw is not None
            else None
        )

        return MotionData(
            joint_positions=resampled_joint_positions,
            joint_velocities=resampled_joint_velocities,
            base_lin_velocities_mixed=resampled_base_lin_vel_mixed,
            base_ang_velocities_mixed=resampled_base_ang_vel_mixed,
            base_lin_velocities_local=resampled_base_lin_vel_local,
            base_ang_velocities_local=resampled_base_ang_vel_local,
            base_quat=resampled_base_orientations,
            amp_joint_positions=resampled_amp_joint_positions,
            amp_joint_velocities=resampled_amp_joint_velocities,
            amp_base_lin_velocities_local=resampled_amp_base_lin_vel_local,
            amp_base_ang_velocities_local=resampled_amp_base_ang_vel_local,
            amp_extra_observations=resampled_amp_extra_observations,
            device=self.device,
        )

    def load_expert_txt_data(
        self,
        dataset_path: Path,
        simulation_dt: float,
        slow_down_factor: int,
        expected_joint_names: List[str],
    ) -> MotionData:
        """Load the supported TianKung-style 30D expert txt/json file."""
        payload = load_motion_txt(dataset_path)
        if "FrameLayout" in payload and "Frames" in payload:
            return self._load_tienkung_style_txt_data(payload, simulation_dt, slow_down_factor, expected_joint_names)
        raise ValueError(
            f"Unsupported txt AMP expert format in '{dataset_path}'. "
            "Expected the TianKung-style 30D frame layout payload used by this project."
        )

    def _load_tienkung_style_txt_data(
        self,
        payload: dict,
        simulation_dt: float,
        slow_down_factor: int,
        expected_joint_names: List[str],
    ) -> MotionData:
        """Load the TianKung-style 30D AMP frame layout.

        The exported frame contract is:
            right_leg_q(6), left_leg_q(6),
            right_leg_dq(6), left_leg_dq(6),
            left_foot_pos_local(3), right_foot_pos_local(3)
        """
        frames = np.asarray(payload["Frames"], dtype=np.float32)
        if frames.ndim != 2:
            raise ValueError(f"Expected 2D frame array, got shape {frames.shape}.")

        frame_layout = payload.get("FrameLayout", [])
        frame_layout_dims = payload.get("FrameLayoutDims", [])
        joint_order = payload.get("JointOrder", {})
        if not frame_layout or not frame_layout_dims:
            raise ValueError("TianKung-style expert txt is missing FrameLayout/FrameLayoutDims.")
        if len(frame_layout) != len(frame_layout_dims):
            raise ValueError("FrameLayout and FrameLayoutDims must have the same length.")

        right_leg_names = list(joint_order.get("right_leg_q", []))
        left_leg_names = list(joint_order.get("left_leg_q", []))
        dataset_joint_names = right_leg_names + left_leg_names
        if not dataset_joint_names:
            raise ValueError("TianKung-style expert txt is missing joint ordering metadata.")

        cursor = 0
        blocks: dict[str, np.ndarray] = {}
        for name, width in zip(frame_layout, frame_layout_dims):
            width = int(width)
            blocks[name] = frames[:, cursor : cursor + width]
            cursor += width
        if cursor != frames.shape[1]:
            raise ValueError(
                f"Frame layout width mismatch: parsed {cursor} values but frames have width {frames.shape[1]}."
            )

        required_blocks = [
            "right_leg_q",
            "left_leg_q",
            "right_leg_dq",
            "left_leg_dq",
            "left_foot_pos_local",
            "right_foot_pos_local",
        ]
        missing_blocks = [name for name in required_blocks if name not in blocks]
        if missing_blocks:
            raise ValueError(f"TianKung-style expert txt is missing blocks: {missing_blocks}")

        joint_positions = np.concatenate([blocks["right_leg_q"], blocks["left_leg_q"]], axis=1)
        joint_velocities = np.concatenate([blocks["right_leg_dq"], blocks["left_leg_dq"]], axis=1)
        foot_positions_local = np.concatenate(
            [blocks["left_foot_pos_local"], blocks["right_foot_pos_local"]],
            axis=1,
        )

        idx_map: List[Union[int, None]] = []
        for joint_name in expected_joint_names:
            if joint_name in dataset_joint_names:
                idx_map.append(dataset_joint_names.index(joint_name))
            else:
                idx_map.append(None)

        def reorder_joint_block(block: np.ndarray) -> np.ndarray:
            out = np.zeros((block.shape[0], len(idx_map)), dtype=block.dtype)
            for target_idx, src_idx in enumerate(idx_map):
                if src_idx is not None:
                    out[:, target_idx] = block[:, src_idx]
            return out

        joint_positions = reorder_joint_block(joint_positions)
        joint_velocities = reorder_joint_block(joint_velocities)

        frame_dt = float(payload["FrameDuration"]) * float(slow_down_factor)
        num_frames = frames.shape[0]
        t_orig = np.linspace(0.0, num_frames * frame_dt, num_frames)
        num_frames_resampled = max(2, int(num_frames * frame_dt / simulation_dt))
        t_new = np.linspace(0.0, num_frames * frame_dt, num_frames_resampled)

        resampled_joint_positions = self._resample_data_Rn(joint_positions, t_orig, t_new)
        resampled_joint_velocities = self._resample_data_Rn(joint_velocities, t_orig, t_new)
        resampled_foot_positions_local = self._resample_data_Rn(foot_positions_local, t_orig, t_new)

        zeros3 = np.zeros((num_frames_resampled, 3), dtype=np.float32)
        zeros0 = np.zeros((num_frames_resampled, 0), dtype=np.float32)

        return MotionData(
            joint_positions=resampled_joint_positions,
            joint_velocities=resampled_joint_velocities,
            base_lin_velocities_mixed=zeros3,
            base_ang_velocities_mixed=zeros3,
            base_lin_velocities_local=zeros3,
            base_ang_velocities_local=zeros3,
            base_quat=_identity_quat_wxyz(num_frames_resampled),
            amp_joint_positions=resampled_joint_positions,
            amp_joint_velocities=resampled_joint_velocities,
            amp_base_lin_velocities_local=zeros0,
            amp_base_ang_velocities_local=zeros0,
            amp_extra_observations=resampled_foot_positions_local,
            device=self.device,
        )

    def feed_forward_generator(
        self, num_mini_batch: int, mini_batch_size: int
    ) -> Generator[Tuple[torch.Tensor, torch.Tensor], None, None]:
        """
        Yields mini-batches of (state, next_state) pairs for training,
        sampled directly from precomputed buffers.

        Args:
            num_mini_batch: Number of mini-batches to yield
            mini_batch_size: Size of each mini-batch
        Yields:
            Tuple of (state, next_state) tensors
        """
        for _ in range(num_mini_batch):
            idx = torch.multinomial(
                self.per_frame_weights, mini_batch_size, replacement=True
            )
            yield self.all_obs[idx], self.all_next_obs[idx]

    def sample_conditioned(
        self,
        command_speeds: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample expert transitions conditioned on commanded forward speed.

        Each command speed first samples a clip using a soft weighting over clip-level
        speeds: ``clip_weight * exp(-|speed_clip - speed_cmd| / tau)``. A frame is then
        drawn uniformly from that clip. This avoids biasing the expert sampler toward
        longer clips while still allowing smooth mixing near speed boundaries.
        """
        if command_speeds.ndim == 2 and command_speeds.shape[1] == 1:
            command_speeds = command_speeds[:, 0]
        command_speeds = command_speeds.to(self.device, dtype=torch.float32).reshape(-1)

        tau = max(self.speed_conditioning_tau, 1.0e-6)
        log_base = torch.log(self.dataset_weights + 1.0e-12).unsqueeze(0)
        speed_distance = torch.abs(command_speeds.unsqueeze(1) - self.clip_speeds.unsqueeze(0))
        logits = log_base - speed_distance / tau
        if torch.any(torch.abs(command_speeds) <= self.stand_only_speed_threshold):
            stand_mask = self.clip_is_stand.unsqueeze(0).expand(command_speeds.shape[0], -1)
            near_zero_mask = (torch.abs(command_speeds) <= self.stand_only_speed_threshold).unsqueeze(1)
            logits = torch.where(near_zero_mask & (~stand_mask), torch.full_like(logits, -1.0e9), logits)
        clip_weights = torch.softmax(logits, dim=1)
        clip_idx = torch.multinomial(clip_weights, num_samples=1, replacement=True).squeeze(1)

        clip_starts = self.clip_start_indices_tensor[clip_idx]
        clip_lengths = self.clip_lengths_tensor[clip_idx]
        local_idx = torch.floor(
            torch.rand(clip_idx.shape[0], device=self.device) * clip_lengths.to(torch.float32)
        ).to(torch.long)
        frame_idx = clip_starts + local_idx
        return self.all_obs[frame_idx], self.all_next_obs[frame_idx]

    def get_state_for_reset(self, number_of_samples: int) -> Tuple[torch.Tensor, ...]:
        """
        Randomly samples full states for environment resets,
        sampled directly from the precomputed state buffer.

        Args:
            number_of_samples: Number of samples to retrieve
        Returns:
            Tuple of (quat, joint_positions, joint_velocities, base_lin_velocities, base_ang_velocities)
        """
        idx = torch.multinomial(
            self.per_frame_weights, number_of_samples, replacement=True
        )
        full = self.all_states[idx]
        joint_dim = self.motion_data[0].joint_positions.shape[1]

        # The dimensions of the full state are:
        #   - 4 (quat) + joint_dim (joint_positions) + joint_dim (joint_velocities)
        #   + 3 (base_lin_velocities) + 3 (base_ang_velocities)
        #   = 4 + joint_dim + joint_dim + 3 + 3
        dims = [4, joint_dim, joint_dim, 3, 3]
        return torch.split(full, dims, dim=1)
