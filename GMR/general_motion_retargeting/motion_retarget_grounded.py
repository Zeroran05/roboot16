import math
from typing import Optional, Sequence

import mujoco as mj
import numpy as np

from .motion_retarget import GeneralMotionRetargeting


class GroundedMotionRetargeting(GeneralMotionRetargeting):
    """GMR retargeter with PHUMA-inspired ground projection.

    The base GMR solver still handles the frame-task IK. This wrapper adds a
    post-IK projection pass that keeps the configured feet above the ground and
    optionally locks stance feet in XY to reduce foot skating.
    """

    def __init__(
        self,
        *args,
        ground_height: Optional[float] = None,
        ground_clearance: float = 0.005,
        foot_bodies: Optional[Sequence[str]] = None,
        foot_geom_groups: Optional[Sequence[int]] = (3,),
        ground_mode: str = "support",
        max_root_z_correction: float = 0.25,
        max_root_z_step_down: float = 0.025,
        max_root_z_step_up: float = 0.08,
        root_z_pull_down_alpha: float = 0.45,
        root_z_lift_alpha: float = 1.0,
        contact_source: str = "hybrid",
        support_height_mode: str = "mean",
        use_calibrated_sole: bool = True,
        human_contact_height_threshold: float = 0.06,
        human_contact_speed_threshold: float = 0.35,
        enable_foot_flattening: bool = True,
        foot_flatten_iterations: int = 4,
        foot_flatten_step: float = 0.02,
        foot_flatten_tracking_weight: float = 0.08,
        enable_stance_lock: bool = True,
        stance_height_threshold: float = 0.04,
        stance_speed_threshold: float = 0.35,
        stance_xy_blend: float = 1.0,
        max_stance_xy_correction: float = 0.06,
        recompute_velocity: bool = True,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.ground_height = float(self.ground[2] if ground_height is None else ground_height)
        self.ground_clearance = float(ground_clearance)
        self.foot_bodies = self._resolve_foot_bodies(foot_bodies)
        self.foot_geom_groups = None if foot_geom_groups is None else tuple(int(g) for g in foot_geom_groups)
        self.ground_mode = self._check_choice("ground_mode", ground_mode, ("support", "prevent_penetration"))
        self.max_root_z_correction = float(max_root_z_correction)
        self.max_root_z_step_down = float(max_root_z_step_down)
        self.max_root_z_step_up = float(max_root_z_step_up)
        self.root_z_pull_down_alpha = float(root_z_pull_down_alpha)
        self.root_z_lift_alpha = float(root_z_lift_alpha)
        self.contact_source = self._check_choice("contact_source", contact_source, ("human", "robot", "hybrid"))
        self.support_height_mode = self._check_choice("support_height_mode", support_height_mode, ("lowest", "mean", "highest"))
        self.use_calibrated_sole = bool(use_calibrated_sole)
        self.human_contact_height_threshold = float(human_contact_height_threshold)
        self.human_contact_speed_threshold = float(human_contact_speed_threshold)
        self.enable_foot_flattening = bool(enable_foot_flattening)
        self.foot_flatten_iterations = int(foot_flatten_iterations)
        self.foot_flatten_step = float(foot_flatten_step)
        self.foot_flatten_tracking_weight = float(foot_flatten_tracking_weight)
        self.enable_stance_lock = bool(enable_stance_lock)
        self.stance_height_threshold = float(stance_height_threshold)
        self.stance_speed_threshold = float(stance_speed_threshold)
        self.stance_xy_blend = float(stance_xy_blend)
        self.max_stance_xy_correction = float(max_stance_xy_correction)
        self.recompute_velocity = bool(recompute_velocity)

        self._foot_body_ids = [self.robot_body_names[name] for name in self.foot_bodies]
        self._foot_body_groups = [self._collect_body_descendants(body_id) for body_id in self._foot_body_ids]
        self._foot_part_z_offsets = self._compute_foot_part_z_offsets()
        self._qpos_index_to_joint = {
            int(self.model.jnt_qposadr[joint_id]): int(joint_id)
            for joint_id in range(self.model.njnt)
        }
        self._foot_joint_qpos = self._resolve_foot_flatten_joints()
        self._prev_human_foot_xy: dict[str, np.ndarray] = {}
        self._human_foot_ground_z: dict[str, float] = {}
        self._prev_robot_contact_xy: dict[str, np.ndarray] = {}
        self._prev_foot_xy: dict[str, np.ndarray] = {}
        self._stance_anchors: dict[str, Optional[np.ndarray]] = {name: None for name in self.foot_bodies}
        self._last_ground_report: dict[str, object] = {}
        self.ground_stats = {
            "frames": 0,
            "z_corrected_frames": 0,
            "z_pulled_down_frames": 0,
            "z_lifted_frames": 0,
            "xy_locked_frames": 0,
            "foot_flattened_frames": 0,
            "max_z_correction": 0.0,
            "max_xy_correction": 0.0,
            "max_foot_flat_error": 0.0,
        }

    def retarget(self, human_data, offset_to_ground: bool = False, frame_dt_target: float = None):
        prev_qpos = self.configuration.data.qpos.copy()
        ret = super().retarget(
            human_data,
            offset_to_ground=offset_to_ground,
            frame_dt_target=frame_dt_target,
        )

        if isinstance(ret, tuple):
            if len(ret) == 3:
                _, last_qvel, avg_qvel = ret
            elif len(ret) == 2:
                _, avg_qvel = ret
                last_qvel = avg_qvel
            else:
                raise RuntimeError(f"Unexpected retarget() return length: {len(ret)}")
        else:
            last_qvel = self.configuration.data.qvel.copy()
            avg_qvel = last_qvel.copy()

        dt = float(getattr(self, "last_frame_dt", 0.0) or 0.0)
        stance_feet = self._infer_stance_feet(dt)
        if self.enable_foot_flattening and stance_feet:
            self._apply_support_foot_flattening(stance_feet)
        self._apply_ground_projection(stance_feet)
        if self.enable_stance_lock:
            self._apply_stance_xy_lock(dt, stance_feet)

        corrected_qpos = self.configuration.data.qpos.copy()
        if self.recompute_velocity:
            dt = max(float(getattr(self, "last_frame_dt", 0.0) or 0.0), 1e-12)
            corrected_qvel = np.zeros(self.configuration.model.nv, dtype=float)
            mj.mj_differentiatePos(self.configuration.model, corrected_qvel, dt, prev_qpos, corrected_qpos)
            self.configuration.data.qvel[:] = corrected_qvel
            last_qvel = corrected_qvel
            avg_qvel = corrected_qvel

        self.ground_stats["frames"] += 1
        return corrected_qpos, last_qvel.copy(), avg_qvel.copy()

    def get_ground_report(self) -> dict[str, object]:
        return dict(self._last_ground_report)

    def smooth_root_z_sequence(
        self,
        qpos_seq,
        stance_feet_seq=None,
        window: int = 11,
        passes: int = 2,
        max_step_down: float = 0.006,
        max_step_up: float = 0.04,
        contact_pre_roll: int = 6,
        contact_post_roll: int = 2,
        show_progress: bool = False,
    ) -> np.ndarray:
        """Smooth the exported root-z path with whole-sequence ground constraints.

        Per-frame projection is causal, so a newly detected contact can look like
        a late downward stomp. This pass borrows PHUMA's sequence-level spirit:
        compute the desired stance/ground correction for all frames, smooth that
        correction non-causally, then clamp it so feet still cannot penetrate.
        """
        qpos_arr = np.asarray(qpos_seq, dtype=float).copy()
        if qpos_arr.ndim != 2 or qpos_arr.shape[0] == 0:
            return qpos_arr

        stance_feet_seq = self._expanded_stance_feet_seq(
            stance_feet_seq,
            num_frames=qpos_arr.shape[0],
            pre_roll=contact_pre_roll,
            post_roll=contact_post_roll,
        )

        original_qpos = self.configuration.data.qpos.copy()
        target_min_z = self.ground_height + self.ground_clearance
        desired_z = qpos_arr[:, 2].copy()
        min_safe_z = qpos_arr[:, 2].copy()

        try:
            iterator = enumerate(qpos_arr)
            if show_progress:
                try:
                    from tqdm import tqdm

                    iterator = tqdm(iterator, total=qpos_arr.shape[0], desc="Sequence root-z smoothing")
                except Exception:
                    pass
            for frame_idx, qpos in iterator:
                self.configuration.data.qpos[:] = qpos
                mj.mj_forward(self.configuration.model, self.configuration.data)
                foot_states = self._measure_feet()
                min_z = min(state["min_z"] for state in foot_states.values())
                min_safe_dz = target_min_z - min_z
                min_safe_z[frame_idx] = qpos[2] + min_safe_dz

                raw_dz = max(0.0, min_safe_dz)
                support_feet = stance_feet_seq[frame_idx]
                if self.ground_mode == "support" and support_feet:
                    support_errors = []
                    for foot_name in sorted(support_feet):
                        if foot_name in foot_states:
                            support_errors.append(target_min_z - self._support_height(foot_states[foot_name]))
                    if support_errors:
                        raw_dz = max(float(np.mean(support_errors)), min_safe_dz)
                raw_dz = min(max(raw_dz, -self.max_root_z_correction), self.max_root_z_correction)
                desired_z[frame_idx] = qpos[2] + raw_dz

            desired_z = self._smooth_scalar_sequence(desired_z, window=window, passes=passes)
            desired_z = np.maximum(desired_z, min_safe_z)
            desired_z = self._limit_root_z_steps(
                desired_z,
                min_safe_z=min_safe_z,
                max_step_down=max_step_down,
                max_step_up=max_step_up,
            )
            qpos_arr[:, 2] = desired_z
        finally:
            self.configuration.data.qpos[:] = original_qpos
            mj.mj_forward(self.configuration.model, self.configuration.data)

        return qpos_arr

    def recompute_qvel_sequence(self, qpos_seq, frame_dt_seq) -> np.ndarray:
        qpos_arr = np.asarray(qpos_seq, dtype=float)
        qvel_arr = np.zeros((qpos_arr.shape[0], self.configuration.model.nv), dtype=float)
        if qpos_arr.shape[0] <= 1:
            return qvel_arr

        dt_arr = np.asarray(frame_dt_seq, dtype=float)
        for frame_idx in range(1, qpos_arr.shape[0]):
            if dt_arr.shape[0] == qpos_arr.shape[0] + 1:
                dt = dt_arr[frame_idx + 1]
            elif dt_arr.shape[0] == qpos_arr.shape[0]:
                dt = dt_arr[frame_idx]
            else:
                dt = getattr(self, "last_frame_dt", 0.0)
            dt = max(float(dt), 1e-12)
            mj.mj_differentiatePos(
                self.configuration.model,
                qvel_arr[frame_idx],
                dt,
                qpos_arr[frame_idx - 1],
                qpos_arr[frame_idx],
            )
        qvel_arr[0] = qvel_arr[1]
        return qvel_arr

    def flatten_feet_sequence(
        self,
        qpos_seq,
        stance_feet_seq=None,
        contact_pre_roll: int = 8,
        contact_post_roll: int = 3,
        show_progress: bool = False,
    ) -> np.ndarray:
        qpos_arr = np.asarray(qpos_seq, dtype=float).copy()
        if qpos_arr.ndim != 2 or qpos_arr.shape[0] == 0:
            return qpos_arr

        stance_feet_seq = self._expanded_stance_feet_seq(
            stance_feet_seq,
            num_frames=qpos_arr.shape[0],
            pre_roll=contact_pre_roll,
            post_roll=contact_post_roll,
        )

        original_qpos = self.configuration.data.qpos.copy()
        target_min_z = self.ground_height + self.ground_clearance
        try:
            iterator = enumerate(stance_feet_seq)
            if show_progress:
                try:
                    from tqdm import tqdm

                    iterator = tqdm(iterator, total=qpos_arr.shape[0], desc="Sequence foot flattening")
                except Exception:
                    pass
            for frame_idx, stance_feet in iterator:
                if not stance_feet:
                    continue
                self.configuration.data.qpos[:] = qpos_arr[frame_idx]
                mj.mj_forward(self.configuration.model, self.configuration.data)
                self._apply_support_foot_flattening(stance_feet)
                foot_states = self._measure_feet()
                min_z = min(state["min_z"] for state in foot_states.values())
                if min_z < target_min_z:
                    self.configuration.data.qpos[2] += target_min_z - min_z
                    mj.mj_forward(self.configuration.model, self.configuration.data)
                qpos_arr[frame_idx] = self.configuration.data.qpos.copy()
        finally:
            self.configuration.data.qpos[:] = original_qpos
            mj.mj_forward(self.configuration.model, self.configuration.data)

        return qpos_arr

    def _expanded_stance_feet_seq(
        self,
        stance_feet_seq,
        num_frames: int,
        pre_roll: int,
        post_roll: int,
    ) -> list[set[str]]:
        expanded = [set() for _ in range(num_frames)]
        if stance_feet_seq is None:
            return expanded

        pre_roll = max(0, int(pre_roll))
        post_roll = max(0, int(post_roll))
        for frame_idx, feet in enumerate(stance_feet_seq[:num_frames]):
            if not feet:
                continue
            start = max(0, frame_idx - pre_roll)
            end = min(num_frames, frame_idx + post_roll + 1)
            for dst_idx in range(start, end):
                expanded[dst_idx].update(feet)
        return expanded

    def _smooth_scalar_sequence(self, values: np.ndarray, window: int, passes: int) -> np.ndarray:
        values = np.asarray(values, dtype=float).copy()
        window = int(window)
        if window <= 1 or values.shape[0] <= 2:
            return values
        if window % 2 == 0:
            window += 1
        window = min(window, values.shape[0] if values.shape[0] % 2 == 1 else values.shape[0] - 1)
        if window <= 1:
            return values

        kernel = np.ones(window, dtype=float) / float(window)
        pad = window // 2
        for _ in range(max(1, int(passes))):
            padded = np.pad(values, (pad, pad), mode="edge")
            values = np.convolve(padded, kernel, mode="valid")
        return values

    def _limit_root_z_steps(
        self,
        values: np.ndarray,
        min_safe_z: np.ndarray,
        max_step_down: float,
        max_step_up: float,
    ) -> np.ndarray:
        values = np.asarray(values, dtype=float).copy()
        min_safe_z = np.asarray(min_safe_z, dtype=float)
        if values.shape[0] <= 1:
            return np.maximum(values, min_safe_z)

        values[0] = max(values[0], min_safe_z[0])
        max_step_down = float(max_step_down)
        max_step_up = float(max_step_up)
        for idx in range(1, values.shape[0]):
            values[idx] = max(values[idx], min_safe_z[idx])
            prev = values[idx - 1]
            if max_step_down > 0.0 and values[idx] < prev - max_step_down:
                values[idx] = prev - max_step_down
            if max_step_up > 0.0 and values[idx] > prev + max_step_up:
                values[idx] = max(min_safe_z[idx], prev + max_step_up)
        return values

    def _check_choice(self, label: str, value: str, choices: Sequence[str]) -> str:
        value = str(value)
        if value not in choices:
            raise ValueError(f"{label} must be one of {choices}, got {value!r}")
        return value

    def _resolve_foot_bodies(self, foot_bodies):
        if foot_bodies:
            missing = [name for name in foot_bodies if name not in self.robot_body_names]
            if missing:
                raise ValueError(f"Foot body not found in robot model: {missing}")
            return list(foot_bodies)

        candidates = []
        for preferred in ("left_ankle_roll_link", "right_ankle_roll_link"):
            if preferred in self.robot_body_names:
                candidates.append(preferred)
        if len(candidates) >= 2:
            return candidates

        for name in self.robot_body_names:
            lower = name.lower()
            if ("foot" in lower or ("ankle" in lower and "roll" in lower)) and name not in candidates:
                candidates.append(name)
        if not candidates:
            raise ValueError("Could not auto-detect foot bodies. Pass --foot_bodies explicitly.")
        return candidates

    def _collect_body_descendants(self, body_id: int) -> set[int]:
        ids = {int(body_id)}
        changed = True
        while changed:
            changed = False
            for candidate in range(self.model.nbody):
                parent = int(self.model.body_parentid[candidate])
                if parent in ids and candidate not in ids:
                    ids.add(candidate)
                    changed = True
        return ids

    def _compute_foot_part_z_offsets(self) -> dict[str, dict[str, float]]:
        offsets = {}
        for foot_name, body_ids in zip(self.foot_bodies, self._foot_body_groups):
            part_local_min = {}
            for geom_id in self._foot_geom_ids(body_ids):
                part_name = self._geom_part(geom_id)
                part_local_min[part_name] = min(
                    part_local_min.get(part_name, float("inf")),
                    self._geom_local_min_z(geom_id),
                )
            if not part_local_min:
                offsets[foot_name] = {}
                continue

            sole_reference = min(part_local_min.values())
            offsets[foot_name] = {
                part_name: float(local_min_z - sole_reference)
                for part_name, local_min_z in part_local_min.items()
            }
        return offsets

    def _foot_geom_ids(self, body_ids: set[int]) -> list[int]:
        geom_ids = [
            geom_id
            for geom_id in range(self.model.ngeom)
            if int(self.model.geom_bodyid[geom_id]) in body_ids and self._geom_group_enabled(geom_id)
        ]
        if geom_ids:
            return geom_ids
        return [
            geom_id
            for geom_id in range(self.model.ngeom)
            if int(self.model.geom_bodyid[geom_id]) in body_ids
        ]

    def _resolve_foot_flatten_joints(self) -> dict[str, list[int]]:
        joint_qpos = {}
        for foot_name in self.foot_bodies:
            lower = foot_name.lower()
            side = "left" if "left" in lower else "right" if "right" in lower else None
            if side is not None:
                candidates = [f"{side}_ankle_pitch_joint", f"{side}_ankle_roll_joint"]
            else:
                candidates = [
                    name
                    for name in self.robot_dof_names
                    if "ankle" in name.lower() and ("pitch" in name.lower() or "roll" in name.lower())
                ]

            qpos_indices = []
            for joint_name in candidates:
                joint_id = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_JOINT, joint_name)
                if joint_id < 0:
                    continue
                qpos_adr = int(self.model.jnt_qposadr[joint_id])
                if qpos_adr not in qpos_indices:
                    qpos_indices.append(qpos_adr)
            joint_qpos[foot_name] = qpos_indices
        return joint_qpos

    def _infer_stance_feet(self, dt: float) -> set[str]:
        stance_feet: set[str] = set()
        debug = {}

        if self.contact_source in {"human", "hybrid"}:
            for foot_name in self.foot_bodies:
                contact, info = self._infer_human_foot_contact(foot_name, dt)
                debug[foot_name] = info
                if contact:
                    stance_feet.add(foot_name)

        if self.contact_source in {"robot", "hybrid"}:
            robot_stance, robot_debug = self._infer_robot_foot_contacts(dt)
            stance_feet.update(robot_stance)
            for foot_name, info in robot_debug.items():
                debug.setdefault(foot_name, {}).update({f"robot_{k}": v for k, v in info.items()})

        self._last_contact_debug = debug
        return stance_feet

    def _infer_human_foot_contact(self, foot_name: str, dt: float) -> tuple[bool, dict[str, float]]:
        key = self._human_foot_key_for_robot_foot(foot_name)
        if key is None or not hasattr(self, "scaled_human_data") or key not in self.scaled_human_data:
            return False, {"contact": False}

        pos = np.asarray(self.scaled_human_data[key][0], dtype=float)
        prev_xy = self._prev_human_foot_xy.get(foot_name)
        speed = 0.0
        if prev_xy is not None and dt > 0.0 and math.isfinite(dt):
            speed = float(np.linalg.norm(pos[:2] - prev_xy) / dt)

        running_low = self._human_foot_ground_z.get(foot_name, float(pos[2]))
        running_low = min(running_low, float(pos[2]))
        self._human_foot_ground_z[foot_name] = running_low
        self._prev_human_foot_xy[foot_name] = pos[:2].copy()

        height_from_low = float(pos[2] - running_low)
        contact = (
            height_from_low <= self.human_contact_height_threshold
            and speed <= self.human_contact_speed_threshold
        )
        return contact, {
            "contact": bool(contact),
            "human_key": key,
            "height_from_low": height_from_low,
            "speed": speed,
        }

    def _human_foot_key_for_robot_foot(self, foot_name: str) -> Optional[str]:
        lower = foot_name.lower()
        if "left" in lower:
            candidates = ("LeftFootMod", "LeftFoot", "LeftToe", "LeftToeBase")
        elif "right" in lower:
            candidates = ("RightFootMod", "RightFoot", "RightToe", "RightToeBase")
        else:
            candidates = ("LeftFootMod", "RightFootMod", "LeftFoot", "RightFoot")
        if not hasattr(self, "scaled_human_data"):
            return None
        for key in candidates:
            if key in self.scaled_human_data:
                return key
        return None

    def _infer_robot_foot_contacts(self, dt: float) -> tuple[set[str], dict[str, dict[str, float]]]:
        foot_states = self._measure_feet()
        stance_feet = set()
        debug = {}
        for foot_name, state in foot_states.items():
            xy = state["xy"]
            prev_xy = self._prev_robot_contact_xy.get(foot_name)
            speed = 0.0
            if prev_xy is not None and dt > 0.0 and math.isfinite(dt):
                speed = float(np.linalg.norm(xy - prev_xy) / dt)
            near_ground = state["min_z"] <= self.ground_height + self.ground_clearance + self.stance_height_threshold
            contact = near_ground and speed <= self.stance_speed_threshold
            if contact:
                stance_feet.add(foot_name)
            self._prev_robot_contact_xy[foot_name] = xy.copy()
            debug[foot_name] = {"contact": bool(contact), "speed": speed, "near_ground": bool(near_ground)}
        return stance_feet, debug

    def _apply_support_foot_flattening(self, stance_feet: set[str]) -> None:
        if self.foot_flatten_iterations <= 0 or self.foot_flatten_step <= 0.0:
            return

        changed = False
        original_qpos = self.configuration.data.qpos.copy()
        for foot_name in sorted(stance_feet):
            qpos_indices = self._foot_joint_qpos.get(foot_name, [])
            if not qpos_indices:
                continue
            for _ in range(self.foot_flatten_iterations):
                for qpos_idx in qpos_indices:
                    best_value = float(self.configuration.data.qpos[qpos_idx])
                    best_loss = self._foot_flatten_loss(foot_name, original_qpos)
                    for delta in (-self.foot_flatten_step, self.foot_flatten_step):
                        candidate = self._clip_qpos_index(qpos_idx, best_value + delta)
                        old_value = float(self.configuration.data.qpos[qpos_idx])
                        self.configuration.data.qpos[qpos_idx] = candidate
                        mj.mj_forward(self.configuration.model, self.configuration.data)
                        loss = self._foot_flatten_loss(foot_name, original_qpos)
                        if loss < best_loss:
                            best_loss = loss
                            best_value = candidate
                        self.configuration.data.qpos[qpos_idx] = old_value
                        mj.mj_forward(self.configuration.model, self.configuration.data)
                    if abs(float(self.configuration.data.qpos[qpos_idx]) - best_value) > 1e-12:
                        self.configuration.data.qpos[qpos_idx] = best_value
                        mj.mj_forward(self.configuration.model, self.configuration.data)
                        changed = True

        if changed:
            self.ground_stats["foot_flattened_frames"] += 1

    def _foot_flatten_loss(self, foot_name: str, original_qpos: np.ndarray) -> float:
        state = self._measure_feet()[foot_name]
        heights = self._support_part_heights(state)
        if len(heights) < 2:
            return 0.0

        heights_arr = np.asarray(heights, dtype=float)
        flat_error = float(np.max(heights_arr) - np.min(heights_arr))
        target_z = self.ground_height + self.ground_clearance
        mean_height = float(np.mean(heights_arr))
        penetration = max(0.0, target_z - float(state["min_z"]))
        hover = max(0.0, mean_height - target_z)

        tracking = 0.0
        for qpos_idx in self._foot_joint_qpos.get(foot_name, []):
            tracking += float((self.configuration.data.qpos[qpos_idx] - original_qpos[qpos_idx]) ** 2)

        return (
            flat_error * flat_error
            + 0.25 * hover * hover
            + 8.0 * penetration * penetration
            + self.foot_flatten_tracking_weight * tracking
        )

    def _clip_qpos_index(self, qpos_idx: int, value: float) -> float:
        joint_id = self._qpos_index_to_joint.get(int(qpos_idx))
        if joint_id is None:
            return float(value)
        if bool(self.model.jnt_limited[joint_id]):
            lo, hi = self.model.jnt_range[joint_id]
            value = min(max(float(value), float(lo)), float(hi))
        return float(value)

    def _apply_ground_projection(self, stance_feet: Optional[set[str]] = None) -> None:
        foot_states = self._measure_feet()
        min_z = min(state["min_z"] for state in foot_states.values())
        target_min_z = self.ground_height + self.ground_clearance
        penetration_dz = max(0.0, target_min_z - min_z)
        raw_dz = penetration_dz

        support_feet_used = []
        if self.ground_mode == "support" and stance_feet:
            support_errors = []
            for foot_name in sorted(stance_feet):
                if foot_name not in foot_states:
                    continue
                support_height = self._support_height(foot_states[foot_name])
                support_errors.append(target_min_z - support_height)
                support_feet_used.append(foot_name)
            if support_errors:
                support_dz = float(np.mean(support_errors))
                min_allowed_dz = target_min_z - min_z
                raw_dz = max(support_dz, min_allowed_dz)

        raw_dz = min(max(raw_dz, -self.max_root_z_correction), self.max_root_z_correction)
        dz = self._filter_root_z_correction(raw_dz)

        if abs(dz) > 1e-12:
            self.configuration.data.qpos[2] += dz
            mj.mj_forward(self.configuration.model, self.configuration.data)
            self.ground_stats["z_corrected_frames"] += 1
            if dz < 0.0:
                self.ground_stats["z_pulled_down_frames"] += 1
            else:
                self.ground_stats["z_lifted_frames"] += 1
            self.ground_stats["max_z_correction"] = max(self.ground_stats["max_z_correction"], abs(float(dz)))

        post_states = self._measure_feet()
        max_flat_error = max(self._foot_flat_error(state) for state in post_states.values())
        self.ground_stats["max_foot_flat_error"] = max(
            self.ground_stats["max_foot_flat_error"],
            float(max_flat_error),
        )
        self._last_ground_report = {
            "min_foot_z": float(min_z),
            "target_min_z": float(target_min_z),
            "raw_root_z_correction": float(raw_dz),
            "root_z_correction": float(dz),
            "stance_feet": sorted(stance_feet or []),
            "support_feet_used": support_feet_used,
            "contact_debug": getattr(self, "_last_contact_debug", {}),
            "feet": post_states,
        }

    def _filter_root_z_correction(self, raw_dz: float) -> float:
        if raw_dz < 0.0:
            dz = raw_dz * self.root_z_pull_down_alpha
            if self.max_root_z_step_down > 0.0:
                dz = max(dz, -self.max_root_z_step_down)
            return float(dz)

        dz = raw_dz * self.root_z_lift_alpha
        if self.max_root_z_step_up > 0.0:
            dz = min(dz, self.max_root_z_step_up)
        return float(dz)

    def _support_height(self, foot_state: dict[str, object]) -> float:
        heights = self._support_part_heights(foot_state)
        if not heights:
            return float(foot_state["min_z"])
        if self.support_height_mode == "lowest":
            return float(np.min(heights))
        if self.support_height_mode == "highest":
            return float(np.max(heights))
        return float(np.mean(heights))

    def _support_part_heights(self, foot_state: dict[str, object]) -> list[float]:
        parts = foot_state.get("parts", {})
        heights = []
        for part_name in ("fore", "rear", "heel", "sole"):
            if part_name in parts:
                key = "calibrated_min_z" if self.use_calibrated_sole else "min_z"
                heights.append(float(parts[part_name].get(key, parts[part_name]["min_z"])))
        if not heights and "other" in parts:
            key = "calibrated_min_z" if self.use_calibrated_sole else "min_z"
            heights.append(float(parts["other"].get(key, parts["other"]["min_z"])))
        return heights

    def _foot_flat_error(self, foot_state: dict[str, object]) -> float:
        heights = self._support_part_heights(foot_state)
        if len(heights) < 2:
            return 0.0
        return float(np.max(heights) - np.min(heights))

    def _apply_stance_xy_lock(self, dt: float, stance_feet: Optional[set[str]] = None) -> None:
        if dt <= 0.0 or not math.isfinite(dt):
            return

        foot_states = self._measure_feet()
        corrections = []
        for name, state in foot_states.items():
            xy = state["xy"]
            prev_xy = self._prev_foot_xy.get(name)
            speed = 0.0 if prev_xy is None else float(np.linalg.norm(xy - prev_xy) / dt)
            near_ground = state["min_z"] <= self.ground_height + self.ground_clearance + self.stance_height_threshold
            if stance_feet is None:
                is_stance = near_ground and speed <= self.stance_speed_threshold
            else:
                is_stance = name in stance_feet and near_ground

            if is_stance:
                anchor = self._stance_anchors.get(name)
                if anchor is None:
                    self._stance_anchors[name] = xy.copy()
                else:
                    corrections.append(anchor - xy)
            else:
                self._stance_anchors[name] = None
            self._prev_foot_xy[name] = xy.copy()

        if not corrections:
            return

        dxy = np.mean(np.stack(corrections, axis=0), axis=0) * self.stance_xy_blend
        norm = float(np.linalg.norm(dxy))
        if norm > self.max_stance_xy_correction > 0.0:
            dxy *= self.max_stance_xy_correction / norm
            norm = self.max_stance_xy_correction

        if norm > 0.0:
            self.configuration.data.qpos[0:2] += dxy
            mj.mj_forward(self.configuration.model, self.configuration.data)
            self.ground_stats["xy_locked_frames"] += 1
            self.ground_stats["max_xy_correction"] = max(self.ground_stats["max_xy_correction"], norm)
            self._last_ground_report["root_xy_correction"] = dxy.astype(float).tolist()

    def _measure_feet(self) -> dict[str, dict[str, object]]:
        mj.mj_forward(self.configuration.model, self.configuration.data)
        states = {}
        for foot_name, body_ids in zip(self.foot_bodies, self._foot_body_groups):
            geom_ids = self._foot_geom_ids(body_ids)
            if not geom_ids:
                body_id = self.robot_body_names[foot_name]
                body_pos = self.configuration.data.xpos[body_id]
                states[foot_name] = {
                    "min_z": float(body_pos[2]),
                    "xy": body_pos[:2].copy(),
                    "geom_ids": [],
                    "parts": {},
                    "flat_error": 0.0,
                }
                continue

            min_z = float("inf")
            calibrated_min_z = float("inf")
            xy_accum = []
            parts = {}
            part_offsets = self._foot_part_z_offsets.get(foot_name, {})
            for geom_id in geom_ids:
                geom_min_z = self._geom_min_z(geom_id)
                geom_xy = self.configuration.data.geom_xpos[geom_id, :2].copy()
                part_name = self._geom_part(geom_id)
                geom_calibrated_min_z = geom_min_z - part_offsets.get(part_name, 0.0)
                min_z = min(min_z, geom_min_z)
                calibrated_min_z = min(calibrated_min_z, geom_calibrated_min_z)
                xy_accum.append(geom_xy)
                part = parts.setdefault(
                    part_name,
                    {"min_z": float("inf"), "calibrated_min_z": float("inf"), "xy_values": [], "geom_ids": []},
                )
                part["min_z"] = min(float(part["min_z"]), float(geom_min_z))
                part["calibrated_min_z"] = min(float(part["calibrated_min_z"]), float(geom_calibrated_min_z))
                part["xy_values"].append(geom_xy)
                part["geom_ids"].append(geom_id)

            part_report = {}
            for part_name, part in parts.items():
                part_report[part_name] = {
                    "min_z": float(part["min_z"]),
                    "calibrated_min_z": float(part["calibrated_min_z"]),
                    "sole_z_offset": float(part_offsets.get(part_name, 0.0)),
                    "xy": np.mean(np.stack(part["xy_values"], axis=0), axis=0),
                    "geom_ids": part["geom_ids"],
                }
            states[foot_name] = {
                "min_z": float(min_z),
                "calibrated_min_z": float(calibrated_min_z),
                "xy": np.mean(np.stack(xy_accum, axis=0), axis=0),
                "geom_ids": geom_ids,
                "parts": part_report,
            }
            states[foot_name]["flat_error"] = self._foot_flat_error(states[foot_name])
        return states

    def _geom_part(self, geom_id: int) -> str:
        name = mj.mj_id2name(self.model, mj.mjtObj.mjOBJ_GEOM, geom_id) or ""
        lower = name.lower()
        if "fore" in lower or "toe" in lower:
            return "fore"
        if "rear" in lower:
            return "rear"
        if "heel" in lower:
            return "heel"
        if "sole" in lower:
            return "sole"
        return "other"

    def _geom_group_enabled(self, geom_id: int) -> bool:
        return self.foot_geom_groups is None or int(self.model.geom_group[geom_id]) in self.foot_geom_groups

    def _geom_min_z(self, geom_id: int) -> float:
        geom_type = self.model.geom_type[geom_id]
        pos = self.configuration.data.geom_xpos[geom_id]
        mat = self.configuration.data.geom_xmat[geom_id].reshape(3, 3)
        size = self.model.geom_size[geom_id]

        if geom_type == mj.mjtGeom.mjGEOM_MESH:
            mesh_id = self.model.geom_dataid[geom_id]
            vert_adr = self.model.mesh_vertadr[mesh_id]
            vert_num = self.model.mesh_vertnum[mesh_id]
            verts = self.model.mesh_vert[vert_adr: vert_adr + vert_num]
            world = verts @ mat.T + pos
            return float(np.min(world[:, 2]))

        if geom_type == mj.mjtGeom.mjGEOM_BOX:
            sx, sy, sz = size[:3]
            corners = np.array(
                [[x, y, z] for x in (-sx, sx) for y in (-sy, sy) for z in (-sz, sz)],
                dtype=float,
            )
            world = corners @ mat.T + pos
            return float(np.min(world[:, 2]))

        if geom_type == mj.mjtGeom.mjGEOM_SPHERE:
            return float(pos[2] - size[0])

        if geom_type in (mj.mjtGeom.mjGEOM_CAPSULE, mj.mjtGeom.mjGEOM_CYLINDER):
            axis = mat @ np.array([0.0, 0.0, 1.0])
            half_len = size[1]
            radius = size[0]
            end_a = pos - axis * half_len
            end_b = pos + axis * half_len
            return float(min(end_a[2], end_b[2]) - radius)

        return float(pos[2])

    def _geom_local_min_z(self, geom_id: int) -> float:
        geom_type = self.model.geom_type[geom_id]
        pos = self.model.geom_pos[geom_id]
        mat = self._quat_to_mat_wxyz(self.model.geom_quat[geom_id])
        size = self.model.geom_size[geom_id]

        if geom_type == mj.mjtGeom.mjGEOM_MESH:
            mesh_id = self.model.geom_dataid[geom_id]
            vert_adr = self.model.mesh_vertadr[mesh_id]
            vert_num = self.model.mesh_vertnum[mesh_id]
            verts = self.model.mesh_vert[vert_adr: vert_adr + vert_num]
            local = verts @ mat.T + pos
            return float(np.min(local[:, 2]))

        if geom_type == mj.mjtGeom.mjGEOM_BOX:
            sx, sy, sz = size[:3]
            corners = np.array(
                [[x, y, z] for x in (-sx, sx) for y in (-sy, sy) for z in (-sz, sz)],
                dtype=float,
            )
            local = corners @ mat.T + pos
            return float(np.min(local[:, 2]))

        if geom_type == mj.mjtGeom.mjGEOM_SPHERE:
            return float(pos[2] - size[0])

        if geom_type in (mj.mjtGeom.mjGEOM_CAPSULE, mj.mjtGeom.mjGEOM_CYLINDER):
            axis = mat @ np.array([0.0, 0.0, 1.0])
            half_len = size[1]
            radius = size[0]
            end_a = pos - axis * half_len
            end_b = pos + axis * half_len
            return float(min(end_a[2], end_b[2]) - radius)

        return float(pos[2])

    def _quat_to_mat_wxyz(self, quat) -> np.ndarray:
        quat = np.asarray(quat, dtype=float)
        norm = float(np.linalg.norm(quat))
        if norm <= 1e-12:
            return np.eye(3)
        w, x, y, z = quat / norm
        return np.array(
            [
                [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
                [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
                [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
            ],
            dtype=float,
        )
