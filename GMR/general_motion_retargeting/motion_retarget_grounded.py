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
        max_root_z_correction: float = 0.25,
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
        self.max_root_z_correction = float(max_root_z_correction)
        self.enable_stance_lock = bool(enable_stance_lock)
        self.stance_height_threshold = float(stance_height_threshold)
        self.stance_speed_threshold = float(stance_speed_threshold)
        self.stance_xy_blend = float(stance_xy_blend)
        self.max_stance_xy_correction = float(max_stance_xy_correction)
        self.recompute_velocity = bool(recompute_velocity)

        self._foot_body_ids = [self.robot_body_names[name] for name in self.foot_bodies]
        self._foot_body_groups = [self._collect_body_descendants(body_id) for body_id in self._foot_body_ids]
        self._prev_foot_xy: dict[str, np.ndarray] = {}
        self._stance_anchors: dict[str, Optional[np.ndarray]] = {name: None for name in self.foot_bodies}
        self._last_ground_report: dict[str, object] = {}
        self.ground_stats = {
            "frames": 0,
            "z_corrected_frames": 0,
            "xy_locked_frames": 0,
            "max_z_correction": 0.0,
            "max_xy_correction": 0.0,
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

        self._apply_ground_projection()
        if self.enable_stance_lock:
            self._apply_stance_xy_lock(float(getattr(self, "last_frame_dt", 0.0) or 0.0))

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

    def _apply_ground_projection(self) -> None:
        foot_states = self._measure_feet()
        min_z = min(state["min_z"] for state in foot_states.values())
        target_min_z = self.ground_height + self.ground_clearance
        dz = max(0.0, target_min_z - min_z)
        dz = min(dz, self.max_root_z_correction)

        if dz > 0.0:
            self.configuration.data.qpos[2] += dz
            mj.mj_forward(self.configuration.model, self.configuration.data)
            self.ground_stats["z_corrected_frames"] += 1
            self.ground_stats["max_z_correction"] = max(self.ground_stats["max_z_correction"], float(dz))

        self._last_ground_report = {
            "min_foot_z": float(min_z),
            "target_min_z": float(target_min_z),
            "root_z_correction": float(dz),
            "feet": foot_states,
        }

    def _apply_stance_xy_lock(self, dt: float) -> None:
        if dt <= 0.0 or not math.isfinite(dt):
            return

        foot_states = self._measure_feet()
        corrections = []
        for name, state in foot_states.items():
            xy = state["xy"]
            prev_xy = self._prev_foot_xy.get(name)
            speed = 0.0 if prev_xy is None else float(np.linalg.norm(xy - prev_xy) / dt)
            near_ground = state["min_z"] <= self.ground_height + self.ground_clearance + self.stance_height_threshold
            is_stance = near_ground and speed <= self.stance_speed_threshold

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
            geom_ids = [
                geom_id
                for geom_id in range(self.model.ngeom)
                if int(self.model.geom_bodyid[geom_id]) in body_ids and self._geom_group_enabled(geom_id)
            ]
            if not geom_ids:
                geom_ids = [
                    geom_id
                    for geom_id in range(self.model.ngeom)
                    if int(self.model.geom_bodyid[geom_id]) in body_ids
                ]
            if not geom_ids:
                body_id = self.robot_body_names[foot_name]
                body_pos = self.configuration.data.xpos[body_id]
                states[foot_name] = {"min_z": float(body_pos[2]), "xy": body_pos[:2].copy(), "geom_ids": []}
                continue

            min_z = float("inf")
            xy_accum = []
            for geom_id in geom_ids:
                min_z = min(min_z, self._geom_min_z(geom_id))
                xy_accum.append(self.configuration.data.geom_xpos[geom_id, :2].copy())
            states[foot_name] = {
                "min_z": float(min_z),
                "xy": np.mean(np.stack(xy_accum, axis=0), axis=0),
                "geom_ids": geom_ids,
            }
        return states

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
