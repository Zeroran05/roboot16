from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import numpy as np


# Roboot16 motion PKL convention used here:
# - root_pos: (T, 3), world frame, xyz
# - root_rot: (T, 4), quaternion in xyzw convention
# - root_vel: (T, 3), world-frame linear velocity
# - root_vel_body: (T, 3), body/local-frame linear velocity
# - root_rot_vel: (T, 3), body/local-frame angular velocity
# - dof_pos / dof_vel: (T, 12), joint order:
#   [right_leg(6), left_leg(6)]
#
# Left-right mirror mode for Roboot16 running clips:
# - keep forward running direction along +x
# - mirror lateral offset across the world xz-plane: y -> -y
# - swap left/right leg trajectories
# - for this dataset, left/right joint axes are already mirror-defined in the
#   exported motion, so swapping the two 6-DoF leg blocks is enough and joint
#   signs stay unchanged


LEG_MIRROR_SIGNS = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float32)
LINEAR_REFLECTION_SIGNS = np.array([1.0, -1.0, 1.0], dtype=np.float32)
ANGULAR_REFLECTION_SIGNS = np.array([-1.0, 1.0, -1.0], dtype=np.float32)
REFLECTION_MATRIX = np.diag(LINEAR_REFLECTION_SIGNS.astype(np.float64))


def quat_xyzw_to_matrix(quat_xyzw: np.ndarray) -> np.ndarray:
    q = np.asarray(quat_xyzw, dtype=np.float64)
    single = q.ndim == 1
    if single:
        q = q[None, :]

    x = q[:, 0]
    y = q[:, 1]
    z = q[:, 2]
    w = q[:, 3]

    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z

    mats = np.empty((q.shape[0], 3, 3), dtype=np.float64)
    mats[:, 0, 0] = 1.0 - 2.0 * (yy + zz)
    mats[:, 0, 1] = 2.0 * (xy - wz)
    mats[:, 0, 2] = 2.0 * (xz + wy)
    mats[:, 1, 0] = 2.0 * (xy + wz)
    mats[:, 1, 1] = 1.0 - 2.0 * (xx + zz)
    mats[:, 1, 2] = 2.0 * (yz - wx)
    mats[:, 2, 0] = 2.0 * (xz - wy)
    mats[:, 2, 1] = 2.0 * (yz + wx)
    mats[:, 2, 2] = 1.0 - 2.0 * (xx + yy)
    return mats[0] if single else mats


def matrix_to_quat_xyzw(rot: np.ndarray) -> np.ndarray:
    r = np.asarray(rot, dtype=np.float64)
    single = r.ndim == 2
    if single:
        r = r[None, :, :]

    out = np.empty((r.shape[0], 4), dtype=np.float64)
    for i, m in enumerate(r):
        trace = float(np.trace(m))
        if trace > 0.0:
            s = 2.0 * np.sqrt(trace + 1.0)
            w = 0.25 * s
            x = (m[2, 1] - m[1, 2]) / s
            y = (m[0, 2] - m[2, 0]) / s
            z = (m[1, 0] - m[0, 1]) / s
        elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
            s = 2.0 * np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2])
            w = (m[2, 1] - m[1, 2]) / s
            x = 0.25 * s
            y = (m[0, 1] + m[1, 0]) / s
            z = (m[0, 2] + m[2, 0]) / s
        elif m[1, 1] > m[2, 2]:
            s = 2.0 * np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2])
            w = (m[0, 2] - m[2, 0]) / s
            x = (m[0, 1] + m[1, 0]) / s
            y = 0.25 * s
            z = (m[1, 2] + m[2, 1]) / s
        else:
            s = 2.0 * np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1])
            w = (m[1, 0] - m[0, 1]) / s
            x = (m[0, 2] + m[2, 0]) / s
            y = (m[1, 2] + m[2, 1]) / s
            z = 0.25 * s
        quat = np.array([x, y, z, w], dtype=np.float64)
        quat /= np.linalg.norm(quat) + 1e-12
        out[i] = quat
    return out[0] if single else out


def relative_rotvec_body_xyzw(root_rot_xyzw: np.ndarray, dt: float) -> np.ndarray:
    rot = quat_xyzw_to_matrix(root_rot_xyzw)
    if rot.ndim != 3:
        raise ValueError(f"Expected quaternion sequence with shape (T, 4), got {root_rot_xyzw.shape}")

    num_frames = rot.shape[0]
    omega = np.zeros((num_frames, 3), dtype=np.float64)
    if num_frames <= 1 or dt <= 0.0:
        return omega.astype(np.float32)

    for i in range(num_frames - 1):
        r_rel = rot[i].T @ rot[i + 1]
        trace = np.clip((np.trace(r_rel) - 1.0) * 0.5, -1.0, 1.0)
        angle = float(np.arccos(trace))
        if angle < 1e-8:
            rotvec = np.zeros(3, dtype=np.float64)
        else:
            axis = np.array(
                [
                    r_rel[2, 1] - r_rel[1, 2],
                    r_rel[0, 2] - r_rel[2, 0],
                    r_rel[1, 0] - r_rel[0, 1],
                ],
                dtype=np.float64,
            )
            axis /= np.linalg.norm(axis) + 1e-12
            rotvec = axis * angle
        omega[i] = rotvec / dt
    omega[-1] = omega[-2]
    return omega.astype(np.float32)


def mirror_motion(data: dict) -> dict:
    mirrored = dict(data)

    root_pos = np.asarray(data["root_pos"], dtype=np.float32).copy()
    root_pos[:, 1] *= -1.0
    mirrored["root_pos"] = root_pos

    root_rot = np.asarray(data["root_rot"], dtype=np.float32)
    root_rot_mats = quat_xyzw_to_matrix(root_rot)
    mirrored_rot_mats = REFLECTION_MATRIX[None, :, :] @ root_rot_mats @ REFLECTION_MATRIX[None, :, :]
    mirrored["root_rot"] = matrix_to_quat_xyzw(mirrored_rot_mats).astype(np.float32)

    if "root_vel" in data:
        root_vel = np.asarray(data["root_vel"], dtype=np.float32).copy()
        root_vel *= LINEAR_REFLECTION_SIGNS
        mirrored["root_vel"] = root_vel

    if "root_vel_body" in data:
        root_vel_body = np.asarray(data["root_vel_body"], dtype=np.float32).copy()
        root_vel_body *= LINEAR_REFLECTION_SIGNS
        mirrored["root_vel_body"] = root_vel_body

    if "root_rot_vel" in data:
        root_rot_vel = np.asarray(data["root_rot_vel"], dtype=np.float32).copy()
        root_rot_vel *= ANGULAR_REFLECTION_SIGNS
        mirrored["root_rot_vel"] = root_rot_vel
    elif "fps" in data:
        dt = 1.0 / float(data["fps"])
        mirrored["root_rot_vel"] = relative_rotvec_body_xyzw(mirrored["root_rot"], dt)

    for key in ("dof_pos", "dof_vel"):
        if key not in data:
            continue
        array = np.asarray(data[key], dtype=np.float32).copy()
        if array.shape[1] != 12:
            raise ValueError(f"Expected {key} to have 12 columns, got {array.shape}.")
        right = array[:, :6].copy()
        left = array[:, 6:].copy()
        array[:, :6] = left * LEG_MIRROR_SIGNS
        array[:, 6:] = right * LEG_MIRROR_SIGNS
        mirrored[key] = array

    return mirrored


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Mirror a Roboot16 motion PKL across left/right.")
    parser.add_argument("input_pkl", type=Path, help="Input PKL file.")
    parser.add_argument("output_pkl", type=Path, help="Output mirrored PKL file.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with args.input_pkl.open("rb") as f:
        data = pickle.load(f)
    mirrored = mirror_motion(data)
    args.output_pkl.parent.mkdir(parents=True, exist_ok=True)
    with args.output_pkl.open("wb") as f:
        pickle.dump(mirrored, f)
    print(f"Mirrored {args.input_pkl} -> {args.output_pkl}")


if __name__ == "__main__":
    main()
