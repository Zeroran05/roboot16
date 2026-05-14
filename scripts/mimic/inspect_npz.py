from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


DEFAULT_ROBOOT16_BODY_NAMES = [
    "base_link",
    "right_hip_pitch_link",
    "right_hip_roll_link",
    "right_hip_yaw_link",
    "right_knee_link",
    "right_ankle_pitch_link",
    "right_ankle_roll_link",
    "left_hip_pitch_link",
    "left_hip_roll_link",
    "left_hip_yaw_link",
    "left_knee_link",
    "left_ankle_pitch_link",
    "left_ankle_roll_link",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect and plot mimic NPZ motion quality.")
    parser.add_argument("--npz", type=Path, required=True, help="Path to the motion npz file.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory to save plots and summary. Defaults to <npz_stem>_inspect.",
    )
    parser.add_argument(
        "--body-names-json",
        type=Path,
        default=None,
        help="Optional JSON file containing a body name list for the NPZ body order.",
    )
    parser.add_argument(
        "--base-body",
        type=str,
        default="base_link",
        help="Base body name used for plots/statistics when body names are available.",
    )
    parser.add_argument(
        "--left-foot-body",
        type=str,
        default="left_ankle_roll_link",
        help="Left foot body name used for plots/statistics when body names are available.",
    )
    parser.add_argument(
        "--right-foot-body",
        type=str,
        default="right_ankle_roll_link",
        help="Right foot body name used for plots/statistics when body names are available.",
    )
    parser.add_argument("--base-index", type=int, default=None, help="Override base body index.")
    parser.add_argument("--left-foot-index", type=int, default=None, help="Override left foot body index.")
    parser.add_argument("--right-foot-index", type=int, default=None, help="Override right foot body index.")
    parser.add_argument(
        "--ground-height",
        type=float,
        default=0.0,
        help="Ground height used for foot penetration checks.",
    )
    parser.add_argument(
        "--thresholds",
        type=float,
        nargs="*",
        default=[0.0, -0.005, -0.01, -0.02],
        help="Foot height thresholds for penetration statistics.",
    )
    parser.add_argument("--no-plots", action="store_true", help="Skip saving plots and only print stats.")
    parser.add_argument("--show", action="store_true", help="Show plots interactively.")
    return parser.parse_args()


def load_body_names(args: argparse.Namespace, body_count: int) -> list[str]:
    if args.body_names_json is not None:
        body_names = json.loads(args.body_names_json.read_text())
    elif body_count == len(DEFAULT_ROBOOT16_BODY_NAMES):
        body_names = DEFAULT_ROBOOT16_BODY_NAMES
    else:
        body_names = [f"body_{index}" for index in range(body_count)]
    if len(body_names) != body_count:
        raise ValueError(f"Body name count mismatch: got {len(body_names)}, expected {body_count}.")
    return body_names


def resolve_body_index(
    override_index: int | None, body_name: str, body_names: list[str], body_count: int, label: str
) -> int:
    if override_index is not None:
        if override_index < 0 or override_index >= body_count:
            raise ValueError(f"{label} index {override_index} is out of range for {body_count} bodies.")
        return override_index
    if body_name not in body_names:
        raise ValueError(f"{label} body '{body_name}' not found in body names: {body_names}")
    return body_names.index(body_name)


def finite_diff(values: np.ndarray, fps: float) -> np.ndarray:
    if len(values) < 2:
        return np.zeros_like(values)
    return np.gradient(values, 1.0 / fps, axis=0)


def norm_last_dim(values: np.ndarray) -> np.ndarray:
    return np.linalg.norm(values, axis=-1)


def summarize_thresholds(values: np.ndarray, thresholds: list[float]) -> list[str]:
    lines = []
    frame_count = len(values)
    for threshold in thresholds:
        count = int(np.sum(values < threshold))
        ratio = 100.0 * count / max(frame_count, 1)
        lines.append(f"  z < {threshold:+.3f} m: {count} frames ({ratio:.2f}%)")
    return lines


def top_k_frames(values: np.ndarray, k: int = 10) -> list[int]:
    if len(values) == 0:
        return []
    k = min(k, len(values))
    indexes = np.argpartition(values, -k)[-k:]
    return indexes[np.argsort(values[indexes])[::-1]].tolist()


def format_stats(name: str, values: np.ndarray, unit: str) -> str:
    return (
        f"{name}: min={np.min(values):.4f} {unit}, "
        f"mean={np.mean(values):.4f} {unit}, "
        f"max={np.max(values):.4f} {unit}"
    )


def save_line_plot(
    output_path: Path,
    time_axis: np.ndarray,
    series: list[tuple[str, np.ndarray]],
    title: str,
    ylabel: str,
    extra_hlines: list[tuple[float, str]] | None = None,
) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(12, 5))
    for label, values in series:
        ax.plot(time_axis, values, linewidth=1.5, label=label)
    if extra_hlines is not None:
        for height, label in extra_hlines:
            ax.axhline(height, linestyle="--", linewidth=1.0, color="gray", label=label)
    ax.set_title(title)
    ax.set_xlabel("Time [s]")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def main() -> None:
    args = parse_args()

    if not args.no_plots or args.show:
        try:
            import matplotlib
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "matplotlib is required for plotting. Install it with `pip install matplotlib`, "
                "or rerun with `--no-plots` to print statistics only."
            ) from exc
        if not args.show:
            matplotlib.use("Agg")

    motion = np.load(args.npz)
    fps_array = np.asarray(motion["fps"]).reshape(-1)
    if len(fps_array) == 0:
        raise ValueError("The motion npz does not contain a valid fps entry.")
    fps = float(fps_array[0])
    joint_pos = motion["joint_pos"]
    joint_vel = motion["joint_vel"]
    body_pos_w = motion["body_pos_w"]
    body_quat_w = motion["body_quat_w"]
    body_lin_vel_w = motion["body_lin_vel_w"]
    body_ang_vel_w = motion["body_ang_vel_w"]

    frame_count = int(joint_pos.shape[0])
    body_count = int(body_pos_w.shape[1])
    time_axis = np.arange(frame_count, dtype=np.float64) / fps
    duration = frame_count / fps if fps > 0 else 0.0

    body_names = load_body_names(args, body_count)
    base_index = resolve_body_index(args.base_index, args.base_body, body_names, body_count, "base")
    left_foot_index = resolve_body_index(
        args.left_foot_index, args.left_foot_body, body_names, body_count, "left foot"
    )
    right_foot_index = resolve_body_index(
        args.right_foot_index, args.right_foot_body, body_names, body_count, "right foot"
    )

    base_pos = body_pos_w[:, base_index]
    base_lin_vel = body_lin_vel_w[:, base_index]
    base_ang_vel = body_ang_vel_w[:, base_index]
    left_foot_pos = body_pos_w[:, left_foot_index]
    right_foot_pos = body_pos_w[:, right_foot_index]
    left_foot_lin_vel = body_lin_vel_w[:, left_foot_index]
    right_foot_lin_vel = body_lin_vel_w[:, right_foot_index]
    left_foot_ang_vel = body_ang_vel_w[:, left_foot_index]
    right_foot_ang_vel = body_ang_vel_w[:, right_foot_index]

    left_foot_z = left_foot_pos[:, 2] - args.ground_height
    right_foot_z = right_foot_pos[:, 2] - args.ground_height
    base_z = base_pos[:, 2] - args.ground_height
    left_foot_vz = left_foot_lin_vel[:, 2]
    right_foot_vz = right_foot_lin_vel[:, 2]
    left_foot_xy_speed = norm_last_dim(left_foot_lin_vel[:, :2])
    right_foot_xy_speed = norm_last_dim(right_foot_lin_vel[:, :2])
    left_foot_lin_speed = norm_last_dim(left_foot_lin_vel)
    right_foot_lin_speed = norm_last_dim(right_foot_lin_vel)
    left_foot_ang_speed = norm_last_dim(left_foot_ang_vel)
    right_foot_ang_speed = norm_last_dim(right_foot_ang_vel)
    base_lin_speed = norm_last_dim(base_lin_vel)
    base_ang_speed = norm_last_dim(base_ang_vel)
    joint_vel_norm = norm_last_dim(joint_vel)

    joint_pos_step = norm_last_dim(np.diff(joint_pos, axis=0))
    joint_vel_step = norm_last_dim(np.diff(joint_vel, axis=0))
    body_lin_vel_step = np.diff(body_lin_vel_w, axis=0)
    body_ang_vel_step = np.diff(body_ang_vel_w, axis=0)
    body_lin_vel_step_norm = np.linalg.norm(body_lin_vel_step, axis=-1).max(axis=-1)
    body_ang_vel_step_norm = np.linalg.norm(body_ang_vel_step, axis=-1).max(axis=-1)
    base_accel_norm = norm_last_dim(finite_diff(base_lin_vel, fps))

    summary_lines = [
        f"Motion file: {args.npz}",
        f"Frames: {frame_count}",
        f"FPS: {fps:.6f}",
        f"Duration: {duration:.3f} s",
        f"Joint pos shape: {joint_pos.shape}",
        f"Joint vel shape: {joint_vel.shape}",
        f"Body pos shape: {body_pos_w.shape}",
        f"Body quat shape: {body_quat_w.shape}",
        f"Body lin vel shape: {body_lin_vel_w.shape}",
        f"Body ang vel shape: {body_ang_vel_w.shape}",
        "",
        f"Body order assumption: {body_names}",
        f"Base body: {body_names[base_index]} (index {base_index})",
        f"Left foot body: {body_names[left_foot_index]} (index {left_foot_index})",
        f"Right foot body: {body_names[right_foot_index]} (index {right_foot_index})",
        "",
        format_stats("Base z", base_z, "m"),
        format_stats("Left foot z", left_foot_z, "m"),
        format_stats("Right foot z", right_foot_z, "m"),
        format_stats("Base linear speed", base_lin_speed, "m/s"),
        format_stats("Base angular speed", base_ang_speed, "rad/s"),
        format_stats("Left foot linear speed", left_foot_lin_speed, "m/s"),
        format_stats("Right foot linear speed", right_foot_lin_speed, "m/s"),
        format_stats("Left foot angular speed", left_foot_ang_speed, "rad/s"),
        format_stats("Right foot angular speed", right_foot_ang_speed, "rad/s"),
        format_stats("Joint velocity norm", joint_vel_norm, "rad/s"),
        format_stats("Base linear acceleration norm", base_accel_norm, "m/s^2"),
        "",
        "Left foot penetration summary:",
        *summarize_thresholds(left_foot_z, args.thresholds),
        "",
        "Right foot penetration summary:",
        *summarize_thresholds(right_foot_z, args.thresholds),
        "",
        f"Max frame-to-frame joint pos delta norm: {np.max(joint_pos_step):.4f} rad",
        f"Max frame-to-frame joint vel delta norm: {np.max(joint_vel_step):.4f} rad/s",
        f"Max frame-to-frame body linear-velocity delta norm: {np.max(body_lin_vel_step_norm):.4f} m/s",
        f"Max frame-to-frame body angular-velocity delta norm: {np.max(body_ang_vel_step_norm):.4f} rad/s",
        f"Most penetrated left-foot frame: {int(np.argmin(left_foot_z))} (z={np.min(left_foot_z):.4f} m)",
        f"Most penetrated right-foot frame: {int(np.argmin(right_foot_z))} (z={np.min(right_foot_z):.4f} m)",
        f"Largest joint-velocity frame jump index: {top_k_frames(joint_vel_step, 1)[0] if len(joint_vel_step) else 0}",
        f"Largest body linear-velocity frame jump index: {top_k_frames(body_lin_vel_step_norm, 1)[0] if len(body_lin_vel_step_norm) else 0}",
        f"Largest body angular-velocity frame jump index: {top_k_frames(body_ang_vel_step_norm, 1)[0] if len(body_ang_vel_step_norm) else 0}",
    ]
    report = "\n".join(summary_lines)
    print(report)

    output_dir = args.output_dir or args.npz.with_name(f"{args.npz.stem}_inspect")
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.txt").write_text(report + "\n")

    plots = {
        "foot_height.png": (
            [
                ("left foot z", left_foot_z),
                ("right foot z", right_foot_z),
                ("base z", base_z),
            ],
            "Foot and base heights",
            "Height above ground [m]",
            [(0.0, "ground")],
        ),
        "foot_vertical_velocity.png": (
            [("left foot vz", left_foot_vz), ("right foot vz", right_foot_vz)],
            "Foot vertical velocity",
            "Vertical velocity [m/s]",
            [(0.0, "zero")],
        ),
        "foot_xy_speed.png": (
            [("left foot xy speed", left_foot_xy_speed), ("right foot xy speed", right_foot_xy_speed)],
            "Foot horizontal speed",
            "Speed [m/s]",
            None,
        ),
        "base_velocity_norms.png": (
            [("base linear speed", base_lin_speed), ("base angular speed", base_ang_speed)],
            "Base velocity norms",
            "Norm",
            None,
        ),
        "foot_angular_speed.png": (
            [("left foot ang speed", left_foot_ang_speed), ("right foot ang speed", right_foot_ang_speed)],
            "Foot angular speed norms",
            "Angular speed [rad/s]",
            None,
        ),
        "joint_velocity_norm.png": (
            [("joint velocity norm", joint_vel_norm)],
            "Joint velocity norm",
            "Norm [rad/s]",
            None,
        ),
        "frame_jump_norms.png": (
            [
                ("joint pos step", np.pad(joint_pos_step, (1, 0))),
                ("joint vel step", np.pad(joint_vel_step, (1, 0))),
                ("body lin vel step", np.pad(body_lin_vel_step_norm, (1, 0))),
                ("body ang vel step", np.pad(body_ang_vel_step_norm, (1, 0))),
            ],
            "Frame-to-frame jump norms",
            "Jump norm",
            None,
        ),
    }

    if not args.no_plots:
        for file_name, (series, title, ylabel, extra_hlines) in plots.items():
            save_line_plot(output_dir / file_name, time_axis, series, title, ylabel, extra_hlines)

    if args.show:
        import matplotlib.pyplot as plt

        save_line_plot(
            output_dir / "_preview.png",
            time_axis,
            [("left foot z", left_foot_z), ("right foot z", right_foot_z), ("base z", base_z)],
            "Foot and base heights",
            "Height above ground [m]",
            [(0.0, "ground")],
        )
        preview = plt.imread(output_dir / "_preview.png")
        plt.figure(figsize=(12, 5))
        plt.imshow(preview)
        plt.axis("off")
        plt.show()


if __name__ == "__main__":
    main()
