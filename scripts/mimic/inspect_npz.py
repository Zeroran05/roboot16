from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


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


def load_body_names(args: argparse.Namespace, body_count: int, npz_path: Path, motion_data: np.lib.npyio.NpzFile) -> list[str]:
    if "body_names" in motion_data.files:
        raw = motion_data["body_names"]
        body_names = [str(name) for name in np.asarray(raw).tolist()]
    elif args.body_names_json is not None:
        body_names = json.loads(args.body_names_json.read_text())
    else:
        sidecar_json = npz_path.with_suffix(".body_names.json")
        if sidecar_json.is_file():
            body_names = json.loads(sidecar_json.read_text())
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


def try_resolve_body_index(
    override_index: int | None, body_name: str, body_names: list[str], body_count: int
) -> int | None:
    if override_index is not None:
        if override_index < 0 or override_index >= body_count:
            raise ValueError(f"Body index {override_index} is out of range for {body_count} bodies.")
        return override_index
    if body_name in body_names:
        return body_names.index(body_name)
    return None


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

    body_names = load_body_names(args, body_count, args.npz, motion)
    base_index = try_resolve_body_index(args.base_index, args.base_body, body_names, body_count)
    left_foot_index = try_resolve_body_index(args.left_foot_index, args.left_foot_body, body_names, body_count)
    right_foot_index = try_resolve_body_index(args.right_foot_index, args.right_foot_body, body_names, body_count)

    base_pos = body_pos_w[:, base_index] if base_index is not None else None
    base_lin_vel = body_lin_vel_w[:, base_index] if base_index is not None else None
    base_ang_vel = body_ang_vel_w[:, base_index] if base_index is not None else None
    left_foot_pos = body_pos_w[:, left_foot_index] if left_foot_index is not None else None
    right_foot_pos = body_pos_w[:, right_foot_index] if right_foot_index is not None else None
    left_foot_lin_vel = body_lin_vel_w[:, left_foot_index] if left_foot_index is not None else None
    right_foot_lin_vel = body_lin_vel_w[:, right_foot_index] if right_foot_index is not None else None
    left_foot_ang_vel = body_ang_vel_w[:, left_foot_index] if left_foot_index is not None else None
    right_foot_ang_vel = body_ang_vel_w[:, right_foot_index] if right_foot_index is not None else None

    all_body_z = body_pos_w[:, :, 2] - args.ground_height
    body_min_z = np.min(all_body_z, axis=0)
    body_mean_z = np.mean(all_body_z, axis=0)
    lowest_body_order = np.argsort(body_min_z)

    left_foot_z = left_foot_pos[:, 2] - args.ground_height if left_foot_pos is not None else None
    right_foot_z = right_foot_pos[:, 2] - args.ground_height if right_foot_pos is not None else None
    base_z = base_pos[:, 2] - args.ground_height if base_pos is not None else None
    left_foot_vz = left_foot_lin_vel[:, 2] if left_foot_lin_vel is not None else None
    right_foot_vz = right_foot_lin_vel[:, 2] if right_foot_lin_vel is not None else None
    left_foot_xy_speed = norm_last_dim(left_foot_lin_vel[:, :2]) if left_foot_lin_vel is not None else None
    right_foot_xy_speed = norm_last_dim(right_foot_lin_vel[:, :2]) if right_foot_lin_vel is not None else None
    left_foot_lin_speed = norm_last_dim(left_foot_lin_vel) if left_foot_lin_vel is not None else None
    right_foot_lin_speed = norm_last_dim(right_foot_lin_vel) if right_foot_lin_vel is not None else None
    left_foot_ang_speed = norm_last_dim(left_foot_ang_vel) if left_foot_ang_vel is not None else None
    right_foot_ang_speed = norm_last_dim(right_foot_ang_vel) if right_foot_ang_vel is not None else None
    base_lin_speed = norm_last_dim(base_lin_vel) if base_lin_vel is not None else None
    base_ang_speed = norm_last_dim(base_ang_vel) if base_ang_vel is not None else None
    joint_vel_norm = norm_last_dim(joint_vel)

    joint_pos_step = norm_last_dim(np.diff(joint_pos, axis=0))
    joint_vel_step = norm_last_dim(np.diff(joint_vel, axis=0))
    body_lin_vel_step = np.diff(body_lin_vel_w, axis=0)
    body_ang_vel_step = np.diff(body_ang_vel_w, axis=0)
    body_lin_vel_step_norm = np.linalg.norm(body_lin_vel_step, axis=-1).max(axis=-1)
    body_ang_vel_step_norm = np.linalg.norm(body_ang_vel_step, axis=-1).max(axis=-1)
    base_accel_norm = norm_last_dim(finite_diff(base_lin_vel, fps)) if base_lin_vel is not None else None

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
        f"Body order labels: {body_names}",
        f"Base body: {body_names[base_index]} (index {base_index})" if base_index is not None else "Base body: unresolved",
        f"Left foot body: {body_names[left_foot_index]} (index {left_foot_index})"
        if left_foot_index is not None
        else "Left foot body: unresolved",
        f"Right foot body: {body_names[right_foot_index]} (index {right_foot_index})"
        if right_foot_index is not None
        else "Right foot body: unresolved",
        "",
    ]
    if base_z is not None:
        summary_lines.extend(
            [
                format_stats("Base z", base_z, "m"),
                format_stats("Base linear speed", base_lin_speed, "m/s"),
                format_stats("Base angular speed", base_ang_speed, "rad/s"),
            ]
        )
    if left_foot_z is not None:
        summary_lines.extend(
            [
                format_stats("Left foot z", left_foot_z, "m"),
                format_stats("Left foot linear speed", left_foot_lin_speed, "m/s"),
                format_stats("Left foot angular speed", left_foot_ang_speed, "rad/s"),
            ]
        )
    if right_foot_z is not None:
        summary_lines.extend(
            [
                format_stats("Right foot z", right_foot_z, "m"),
                format_stats("Right foot linear speed", right_foot_lin_speed, "m/s"),
                format_stats("Right foot angular speed", right_foot_ang_speed, "rad/s"),
            ]
        )
    summary_lines.append(format_stats("Joint velocity norm", joint_vel_norm, "rad/s"))
    if base_accel_norm is not None:
        summary_lines.append(format_stats("Base linear acceleration norm", base_accel_norm, "m/s^2"))
    summary_lines.extend(
        [
            "",
            f"Max frame-to-frame joint pos delta norm: {np.max(joint_pos_step):.4f} rad",
            f"Max frame-to-frame joint vel delta norm: {np.max(joint_vel_step):.4f} rad/s",
            f"Max frame-to-frame body linear-velocity delta norm: {np.max(body_lin_vel_step_norm):.4f} m/s",
            f"Max frame-to-frame body angular-velocity delta norm: {np.max(body_ang_vel_step_norm):.4f} rad/s",
            f"Largest joint-velocity frame jump index: {top_k_frames(joint_vel_step, 1)[0] if len(joint_vel_step) else 0}",
            f"Largest body linear-velocity frame jump index: {top_k_frames(body_lin_vel_step_norm, 1)[0] if len(body_lin_vel_step_norm) else 0}",
            f"Largest body angular-velocity frame jump index: {top_k_frames(body_ang_vel_step_norm, 1)[0] if len(body_ang_vel_step_norm) else 0}",
            "",
            "Bodies sorted by minimum z (lowest first):",
        ]
    )
    for rank, body_idx in enumerate(lowest_body_order.tolist(), start=1):
        summary_lines.append(
            f"  {rank:02d}. {body_names[body_idx]} (index {body_idx}): min_z={body_min_z[body_idx]:.4f} m, "
            f"mean_z={body_mean_z[body_idx]:.4f} m"
        )
    if left_foot_z is not None:
        summary_lines.extend(
            [
                "",
                "Left foot penetration summary:",
                *summarize_thresholds(left_foot_z, args.thresholds),
                f"Most penetrated left-foot frame: {int(np.argmin(left_foot_z))} (z={np.min(left_foot_z):.4f} m)",
            ]
        )
    if right_foot_z is not None:
        summary_lines.extend(
            [
                "",
                "Right foot penetration summary:",
                *summarize_thresholds(right_foot_z, args.thresholds),
                f"Most penetrated right-foot frame: {int(np.argmin(right_foot_z))} (z={np.min(right_foot_z):.4f} m)",
            ]
        )
    report = "\n".join(summary_lines)
    print(report)

    output_dir = args.output_dir or args.npz.with_name(f"{args.npz.stem}_inspect")
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.txt").write_text(report + "\n")

    plots = {
        "all_body_z.png": (
            [(body_names[i], all_body_z[:, i]) for i in range(body_count)],
            "All body z trajectories",
            "Height above ground [m]",
            [(0.0, "ground")],
        ),
        "ankle_z.png": (
            [
                ("left ankle z", left_foot_z),
                ("right ankle z", right_foot_z),
            ],
            "Ankle height (z)",
            "Height above ground [m]",
            [(0.0, "ground")],
        ),
        "ankle_xyz.png": (
            [
                ("left ankle x", left_foot_pos[:, 0]),
                ("left ankle y", left_foot_pos[:, 1]),
                ("left ankle z", left_foot_pos[:, 2]),
                ("right ankle x", right_foot_pos[:, 0]),
                ("right ankle y", right_foot_pos[:, 1]),
                ("right ankle z", right_foot_pos[:, 2]),
            ],
            "Ankle world position",
            "Position [m]",
            None,
        ),
        "ankle_vertical_velocity.png": (
            [("left ankle vz", left_foot_vz), ("right ankle vz", right_foot_vz)],
            "Ankle vertical velocity",
            "Vertical velocity [m/s]",
            [(0.0, "zero")],
        ),
    }

    if left_foot_z is None or right_foot_z is None:
        plots.pop("ankle_z.png", None)
        plots.pop("ankle_xyz.png", None)
        plots.pop("ankle_vertical_velocity.png", None)

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
