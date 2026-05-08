import argparse
import pathlib
import pickle
import shutil
import sys

import numpy as np


REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def load_motion_data(path: pathlib.Path) -> dict:
    with path.open("rb") as f:
        return pickle.load(f)


def estimate_root_vel(root_pos: np.ndarray, fps: float) -> np.ndarray:
    num_frames = len(root_pos)
    vel = np.zeros((num_frames, 3), dtype=np.float32)
    if num_frames <= 1 or fps <= 0:
        return vel

    delta = np.diff(root_pos, axis=0) * float(fps)
    vel[:-1] = delta
    vel[-1] = delta[-1]
    return vel


def compute_speed_stats(motion_data: dict) -> dict[str, float]:
    fps = float(motion_data["fps"])
    root_pos = np.asarray(motion_data["root_pos"], dtype=np.float32)
    root_vel = motion_data.get("root_vel")
    if root_vel is None:
        root_vel = estimate_root_vel(root_pos, fps)
    else:
        root_vel = np.asarray(root_vel, dtype=np.float32)

    speed_xy = np.linalg.norm(root_vel[:, :2], axis=1)

    return {
        "frames": int(len(root_pos)),
        "mean_speed_xy": float(np.mean(speed_xy)),
        "max_speed_xy": float(np.max(speed_xy)) if len(speed_xy) > 0 else 0.0,
    }


def classify_speed_bucket(mean_speed_xy: float, max_speed_xy: float) -> str:
    if round(mean_speed_xy, 1) == 0.0 and max_speed_xy < 0.2:
        return "stand"
    if mean_speed_xy < 2.0:
        return "walk"
    if mean_speed_xy < 3.0:
        return "run"
    return "sprint"


def format_speed_token(mean_speed_xy: float) -> str:
    return f"{mean_speed_xy:.1f}".replace(".", "p")


def is_mirrored_name(path: pathlib.Path) -> bool:
    stem_lower = path.stem.lower()
    return stem_lower.endswith("_mirrored") or stem_lower.endswith("_mirror")


def copy_into_speed_buckets(
    pkl_files: list[pathlib.Path],
    stats_by_path: dict[pathlib.Path, dict[str, float]],
    output_dir: pathlib.Path,
) -> None:
    bucket_counts = {"stand": 0, "walk": 0, "run": 0, "sprint": 0}
    for bucket in bucket_counts:
        (output_dir / bucket).mkdir(parents=True, exist_ok=True)

    for path in pkl_files:
        stats = stats_by_path[path]
        bucket = classify_speed_bucket(stats["mean_speed_xy"], stats["max_speed_xy"])
        bucket_counts[bucket] += 1
        speed_token = format_speed_token(stats["mean_speed_xy"])
        mirror_suffix = "_mirror" if is_mirrored_name(path) else ""
        output_name = f"{bucket}{bucket_counts[bucket]}_{speed_token}{mirror_suffix}.pkl"
        output_path = output_dir / bucket / output_name
        shutil.copy2(path, output_path)
        print(f"[copy] {path.name} -> {bucket}/{output_name}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Print average speed stats for all robot motion PKLs in a folder."
    )
    parser.add_argument("input_dir", type=pathlib.Path, help="Directory containing motion PKLs.")
    parser.add_argument(
        "--classify-output-dir",
        type=pathlib.Path,
        default=None,
        help="Optional output directory for copied PKLs grouped into stand/walk/run/sprint folders.",
    )
    args = parser.parse_args()

    input_dir = args.input_dir.resolve()
    if not input_dir.is_dir():
        raise NotADirectoryError(f"Input dir not found: {input_dir}")

    classify_output_dir = args.classify_output_dir.resolve() if args.classify_output_dir is not None else None

    pkl_files = sorted(
        path for path in input_dir.rglob("*.pkl")
        if classify_output_dir is None or classify_output_dir not in path.parents
    )
    if not pkl_files:
        raise FileNotFoundError(f"No PKL files found in: {input_dir}")

    print(f"Found {len(pkl_files)} PKL files in {input_dir}")
    stats_by_path: dict[pathlib.Path, dict[str, float]] = {}
    for path in pkl_files:
        stats = compute_speed_stats(load_motion_data(path))
        stats_by_path[path] = stats
        print(
            f"{path.relative_to(input_dir)}: "
            f"frames={stats['frames']} | "
            f"mean_speed_xy={stats['mean_speed_xy']:.3f} | "
            f"max_speed_xy={stats['max_speed_xy']:.3f}"
        )

    if classify_output_dir is not None:
        classify_output_dir.mkdir(parents=True, exist_ok=True)
        print(f"\nWriting classified PKLs into {classify_output_dir}")
        copy_into_speed_buckets(pkl_files, stats_by_path, classify_output_dir)


if __name__ == "__main__":
    main()
