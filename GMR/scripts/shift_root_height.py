import argparse
import pickle
import shutil
from pathlib import Path

import numpy as np


def _load_pkl(path: Path):
    with path.open("rb") as f:
        return pickle.load(f)


def _save_pkl(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        pickle.dump(data, f)


def _shift_motion(data, dz: float):
    if not isinstance(data, dict):
        raise TypeError("PKL content must be a dict.")
    if "root_pos" not in data:
        raise KeyError("PKL dict must contain 'root_pos'.")

    root_pos = np.asarray(data["root_pos"])
    if root_pos.ndim != 2 or root_pos.shape[1] != 3:
        raise ValueError("'root_pos' must have shape (T, 3).")

    before = {
        "min": float(root_pos[:, 2].min()),
        "max": float(root_pos[:, 2].max()),
        "mean": float(root_pos[:, 2].mean()),
    }

    data["root_pos"] = root_pos.copy()
    data["root_pos"][:, 2] += dz

    after_root_pos = np.asarray(data["root_pos"])
    after = {
        "min": float(after_root_pos[:, 2].min()),
        "max": float(after_root_pos[:, 2].max()),
        "mean": float(after_root_pos[:, 2].mean()),
    }
    return data, before, after


def _default_output_path(input_path: Path, suffix: str) -> Path:
    if input_path.suffix == ".pkl":
        return input_path.with_name(f"{input_path.stem}{suffix}.pkl")
    return input_path.with_name(f"{input_path.name}{suffix}")


def _process_file(
    input_path: Path,
    output_path: Path,
    dz: float,
    inplace: bool,
    backup_suffix: str,
):
    data = _load_pkl(input_path)
    data, before, after = _shift_motion(data, dz)

    if inplace:
        backup_path = input_path.with_name(f"{input_path.name}{backup_suffix}")
        shutil.copy2(input_path, backup_path)
        _save_pkl(input_path, data)
        written_path = input_path
    else:
        _save_pkl(output_path, data)
        written_path = output_path

    print(f"[OK] {input_path}")
    if inplace:
        print(f"  backup: {backup_path}")
    print(f"  output: {written_path}")
    print(
        "  root_z: "
        f"min {before['min']:.6f} -> {after['min']:.6f}, "
        f"max {before['max']:.6f} -> {after['max']:.6f}, "
        f"mean {before['mean']:.6f} -> {after['mean']:.6f}"
    )


def main():
    parser = argparse.ArgumentParser(
        description="Shift root_pos[:, 2] for GMR motion PKL files."
    )
    parser.add_argument(
        "input_path",
        type=Path,
        help="Input .pkl file or a folder containing .pkl files.",
    )
    parser.add_argument(
        "--dz",
        type=float,
        required=True,
        help="Global z offset added to root_pos[:, 2]. Example: 0.185",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output .pkl path for single-file mode, or output folder for directory mode.",
    )
    parser.add_argument(
        "--inplace",
        action="store_true",
        help="Modify source file(s) in place. A backup file will be created first.",
    )
    parser.add_argument(
        "--backup-suffix",
        type=str,
        default=".bak",
        help="Backup suffix used with --inplace. Default: .bak",
    )
    parser.add_argument(
        "--suffix",
        type=str,
        default="_zshift",
        help="Suffix for auto-generated output filenames when not using --inplace.",
    )
    args = parser.parse_args()

    input_path = args.input_path.resolve()
    if not input_path.exists():
        raise FileNotFoundError(f"Input path does not exist: {input_path}")

    if input_path.is_file():
        if args.inplace:
            _process_file(
                input_path=input_path,
                output_path=input_path,
                dz=args.dz,
                inplace=True,
                backup_suffix=args.backup_suffix,
            )
        else:
            output_path = args.output.resolve() if args.output else _default_output_path(
                input_path, args.suffix
            )
            _process_file(
                input_path=input_path,
                output_path=output_path,
                dz=args.dz,
                inplace=False,
                backup_suffix=args.backup_suffix,
            )
        return

    pkl_files = sorted(input_path.glob("*.pkl"))
    if not pkl_files:
        raise FileNotFoundError(f"No .pkl files found in folder: {input_path}")

    if args.inplace:
        for pkl_path in pkl_files:
            _process_file(
                input_path=pkl_path,
                output_path=pkl_path,
                dz=args.dz,
                inplace=True,
                backup_suffix=args.backup_suffix,
            )
    else:
        output_dir = args.output.resolve() if args.output else input_path.parent / f"{input_path.name}_zshifted"
        output_dir.mkdir(parents=True, exist_ok=True)
        for pkl_path in pkl_files:
            output_path = output_dir / pkl_path.name
            _process_file(
                input_path=pkl_path,
                output_path=output_path,
                dz=args.dz,
                inplace=False,
                backup_suffix=args.backup_suffix,
            )


if __name__ == "__main__":
    main()
