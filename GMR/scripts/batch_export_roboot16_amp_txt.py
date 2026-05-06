from __future__ import annotations

import argparse
import pathlib
import subprocess
import sys


GMR_ROOT = pathlib.Path(__file__).resolve().parents[1]
DEFAULT_EXPORTER = pathlib.Path(__file__).resolve().with_name("export_roboot16_amp_expert.py")
DEFAULT_ROBOT_XML = GMR_ROOT / "Roboot1.6" / "xml" / "roboot16_1.xml"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Batch export Roboot16 motion PKLs to AMP expert txt files."
    )
    parser.add_argument(
        "input_dir",
        type=pathlib.Path,
        help="Directory containing input PKL files.",
    )
    parser.add_argument(
        "--output-dir",
        type=pathlib.Path,
        required=True,
        help="Directory for exported txt files.",
    )
    parser.add_argument(
        "--debug-pkl-dir",
        type=pathlib.Path,
        default=None,
        help="Optional directory for exported debug PKLs with foot positions.",
    )
    parser.add_argument(
        "--robot-xml",
        type=pathlib.Path,
        default=DEFAULT_ROBOT_XML,
        help="MuJoCo XML used for FK/export.",
    )
    parser.add_argument(
        "--python",
        type=str,
        default=sys.executable,
        help="Python executable used to run export_roboot16_amp_expert.py.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing txt/debug outputs.",
    )
    args = parser.parse_args()

    input_dir = args.input_dir.resolve()
    output_dir = args.output_dir.resolve()
    robot_xml = args.robot_xml.resolve()
    exporter = DEFAULT_EXPORTER.resolve()

    if not input_dir.is_dir():
        raise NotADirectoryError(f"Input dir not found: {input_dir}")
    if not exporter.is_file():
        raise FileNotFoundError(f"Exporter script not found: {exporter}")
    if not robot_xml.is_file():
        raise FileNotFoundError(f"Robot XML not found: {robot_xml}")

    output_dir.mkdir(parents=True, exist_ok=True)
    if args.debug_pkl_dir is not None:
        debug_dir = args.debug_pkl_dir.resolve()
        debug_dir.mkdir(parents=True, exist_ok=True)
    else:
        debug_dir = None

    input_files = sorted(input_dir.glob("*.pkl"))
    if not input_files:
        raise FileNotFoundError(f"No PKL files found in: {input_dir}")

    for input_pkl in input_files:
        output_txt = output_dir / f"{input_pkl.stem}.txt"
        output_debug = debug_dir / f"{input_pkl.stem}_debug.pkl" if debug_dir is not None else None

        if output_txt.exists() and not args.overwrite:
            print(f"[skip] txt exists: {output_txt}")
            continue

        cmd = [
            args.python,
            str(exporter),
            "--input_pkl",
            str(input_pkl),
            "--output_txt",
            str(output_txt),
            "--robot",
            "roboot16",
            "--robot_xml",
            str(robot_xml),
        ]
        if output_debug is not None:
            if output_debug.exists() and args.overwrite:
                output_debug.unlink()
            cmd.extend(["--output_debug_pkl", str(output_debug)])

        print("[run]", " ".join(cmd))
        subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
