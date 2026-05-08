import argparse
import copy
import os
import pathlib
import pickle
import re
import sys

import numpy as np

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def parse_segment(segment_text):
    match = re.fullmatch(r"\s*(\d+)\s*[:-]\s*(\d+)\s*", segment_text)
    if match is None:
        raise ValueError(f"Invalid segment format: {segment_text!r}. Expected START:END")
    start_frame = int(match.group(1))
    end_frame = int(match.group(2))
    if start_frame <= 0 or end_frame <= 0:
        raise ValueError(f"Frame numbers must be positive: {segment_text!r}")
    if end_frame < start_frame:
        raise ValueError(f"End frame must be >= start frame: {segment_text!r}")
    return start_frame, end_frame


def slice_value(value, start_idx, end_idx_exclusive, total_frames):
    if isinstance(value, np.ndarray) and value.shape[:1] == (total_frames,):
        return value[start_idx:end_idx_exclusive].copy()
    if isinstance(value, list) and len(value) == total_frames:
        return copy.deepcopy(value[start_idx:end_idx_exclusive])
    return copy.deepcopy(value)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to input robot motion .pkl")
    parser.add_argument("--output_dir", required=True, help="Directory to save split .pkl files")
    parser.add_argument(
        "--segment",
        action="append",
        required=True,
        help="Frame range in 1-based inclusive format, e.g. 3362:3415",
    )
    parser.add_argument(
        "--output_name",
        type=str,
        default=None,
        help="Custom output file name (only valid when splitting a single segment)",
    )
    args = parser.parse_args()

    input_path = pathlib.Path(args.input)
    output_dir = pathlib.Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with input_path.open("rb") as f:
        motion_data = pickle.load(f)

    total_frames = int(len(motion_data["root_pos"]))
    base_name = input_path.stem

    print(f"Loaded {input_path}")
    print(f"Total frames: {total_frames}")

    for segment_id, segment_text in enumerate(args.segment, start=1):
        start_frame_1b, end_frame_1b = parse_segment(segment_text)
        start_idx = start_frame_1b - 1
        end_idx_exclusive = end_frame_1b

        if start_idx >= total_frames or end_idx_exclusive > total_frames:
            raise IndexError(
                f"Segment {segment_text!r} exceeds total frames {total_frames}"
            )

        segment_data = {}
        for key, value in motion_data.items():
            segment_data[key] = slice_value(value, start_idx, end_idx_exclusive, total_frames)

        # 支持自定义输出名，仅在只切一个片段时生效
        if args.output_name is not None:
            if len(args.segment) == 1:
                output_path = output_dir / args.output_name
            else:
                print("[WARN] --output_name 仅在切分单个片段时生效，已忽略。")
                output_path = output_dir / f"{base_name}_seg{segment_id:02d}_{start_frame_1b:04d}_{end_frame_1b:04d}.pkl"
        else:
            output_path = output_dir / f"{base_name}_seg{segment_id:02d}_{start_frame_1b:04d}_{end_frame_1b:04d}.pkl"

        output_stem = output_path.stem
        source_motion_file = motion_data.get("motion_file", input_path.stem)
        clip_info = {
            "source_file": str(input_path),
            "source_motion_file": source_motion_file,
            "output_file": str(output_path),
            "output_motion_file": output_stem,
            "segment_id": segment_id,
            "segment_range_1based": [start_frame_1b, end_frame_1b],
            "segment_range_text": f"{start_frame_1b}:{end_frame_1b}",
            "segment_start_frame_1based": start_frame_1b,
            "segment_end_frame_1based": end_frame_1b,
            "segment_length": end_idx_exclusive - start_idx,
        }

        segment_data["motion_file"] = output_stem
        segment_data["clip_info"] = clip_info

        meta = copy.deepcopy(segment_data.get("meta", {}))
        if isinstance(meta, dict):
            meta.update(clip_info)
        segment_data["meta"] = meta

        with output_path.open("wb") as f:
            pickle.dump(segment_data, f)

        print(
            f"[{segment_id}] {output_path.name}: "
            f"frames {start_frame_1b}-{end_frame_1b} "
            f"({end_idx_exclusive - start_idx} frames)"
        )


if __name__ == "__main__":
    main()
