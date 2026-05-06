from __future__ import annotations

import json
from pathlib import Path
from typing import Any


RAW_MOTION_FORMAT = "roboot16_motion_visualization_v1"


def load_motion_txt(path: str | Path) -> dict[str, Any]:
    """Load a JSON-based motion txt file."""
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_motion_txt(path: str | Path, payload: dict[str, Any]) -> None:
    """Save a JSON-based motion txt file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
        f.write("\n")


def resolve_motion_dataset_path(dataset_path_root: str | Path, dataset_name: str) -> Path:
    """Resolve a dataset name to an existing file path.

    The loader accepts:
    - exact paths
    - names without suffix, resolved under ``dataset_path_root``
    - names ending with ``.npy``, ``.txt`` or ``.json``
    """
    dataset_path_root = Path(dataset_path_root)
    candidate = Path(dataset_name)

    if candidate.exists():
        return candidate

    if candidate.is_absolute():
        raise FileNotFoundError(f"Dataset path does not exist: {candidate}")

    direct = dataset_path_root / candidate
    if direct.exists():
        return direct

    for suffix in (".npy", ".txt", ".json"):
        path = dataset_path_root / f"{dataset_name}{suffix}"
        if path.exists():
            return path

    raise FileNotFoundError(
        f"Unable to resolve dataset '{dataset_name}' inside '{dataset_path_root}'. "
        "Expected an existing .npy, .txt, or .json motion file."
    )
