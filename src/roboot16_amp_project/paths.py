from __future__ import annotations

from pathlib import Path


PACKAGE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_ROOT.parents[1]
DATA_DIR = PROJECT_ROOT / "data"
AMP_NPY_DIR = DATA_DIR / "amp_npy"
AMP_EXPERT_DIR = DATA_DIR / "amp_expert"
