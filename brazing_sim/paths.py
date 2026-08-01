"""Canonical filesystem locations for the brazing-line repository."""

from __future__ import annotations

import sys
from pathlib import Path

_SOURCE_ROOT = Path(__file__).resolve().parents[1]
_INSTALLED_ROOT = Path(sys.prefix) / "brazing_sim"


def _has_runtime_resources(root: Path) -> bool:
    return (root / "config").is_dir() and (root / "scenes").is_dir() and (root / "assets").is_dir()


PROJECT_ROOT = next(
    (root for root in (_SOURCE_ROOT, _INSTALLED_ROOT) if _has_runtime_resources(root)),
    _SOURCE_ROOT,
)
ASSETS_DIR = PROJECT_ROOT / "assets"
CONFIG_DIR = PROJECT_ROOT / "config"
SCENES_DIR = PROJECT_ROOT / "scenes"
PRODUCTION_SCENES_DIR = SCENES_DIR / "production"
DEFAULT_SCENE_PATH = PRODUCTION_SCENES_DIR / "brazing_line.xml"
CINEMATIC_SCENE_PATH = PRODUCTION_SCENES_DIR / "brazing_line_cinematic.xml"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"

__all__ = [
    "ARTIFACTS_DIR",
    "ASSETS_DIR",
    "CINEMATIC_SCENE_PATH",
    "CONFIG_DIR",
    "DEFAULT_SCENE_PATH",
    "PRODUCTION_SCENES_DIR",
    "PROJECT_ROOT",
    "SCENES_DIR",
]
