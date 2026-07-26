"""Canonical filesystem locations for the brazing-line repository."""

from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
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
