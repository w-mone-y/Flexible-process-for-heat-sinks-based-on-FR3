"""Rebuild README assets for the V2-Serial control group.

The two underlying capture scripts run the real V2 application.  The
environment variables are set before importing them so every output receives
the ``v2_serial_`` prefix and cannot overwrite flexible-mode assets.
"""

from __future__ import annotations

import os

os.environ.setdefault("README_BENCHMARK_MODE", "SERIAL")

from capture_readme_gifs import main as capture_animations
from capture_v2_readme import main as capture_stills


def main() -> int:
    capture_stills()
    capture_animations()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
