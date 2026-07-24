"""Launch the high-fidelity visual edition without replacing the standard cell."""

from __future__ import annotations

import sys
from pathlib import Path

from brazing_line import main

ROOT = Path(__file__).resolve().parent
CINEMATIC_XML = ROOT / "brazing_line_cinematic.xml"


def cinematic_args(argv: list[str]) -> list[str]:
    """Select the cinematic scene unless the caller explicitly overrides XML."""

    values = list(argv)
    if "--xml" not in values:
        values[0:0] = ["--xml", str(CINEMATIC_XML)]
    return values


if __name__ == "__main__":
    raise SystemExit(main(cinematic_args(sys.argv[1:])))
