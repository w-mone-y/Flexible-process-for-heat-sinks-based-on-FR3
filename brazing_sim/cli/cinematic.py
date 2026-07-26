"""Launch the high-fidelity visual edition without replacing the standard cell."""

from __future__ import annotations

import sys

from brazing_sim.cli.line import main
from brazing_sim.paths import CINEMATIC_SCENE_PATH

CINEMATIC_XML = CINEMATIC_SCENE_PATH


def cinematic_args(argv: list[str]) -> list[str]:
    """Select the cinematic scene unless the caller explicitly overrides XML."""

    values = list(argv)
    if "--xml" not in values:
        values[0:0] = ["--xml", str(CINEMATIC_XML)]
    return values


if __name__ == "__main__":
    raise SystemExit(main(cinematic_args(sys.argv[1:])))
