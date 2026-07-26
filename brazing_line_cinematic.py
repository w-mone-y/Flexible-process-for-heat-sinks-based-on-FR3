"""Compatibility launcher for the high-fidelity visual edition."""

import sys

from brazing_sim.cli.cinematic import *  # noqa: F401,F403
from brazing_sim.cli.cinematic import cinematic_args
from brazing_sim.cli.line import main

if __name__ == "__main__":
    raise SystemExit(main(cinematic_args(sys.argv[1:])))
