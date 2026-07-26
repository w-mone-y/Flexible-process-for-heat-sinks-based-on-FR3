"""Compatibility launcher for the standard FR3 brazing-line simulation."""

from brazing_sim.cli.line import *  # noqa: F401,F403
from brazing_sim.cli.line import main

if __name__ == "__main__":
    raise SystemExit(main())
