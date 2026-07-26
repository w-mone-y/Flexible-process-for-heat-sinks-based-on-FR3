"""Compatibility launcher for strict YAML-driven flexible orders."""

from brazing_sim.cli.flexible_order import *  # noqa: F401,F403
from brazing_sim.cli.flexible_order import main

if __name__ == "__main__":
    raise SystemExit(main())
