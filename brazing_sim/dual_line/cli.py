"""Command-line entry point for the independent dual-install V2 line."""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from typing import Sequence

from brazing_sim.paths import PRODUCTION_SCENES_DIR, PROJECT_ROOT
from brazing_sim.ui import run_ui_client

from .application import V2BrazingApplication

DEFAULT_V2_XML = PRODUCTION_SCENES_DIR / "brazing_line_v2.xml"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="V2 dual-installation FR3 flexible brazing line")
    parser.add_argument("--ui-client", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8767)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--no-ui", action="store_true")
    parser.add_argument("--show-mujoco-ui", action="store_true")
    parser.add_argument(
        "--orders",
        default="",
        help="comma-separated A/B/C/D presets; omitted viewer sessions start idle",
    )
    parser.add_argument("--fast", action="store_true", help="shorten upstream demo operations")
    parser.add_argument(
        "--optimizer",
        choices=("rule", "genetic"),
        default="rule",
        help="V2订单释放策略（默认规则调度）",
    )
    parser.add_argument("--genetic-seed", type=int, default=42, help="V2遗传算法随机种子")
    parser.add_argument("--max-sim-time", type=float, default=300.0)
    parser.add_argument("--dt", type=float, default=0.05)
    parser.add_argument("--xml", default=str(DEFAULT_V2_XML))
    args = parser.parse_args(argv)
    args.line_profile = "V2_DUAL_INSTALL"
    if args.max_sim_time <= 0 or args.dt <= 0:
        parser.error("--max-sim-time and --dt must be positive")
    presets = tuple(value.strip().upper() for value in args.orders.split(",") if value.strip())
    if any(preset not in {"A", "B", "C", "D"} for preset in presets):
        parser.error("--orders must contain only A, B, C and D")
    args.order_presets = presets
    return args


def run_headless(args: argparse.Namespace) -> int:
    application = V2BrazingApplication(args)
    try:
        application.submit_cli_orders()
        return application.run_headless()
    finally:
        application.close()


def run_viewer(args: argparse.Namespace) -> int:
    application = V2BrazingApplication(args)
    try:
        application.submit_cli_orders()
        application.start_services()
        return application.run_viewer()
    finally:
        application.close()


def _respawn_with_mjpython(argv: Sequence[str] | None) -> int | None:
    if sys.platform != "darwin" or os.environ.get("BRAZING_V2_MJPYTHON_CHILD") == "1":
        return None
    executable = shutil.which("mjpython")
    if executable is None:
        return None
    environment = dict(os.environ)
    environment["BRAZING_V2_MJPYTHON_CHILD"] = "1"
    command = [executable, str(PROJECT_ROOT / "brazing_line_v2.py"), *(argv or sys.argv[1:])]
    return subprocess.call(command, cwd=PROJECT_ROOT, env=environment)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.ui_client:
        return run_ui_client(args)
    if not args.headless:
        child_result = _respawn_with_mjpython(argv)
        if child_result is not None:
            return child_result
    return run_headless(args) if args.headless else run_viewer(args)


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["DEFAULT_V2_XML", "main", "parse_args", "run_headless", "run_viewer"]
