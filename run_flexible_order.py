"""CLI for strict YAML-driven flexible brazing orders."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import sys

from brazing_sim.flexible import build_process_plan, validate_process_plan

ROOT = Path(__file__).resolve().parent


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="订单参数驱动的柔性钎焊仿真")
    parser.add_argument("--order", required=True, help="订单YAML文件")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--headless", action="store_true", help="无Viewer执行完整订单")
    mode.add_argument("--dry-run", action="store_true", help="只生成并校验计划")
    parser.add_argument("--fast", action="store_true", help="跳过机器人行程但保留工艺状态")
    parser.add_argument("--no-ui", action="store_true")
    parser.add_argument("--no-terminal-commands", action="store_true")
    parser.add_argument("--port", type=int, default=8766)
    parser.add_argument("--max-sim-time", type=float, default=1800.0)
    return parser.parse_args(argv)


def chinese_summary(plan: object) -> str:
    summary = plan.summary()
    layers = ", ".join(str(index + 1) for index in summary["rack_layers"])
    return "\n".join(
        (
            f"订单：{summary['order_id']}（{summary['product_id']} / {summary['preset']}型）",
            f"数量：{summary['quantity']}件；料架层：{layers}",
            f"产品：{summary['fin_count']}片翅片，{summary['path_count']}条钎料路径，"
            f"单条长度{summary['path_length_m']:.3f} m",
            f"工装：{summary['comb_module']}，目标压紧力{summary['clamping_force_n']:.1f} N",
            f"Arm2：喷嘴中心距{summary['nozzle_spacing_m'] * 1000:.1f} mm，"
            f"涂覆速度{summary['material_speed_m_s']:.3f} m/s",
            "校验结果：通过，可进入仿真执行。",
        )
    )


def _runtime_arguments(args: argparse.Namespace, order_path: Path) -> list[str]:
    values = [
        str(ROOT / "brazing_line.py"),
        "--order-file",
        str(order_path),
        "--port",
        str(args.port),
        "--max-sim-time",
        str(args.max_sim_time),
    ]
    for enabled, option in (
        (args.headless, "--headless"),
        (args.fast, "--fast"),
        (args.no_ui, "--no-ui"),
        (args.no_terminal_commands, "--no-terminal-commands"),
    ):
        if enabled:
            values.append(option)
    return values


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    order_path = Path(args.order).expanduser().resolve()
    try:
        plan = build_process_plan(order_path)
        validate_process_plan(plan)
    except Exception as exc:
        print(f"[柔性订单错误] {exc}", file=sys.stderr)
        return 2
    if args.dry_run:
        print(chinese_summary(plan))
        print(json.dumps(plan.summary(), ensure_ascii=False, indent=2))
        return 0

    runtime = _runtime_arguments(args, order_path)
    if sys.platform == "darwin" and not args.headless and os.environ.get("BRAZING_MJPYTHON_CHILD") != "1":
        executable = shutil.which("mjpython")
        if executable is None:
            print("[柔性订单错误] macOS图形模式需要mjpython，但当前PATH中未找到。", file=sys.stderr)
            return 2
        environment = os.environ.copy()
        environment["BRAZING_MJPYTHON_CHILD"] = "1"
        os.execve(executable, [executable, *runtime], environment)

    from brazing_line import main as run_brazing_line

    return run_brazing_line(runtime[1:])


if __name__ == "__main__":
    raise SystemExit(main())
