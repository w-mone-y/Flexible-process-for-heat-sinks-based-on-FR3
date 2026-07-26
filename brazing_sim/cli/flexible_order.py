"""CLI for strict YAML-driven flexible brazing orders."""

from __future__ import annotations

import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import shutil
import sys

from brazing_sim.flexible import build_process_plan, load_order_plans, validate_process_plan
from brazing_sim.paths import ARTIFACTS_DIR, CONFIG_DIR, PROJECT_ROOT

ROOT = PROJECT_ROOT


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="订单参数驱动的柔性钎焊仿真")
    orders = parser.add_mutually_exclusive_group(required=True)
    orders.add_argument("--order", help="单订单YAML文件")
    orders.add_argument("--orders", help="多订单YAML文件")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--headless", action="store_true", help="无Viewer执行完整订单")
    mode.add_argument("--dry-run", action="store_true", help="只生成并校验计划")
    parser.add_argument("--fast", action="store_true", help="跳过机器人行程但保留工艺状态")
    parser.add_argument("--no-ui", action="store_true")
    parser.add_argument("--no-terminal-commands", action="store_true")
    parser.add_argument("--port", type=int, default=8766)
    parser.add_argument("--max-sim-time", type=float, default=1800.0)
    parser.add_argument("--scheduler", choices=("fixed", "dynamic"), default=None)
    parser.add_argument("--fault-scenario", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--speed", type=float, default=1.0)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--compare", action="store_true", help="以相同输入运行fixed/dynamic并生成对比")
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
    if args.orders:
        try:
            plans = load_order_plans(args.orders)
            for plan in plans:
                validate_process_plan(plan)
        except Exception as exc:
            print(f"[柔性订单错误] {exc}", file=sys.stderr)
            return 2
        if args.dry_run:
            for plan in plans:
                print(chinese_summary(plan))
            print(json.dumps([plan.summary() for plan in plans], ensure_ascii=False, indent=2))
            return 0
        if not args.headless and not args.compare and not args.fault_scenario:
            runtime = [
                str(ROOT / "brazing_line.py"),
                "--orders-file",
                str(Path(args.orders).expanduser().resolve()),
                "--port",
                str(args.port),
                "--max-sim-time",
                str(args.max_sim_time),
            ]
            if args.fast:
                runtime.append("--fast")
            if args.no_ui:
                runtime.append("--no-ui")
            if args.no_terminal_commands:
                runtime.append("--no-terminal-commands")
            if sys.platform == "darwin" and os.environ.get("BRAZING_MJPYTHON_CHILD") != "1":
                executable = shutil.which("mjpython")
                if executable is None:
                    print("[柔性订单错误] macOS图形模式需要mjpython，但当前PATH中未找到。", file=sys.stderr)
                    return 2
                environment = os.environ.copy()
                environment["BRAZING_MJPYTHON_CHILD"] = "1"
                os.execve(executable, [executable, *runtime], environment)
            from brazing_line import main as run_brazing_line

            return run_brazing_line(runtime[1:])
        return _run_v2(args, plans)

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

    if args.scheduler is not None or args.compare or args.fault_scenario:
        return _run_v2(args, (plan,))

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


def _run_runtime_once(
    args: argparse.Namespace,
    plans: tuple[object, ...],
    scheduler: str,
    output_dir: Path,
) -> tuple[dict, Path]:
    from brazing_sim.experiments import ExperimentReporter, MetricsCollector
    from brazing_sim.manufacturing_config import load_fault_scenario
    from brazing_sim.manufacturing_runtime import ManufacturingRuntime

    runtime = ManufacturingRuntime(scheduler_mode=scheduler, flexible_cell=True)
    metrics = MetricsCollector()
    runtime.events.subscribe(None, metrics.handle_event)
    for plan in plans:
        runtime.submit_plan(plan, now=0.0)
    if args.fault_scenario:
        runtime.set_fault_scenario(load_fault_scenario(args.fault_scenario))
    step = 0.05
    now = 0.0
    while now <= float(args.max_sim_time) and not runtime.terminal:
        runtime.tick(now)
        now += step
    if not runtime.terminal:
        print(f"[V2运行错误] 超过最大仿真时间{args.max_sim_time:.1f}s", file=sys.stderr)
        return metrics.calculate(runtime, now), output_dir
    values = metrics.calculate(runtime, now)
    reporter = ExperimentReporter(output_dir)
    configs = [
        args.orders or args.order,
        CONFIG_DIR / "scheduler.yaml",
        CONFIG_DIR / "resources.yaml",
    ]
    if args.fault_scenario:
        configs.append(args.fault_scenario)
    reporter.export(
        runtime,
        values,
        config_files=configs,
        metadata={
            "seed": args.seed,
            "simulation_speed": args.speed,
            "headless": bool(args.headless),
        },
    )
    return values, output_dir


def _run_v2(args: argparse.Namespace, plans: tuple[object, ...]) -> int:
    from brazing_sim.experiments import compare_experiments

    base = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir
        else ARTIFACTS_DIR / "experiments" / datetime.now().strftime("run_%Y%m%d_%H%M%S_%f")
    )
    if args.compare:
        fixed, fixed_dir = _run_runtime_once(args, plans, "fixed", base / "fixed")
        dynamic, dynamic_dir = _run_runtime_once(args, plans, "dynamic", base / "dynamic")
        comparison = compare_experiments(fixed, dynamic)
        base.mkdir(parents=True, exist_ok=True)
        (base / "comparison.json").write_text(
            json.dumps(comparison, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        lines = ["# Fixed / Dynamic 对比", ""]
        for name, values in comparison.items():
            percent = values["percent_change"]
            text = "-" if percent is None else f"{percent:+.2f}%"
            lines.append(f"- {name}: {values['fixed']:.6g} → {values['dynamic']:.6g}（{text}）")
        (base / "comparison.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(
            json.dumps(
                {"fixed": str(fixed_dir), "dynamic": str(dynamic_dir), "comparison": comparison},
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0
    scheduler = args.scheduler or "dynamic"
    values, directory = _run_runtime_once(args, plans, scheduler, base)
    print(json.dumps({"output": str(directory), "metrics": values}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
