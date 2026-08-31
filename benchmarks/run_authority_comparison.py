#!/usr/bin/env python3
"""Compare TwinShield V2 authority with the deterministic rollback scheduler."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time
from typing import Any, Mapping, Sequence

from brazing_sim.dual_line.application import V2BrazingApplication
from brazing_sim.dual_line.cli import parse_args

try:
    from benchmarks.run_reference_plan import parse_orders
except ModuleNotFoundError:  # direct ``python benchmarks/script.py`` execution
    from run_reference_plan import parse_orders


def parse_modes(value: str) -> tuple[str, ...]:
    modes = tuple(dict.fromkeys(part.strip().upper() for part in value.split(",") if part.strip()))
    allowed = {"AUTHORITY", "FALLBACK"}
    if not modes or any(mode not in allowed for mode in modes):
        raise ValueError("模式只能使用AUTHORITY和FALLBACK")
    return modes


def _percent_improvement(baseline: float | None, improved: float | None) -> float | None:
    if baseline is None or improved is None or baseline <= 0.0:
        return None
    return round(100.0 * (baseline - improved) / baseline, 6)


def compare_runs(
    authority: Mapping[str, Any],
    fallback: Mapping[str, Any],
) -> dict[str, Any]:
    both_completed = bool(authority.get("completed")) and bool(fallback.get("completed"))
    return {
        "both_completed": both_completed,
        "makespan_improvement_pct": (
            _percent_improvement(
                float(fallback["simulation_seconds"]),
                float(authority["simulation_seconds"]),
            )
            if both_completed
            else None
        ),
        "arm1_idle_improvement_pct": (
            _percent_improvement(
                float(fallback["arm1_idle_s"]),
                float(authority["arm1_idle_s"]),
            )
            if both_completed
            else None
        ),
        "throughput_improvement_pct": (
            round(
                100.0
                * (
                    float(authority["throughput_per_sim_hour"])
                    - float(fallback["throughput_per_sim_hour"])
                )
                / float(fallback["throughput_per_sim_hour"]),
                6,
            )
            if both_completed
            and float(fallback["throughput_per_sim_hour"]) > 0.0
            else None
        ),
    }


def run_mode(
    orders: tuple[str, ...],
    *,
    mode: str,
    max_sim_time: float = 2000.0,
    dt: float = 0.05,
) -> dict[str, Any]:
    args = parse_args(
        [
            "--headless",
            "--fast",
            "--no-ui",
            "--orders",
            ",".join(orders),
            "--max-sim-time",
            str(max_sim_time),
            "--dt",
            str(dt),
            "--twinshield-mode",
            mode,
        ]
    )
    application = V2BrazingApplication(args)
    started = time.perf_counter()
    try:
        application.submit_cli_orders()
        while (
            application.controls.running
            and (not application.runtime.complete or not application.scene.transport_settled)
            and max(application.runtime.sim_time, float(application.scene.data.time)) + 1.0e-12
            < max_sim_time
        ):
            application.advance_frame()
        wall_seconds = time.perf_counter() - started
        state = application.runtime.snapshot()
        manufacturing = state["manufacturing"]
        parallel = manufacturing["async_line"]["parallelism"]
        twinshield = manufacturing["twinshield"]
        completed = bool(application.runtime.complete and application.scene.transport_settled)
        simulation_seconds = float(application.runtime.sim_time)
        return {
            "mode": mode,
            "completed": completed,
            "simulation_seconds": round(simulation_seconds, 6),
            "wall_seconds": round(wall_seconds, 6),
            "unit_count": len(orders),
            "throughput_per_sim_hour": (
                round(len(orders) * 3600.0 / simulation_seconds, 6)
                if completed and simulation_seconds > 0.0
                else 0.0
            ),
            "arm1_idle_s": round(float(parallel["arm1_idle_s"]), 6),
            "arm1_utilization": round(float(parallel["arm1_utilization"]), 8),
            "multi_arm_overlap_s": round(float(parallel["multi_arm_overlap_s"]), 6),
            "maximum_parallel_tasks": int(parallel["max_parallel_tasks"]),
            "assignment_count": len(manufacturing["assignments"]),
            "authority_count": int(twinshield["authority_count"]),
            "fallback_count": int(twinshield["fallback_count"]),
            "last_fallback_reason": str(twinshield["last_fallback_reason"]),
            "decision_latency_ms": dict(twinshield["decision_latency_ms"]),
            "last_error": str(state.get("last_error", "")),
        }
    finally:
        application.close()


def run_authority_benchmark(
    orders: tuple[str, ...],
    *,
    modes: tuple[str, ...] = ("AUTHORITY", "FALLBACK"),
    max_sim_time: float = 2000.0,
    dt: float = 0.05,
) -> dict[str, Any]:
    runs = {
        mode: run_mode(orders, mode=mode, max_sim_time=max_sim_time, dt=dt)
        for mode in modes
    }
    comparison = None
    if {"AUTHORITY", "FALLBACK"} <= set(runs):
        comparison = compare_runs(runs["AUTHORITY"], runs["FALLBACK"])
    return {
        "schema_version": 1,
        "benchmark": "TWINSHIELD_RH_V2_AUTHORITY_COMPARISON",
        "inputs": {
            "orders": list(orders),
            "unit_count": len(orders),
            "modes": list(modes),
            "max_sim_time": float(max_sim_time),
            "dt": float(dt),
        },
        "runs": runs,
        "comparison": comparison,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--orders", default="A,B,C")
    parser.add_argument("--modes", default="AUTHORITY,FALLBACK")
    parser.add_argument("--max-sim-time", type=float, default=2000.0)
    parser.add_argument("--dt", type=float, default=0.05)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    if args.max_sim_time <= 0.0 or args.dt <= 0.0:
        parser.error("--max-sim-time和--dt必须大于0")
    try:
        orders = parse_orders(args.orders)
        modes = parse_modes(args.modes)
    except ValueError as exc:
        parser.error(str(exc))
    report = run_authority_benchmark(
        orders,
        modes=modes,
        max_sim_time=args.max_sim_time,
        dt=args.dt,
    )
    encoded = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded + "\n", encoding="utf-8")
    print(encoded)
    return 0 if all(bool(run["completed"]) for run in report["runs"].values()) else 2


if __name__ == "__main__":
    raise SystemExit(main())
