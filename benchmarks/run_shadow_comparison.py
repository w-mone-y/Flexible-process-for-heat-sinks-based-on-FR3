#!/usr/bin/env python3
"""Compare current dispatch boundary, TwinShield-RH and CP-SAT in shadow mode."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

from brazing_sim.dual_line.unified_runtime import UnifiedV2Runtime
try:
    from benchmarks.run_reference_plan import parse_orders
except ModuleNotFoundError:  # direct ``python benchmarks/script.py`` execution
    from run_reference_plan import parse_orders


def build_comparison(
    orders: tuple[str, ...], *, time_limit_s: float = 10.0, random_seed: int = 0
) -> dict[str, Any]:
    runtime = UnifiedV2Runtime(fast=True)
    for index, preset in enumerate(orders, start=1):
        runtime.submit_order(preset, order_id=f"SHADOW_{index:02d}_{preset}")
    before = len(runtime.manufacturing_runtime.assignment_history)
    proposal = runtime.compute_shadow_schedule(
        include_reference=True,
        time_limit_s=time_limit_s,
        random_seed=random_seed,
        emit_event=False,
    )
    return {
        "schema_version": 1,
        "benchmark": "TWINSHIELD_RH_SHADOW_COMPARISON",
        "inputs": {
            "orders": list(orders),
            "unit_count": len(orders),
            "time_limit_s": float(time_limit_s),
            "random_seed": int(random_seed),
        },
        "dispatch_boundary": {
            "assignments_before_shadow": before,
            "dispatch_mutated": len(runtime.manufacturing_runtime.assignment_history) != before,
            "scheduler_mode": runtime.manufacturing_runtime.scheduler_mode,
        },
        "shadow_schedule": proposal.as_dict(),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--orders", default="A,B,C")
    parser.add_argument("--time-limit", type=float, default=10.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    if args.time_limit <= 0.0:
        parser.error("--time-limit必须大于0")
    try:
        orders = parse_orders(args.orders)
    except ValueError as exc:
        parser.error(str(exc))
    report = build_comparison(orders, time_limit_s=args.time_limit, random_seed=args.seed)
    encoded = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded + "\n", encoding="utf-8")
    print(encoded)
    return 0 if report["shadow_schedule"]["status"] == "FEASIBLE" else 2


if __name__ == "__main__":
    raise SystemExit(main())
