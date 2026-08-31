#!/usr/bin/env python3
"""Solve and export a bounded CP-SAT reference plan for 1-6 V2 units."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
from typing import Any, Sequence

from brazing_sim.dual_line.unified_runtime import UnifiedV2Runtime
from brazing_sim.optimization import PlanStatus, ReferencePlan


_SUPPORTED_PRESETS = frozenset({"A", "B", "C", "D"})


def parse_orders(value: str) -> tuple[str, ...]:
    orders = tuple(part.strip().upper() for part in str(value).split(",") if part.strip())
    if not 1 <= len(orders) <= 6:
        raise ValueError("CP-SAT小规模参照只接受1至6件订单")
    unsupported = sorted(set(orders) - _SUPPORTED_PRESETS)
    if unsupported:
        raise ValueError(f"不支持的预设订单：{', '.join(unsupported)}")
    return orders


def summarize_plan(
    plan: ReferencePlan,
    *,
    orders: tuple[str, ...],
    active_task_count: int,
    time_limit_s: float,
    random_seed: int,
) -> dict[str, Any]:
    resources: dict[str, dict[str, float | int]] = defaultdict(
        lambda: {"operation_count": 0, "busy_s": 0.0}
    )
    batches: dict[str, list[Any]] = defaultdict(list)
    for operation in plan.operations:
        resource = resources[operation.resource_id]
        resource["operation_count"] += 1
        resource["busy_s"] += operation.duration_s
        if operation.batch_id is not None:
            batches[operation.batch_id].append(operation)

    batch_rows = []
    for batch_id, members in sorted(batches.items()):
        ordered = sorted(members, key=lambda item: item.task_id)
        batch_rows.append(
            {
                "batch_id": batch_id,
                "start_s": round(ordered[0].start_s, 6),
                "end_s": round(ordered[0].end_s, 6),
                "task_ids": [member.task_id for member in ordered],
            }
        )
    resource_rows = {
        resource_id: {
            "operation_count": int(values["operation_count"]),
            "busy_s": round(float(values["busy_s"]), 6),
        }
        for resource_id, values in sorted(resources.items())
    }
    return {
        "schema_version": 1,
        "benchmark": "CP_SAT_REFERENCE_V2",
        "inputs": {
            "orders": list(orders),
            "unit_count": len(orders),
            "time_limit_s": float(time_limit_s),
            "random_seed": int(random_seed),
        },
        "snapshot": {
            "fingerprint": plan.snapshot_fingerprint,
            "plan_version": plan.plan_version,
            "active_task_count": int(active_task_count),
        },
        "result": plan.as_dict(),
        "resources": resource_rows,
        "batches": batch_rows,
    }


def run_reference_benchmark(
    orders: tuple[str, ...],
    *,
    time_limit_s: float = 10.0,
    random_seed: int = 0,
) -> dict[str, Any]:
    runtime = UnifiedV2Runtime(fast=True)
    for index, preset in enumerate(orders, start=1):
        runtime.submit_order(
            preset,
            order_id=f"REFERENCE_{index:02d}_{preset}",
            priority=10,
        )
    snapshot = runtime.manufacturing_runtime.capture_digital_twin(runtime.sim_time)
    plan = runtime.compute_reference_plan(
        time_limit_s=time_limit_s,
        random_seed=random_seed,
        emit_event=False,
    )
    return summarize_plan(
        plan,
        orders=orders,
        active_task_count=len(snapshot.tasks),
        time_limit_s=time_limit_s,
        random_seed=random_seed,
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--orders", default="A,B,C", help="逗号分隔的1至6个预设订单")
    parser.add_argument("--time-limit", type=float, default=10.0, help="求解时限（秒）")
    parser.add_argument("--seed", type=int, default=0, help="CP-SAT确定性随机种子")
    parser.add_argument("--output", type=Path, help="可选JSON输出路径")
    args = parser.parse_args(argv)
    if args.time_limit <= 0.0:
        parser.error("--time-limit必须大于0")
    try:
        orders = parse_orders(args.orders)
    except ValueError as exc:
        parser.error(str(exc))
    report = run_reference_benchmark(
        orders,
        time_limit_s=args.time_limit,
        random_seed=args.seed,
    )
    encoded = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded + "\n", encoding="utf-8")
    print(encoded)
    status = PlanStatus(report["result"]["status"])
    return 0 if status in {PlanStatus.OPTIMAL, PlanStatus.FEASIBLE} else 2


if __name__ == "__main__":
    raise SystemExit(main())
