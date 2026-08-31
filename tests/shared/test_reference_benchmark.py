from __future__ import annotations

import pytest

from benchmarks.run_reference_plan import parse_orders, summarize_plan
from brazing_sim.optimization import (
    PlanOperation,
    PlanStatus,
    PlanValidation,
    ReferencePlan,
)


def test_parse_orders_normalizes_and_limits_reference_horizon() -> None:
    assert parse_orders("a, B,c") == ("A", "B", "C")

    with pytest.raises(ValueError, match="1至6"):
        parse_orders("A,B,C,A,B,C,A")


def test_reference_report_exposes_evidence_and_batch_membership() -> None:
    plan = ReferencePlan(
        status=PlanStatus.OPTIMAL,
        operations=(
            PlanOperation("A_FURNACE", "FURNACE", 2.0, 12.0, "BATCH_01"),
            PlanOperation("B_FURNACE", "FURNACE", 2.0, 12.0, "BATCH_01"),
            PlanOperation("A_FIN", "ARM1", 0.0, 2.0),
        ),
        makespan_s=12.0,
        objective_value=12.0,
        best_bound=12.0,
        optimality_gap=0.0,
        snapshot_fingerprint="abc123",
        validation=PlanValidation(True),
    )

    report = summarize_plan(
        plan,
        orders=("A", "B"),
        active_task_count=3,
        time_limit_s=1.0,
        random_seed=7,
    )

    assert report["inputs"]["orders"] == ["A", "B"]
    assert report["inputs"]["random_seed"] == 7
    assert report["snapshot"]["fingerprint"] == "abc123"
    assert report["result"]["validation"]["valid"] is True
    assert report["resources"]["FURNACE"]["operation_count"] == 2
    assert report["batches"] == [
        {
            "batch_id": "BATCH_01",
            "start_s": 2.0,
            "end_s": 12.0,
            "task_ids": ["A_FURNACE", "B_FURNACE"],
        }
    ]
