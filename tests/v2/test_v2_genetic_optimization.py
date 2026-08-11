from __future__ import annotations

import pytest

from brazing_sim.dual_line import DualLineRuntime, UnifiedV2Runtime
from brazing_sim.dual_line.application import V2BrazingApplication
from brazing_sim.dual_line.cli import parse_args
from brazing_sim.dual_line.optimization import GeneticReleasePlanner, ReleaseCandidate
from brazing_sim.flexible import build_inline_plan


def _candidate(
    unit_id: str,
    *,
    family: str,
    flow: float,
    priority: int = 10,
    urgent: bool = False,
    due_at: float | None = None,
) -> ReleaseCandidate:
    return ReleaseCandidate(
        order_id=unit_id.split("_UNIT", 1)[0],
        unit_id=unit_id,
        family=family,
        priority=priority,
        urgent=urgent,
        inserted_at=0.0,
        due_at=due_at,
        estimated_flow_s=flow,
    )


def test_v2_cli_keeps_rule_as_default_and_exposes_genetic_seed() -> None:
    assert parse_args([]).optimizer == "rule"
    args = parse_args(["--optimizer", "genetic", "--genetic-seed", "7"])
    assert args.optimizer == "genetic"
    assert args.genetic_seed == 7


def test_genetic_release_planner_is_reproducible_and_preserves_permutation() -> None:
    values = (
        _candidate("A_UNIT_01", family="A", flow=12.0),
        _candidate("B_UNIT_01", family="B", flow=4.0),
        _candidate("C_UNIT_01", family="B", flow=4.0, urgent=True),
    )
    first = GeneticReleasePlanner(seed=7).plan(values, now=0.0, current_family=None)
    second = GeneticReleasePlanner(seed=7).plan(values, now=0.0, current_family=None)

    assert first.unit_ids == second.unit_ids
    assert set(first.unit_ids) == {item.unit_id for item in values}
    assert len(first.unit_ids) == len(values)
    assert first.unit_ids[0] == "C_UNIT_01"
    assert first.explored > 0


def test_v2_genetic_runtime_releases_and_completes_a_burst() -> None:
    runtime = DualLineRuntime(fast=True, optimizer="genetic", genetic_seed=7)
    for index, preset in enumerate(("A", "B", "C"), start=1):
        runtime.submit_plan(
            build_inline_plan(
                preset=preset,
                order_id=f"GENETIC_{preset}",
                quantity=1,
                priority=10,
            ),
            dispatch=False,
        )

    runtime.tick(0.05)
    assert runtime.snapshot()["optimization"]["mode"] == "GENETIC"
    assert any(event["type"] == "GENETIC_RELEASE_PLAN" for event in runtime.events)

    snapshot = runtime.run_until_complete(max_sim_time=180.0, dt=0.05)

    assert snapshot["complete"]
    assert snapshot["optimization"]["planner"]["plan_count"] >= 1
    assert all(tray["owner"] == "EMPTY_BUFFER" for tray in snapshot["trays"])


def test_unified_v2_genetic_runtime_keeps_manufacturing_and_physical_release_aligned() -> None:
    runtime = UnifiedV2Runtime(fast=True, optimizer="genetic", genetic_seed=7)
    for preset in ("A", "B", "C"):
        runtime.submit_order(preset, order_id=f"UNIFIED_GENETIC_{preset}")

    snapshot = runtime.tick(0.05)

    assert snapshot["optimization"]["mode"] == "GENETIC"
    running_base_tasks = [
        task
        for task in runtime.manufacturing_runtime.graph
        if task.task_type.value == "PICK_BASE_PLATE" and task.status.value == "RUNNING"
    ]
    assert len(running_base_tasks) <= 1
    assert runtime.next_release_unit_id() is not None or runtime.complete


def test_v2_genetic_application_reaches_the_physical_completion_gate() -> None:
    pytest.importorskip("mujoco")
    application = V2BrazingApplication(
        parse_args(
            [
                "--headless",
                "--no-ui",
                "--fast",
                "--orders",
                "A,B,C",
                "--optimizer",
                "genetic",
                "--genetic-seed",
                "7",
                "--max-sim-time",
                "200",
            ]
        )
    )
    try:
        application.submit_cli_orders()
        while (
            application.controls.running
            and (not application.runtime.complete or not application.scene.transport_settled)
            and max(
                application.runtime.sim_time,
                float(application.scene.data.time),
            )
            + 1.0e-12
            < 200.0
        ):
            application.advance_frame()
        snapshot = application.runtime.snapshot()
        assert application.runtime.complete
        assert application.scene.transport_settled
        assert snapshot["physical_completion_gates"]["passed"]
    finally:
        application.close()
