from __future__ import annotations

import pytest

from brazing_sim.optimization import (
    CpSatReferencePlanner,
    PlanOperation,
    PlanStatus,
    ReferencePlan,
    ReferencePlanValidator,
)
from brazing_sim.twin import DigitalTwinSnapshot


def _snapshot(tasks: list[dict], *, sim_time: float = 0.0) -> DigitalTwinSnapshot:
    return DigitalTwinSnapshot.from_mapping(
        {
            "schema_version": 2,
            "sim_time": sim_time,
            "tasks": tasks,
            "orders": [],
            "resources_v2": {
                "ARM1": {"status": "IDLE"},
                "ARM2": {"status": "IDLE"},
                "ARM3": {"status": "IDLE"},
            },
        },
        source_name="test",
    )


def _task(
    task_id: str,
    *,
    duration: float = 2.0,
    resources: tuple[str, ...] = ("ARM1",),
    predecessors: tuple[str, ...] = (),
    station: str | None = None,
    zones: tuple[str, ...] = (),
    status: str = "READY",
    assigned_resource: str | None = None,
    started_at: float | None = None,
    task_type: str = "INSTALL_FIN",
    payload: dict | None = None,
) -> dict:
    return {
        "task_id": task_id,
        "task_type": task_type,
        "order_id": "ORDER",
        "unit_id": "UNIT",
        "status": status,
        "estimated_duration": duration,
        "eligible_resources": list(resources),
        "predecessors": list(predecessors),
        "station_id": station,
        "required_zones": list(zones),
        "assigned_resource": assigned_resource,
        "started_at": started_at,
        "priority": 1,
        "payload": {} if payload is None else payload,
    }


def test_validator_accepts_parallel_eligible_operations() -> None:
    snapshot = _snapshot(
        [
            _task("A", resources=("ARM1",)),
            _task("B", resources=("ARM3",)),
        ]
    )
    plan = ReferencePlan(
        status=PlanStatus.FEASIBLE,
        operations=(
            PlanOperation("A", "ARM1", 0.0, 2.0),
            PlanOperation("B", "ARM3", 0.0, 2.0),
        ),
        makespan_s=2.0,
    )

    report = ReferencePlanValidator().validate(snapshot, plan)

    assert report.valid
    assert report.violations == ()


@pytest.mark.parametrize(
    ("tasks", "operations", "expected_code"),
    [
        (
            [_task("A"), _task("B", predecessors=("A",))],
            (PlanOperation("A", "ARM1", 0.0, 2.0), PlanOperation("B", "ARM1", 1.0, 3.0)),
            "PRECEDENCE_VIOLATION",
        ),
        (
            [_task("A"), _task("B")],
            (PlanOperation("A", "ARM1", 0.0, 2.0), PlanOperation("B", "ARM1", 1.0, 3.0)),
            "RESOURCE_OVERLAP",
        ),
        (
            [_task("A", resources=("ARM1",))],
            (PlanOperation("A", "ARM3", 0.0, 2.0),),
            "INELIGIBLE_RESOURCE",
        ),
        (
            [_task("A", resources=("ARM1",), station="S4"), _task("B", resources=("ARM3",), station="S4")],
            (PlanOperation("A", "ARM1", 0.0, 2.0), PlanOperation("B", "ARM3", 1.0, 3.0)),
            "STATION_OVERLAP",
        ),
        (
            [_task("A", resources=("ARM1",), zones=("SHARED",)), _task("B", resources=("ARM3",), zones=("SHARED",))],
            (PlanOperation("A", "ARM1", 0.0, 2.0), PlanOperation("B", "ARM3", 1.0, 3.0)),
            "ZONE_OVERLAP",
        ),
    ],
)
def test_validator_rejects_invalid_plans(tasks, operations, expected_code) -> None:
    report = ReferencePlanValidator().validate(
        _snapshot(tasks),
        ReferencePlan(status=PlanStatus.FEASIBLE, operations=operations, makespan_s=3.0),
    )

    assert not report.valid
    assert expected_code in {violation.code for violation in report.violations}


def test_cp_sat_serialises_two_tasks_on_one_resource() -> None:
    snapshot = _snapshot([_task("A", duration=2.0), _task("B", duration=3.0)])

    plan = CpSatReferencePlanner(time_limit_s=1.0).solve(snapshot)

    assert plan.status is PlanStatus.OPTIMAL
    assert plan.makespan_s == pytest.approx(5.0)
    assert plan.optimality_gap == pytest.approx(0.0)
    assert ReferencePlanValidator().validate(snapshot, plan).valid


def test_cp_sat_uses_parallel_resources_and_respects_shared_station() -> None:
    planner = CpSatReferencePlanner(time_limit_s=1.0)
    parallel = _snapshot(
        [
            _task("A", duration=4.0, resources=("ARM1",)),
            _task("B", duration=3.0, resources=("ARM3",)),
        ]
    )
    shared_station = _snapshot(
        [
            _task("A", duration=4.0, resources=("ARM1",), station="S4"),
            _task("B", duration=3.0, resources=("ARM3",), station="S4"),
        ]
    )

    assert planner.solve(parallel).makespan_s == pytest.approx(4.0)
    assert planner.solve(shared_station).makespan_s == pytest.approx(7.0)


def test_cp_sat_keeps_running_task_on_assigned_resource_at_time_zero() -> None:
    snapshot = _snapshot(
        [
            _task(
                "RUNNING",
                duration=5.0,
                resources=("ARM1", "ARM3"),
                status="RUNNING",
                assigned_resource="ARM3",
                started_at=8.0,
            )
        ],
        sim_time=10.0,
    )

    plan = CpSatReferencePlanner(time_limit_s=1.0).solve(snapshot)

    assert plan.operations == (PlanOperation("RUNNING", "ARM3", 0.0, 3.0),)


def test_cp_sat_prioritises_weighted_due_date_before_makespan_tie() -> None:
    ordinary = _task("ORDINARY", duration=3.0)
    urgent = _task("URGENT", duration=2.0)
    ordinary["order_id"] = "ORDINARY_ORDER"
    urgent["order_id"] = "URGENT_ORDER"
    urgent["priority"] = 20
    urgent["payload"] = {"due_at_sim_time": 2.0}

    plan = CpSatReferencePlanner(time_limit_s=1.0).solve(_snapshot([ordinary, urgent]))

    assert plan.status is PlanStatus.OPTIMAL
    assert plan.operations[0].task_id == "URGENT"
    assert plan.weighted_tardiness_s == pytest.approx(0.0)


def test_cp_sat_reports_invalid_input_without_throwing() -> None:
    snapshot = _snapshot([_task("NO_RESOURCE", resources=())])

    plan = CpSatReferencePlanner(time_limit_s=1.0).solve(snapshot)

    assert plan.status is PlanStatus.INVALID_INPUT
    assert plan.validation is not None
    assert not plan.validation.valid


def test_empty_snapshot_has_zero_optimal_reference_plan() -> None:
    plan = CpSatReferencePlanner(time_limit_s=1.0).solve(_snapshot([]))

    assert plan.status is PlanStatus.OPTIMAL
    assert plan.makespan_s == 0.0
    assert plan.operations == ()


def test_missing_ortools_is_reported_without_breaking_runtime(monkeypatch) -> None:
    import brazing_sim.optimization.cp_sat_reference as module

    monkeypatch.setattr(module, "cp_model", None)

    plan = module.CpSatReferencePlanner(time_limit_s=1.0).solve(_snapshot([_task("A")]))

    assert plan.status is PlanStatus.UNAVAILABLE
    assert "OR-Tools" in plan.message


def test_cp_sat_batches_three_compatible_furnace_tasks_into_one_cycle() -> None:
    tasks = [
        _task(
            f"FURNACE_{index}",
            duration=10.0,
            resources=("FURNACE",),
            task_type="RUN_FURNACE",
            zones=("ZONE_FURNACE",),
            payload={"recipe": "CAB_A"},
        )
        for index in range(3)
    ]

    plan = CpSatReferencePlanner(time_limit_s=1.0).solve(_snapshot(tasks))

    assert plan.status is PlanStatus.OPTIMAL
    assert plan.makespan_s == pytest.approx(10.0)
    assert len({operation.batch_id for operation in plan.operations}) == 1
    assert ReferencePlanValidator().validate(_snapshot(tasks), plan).valid


def test_cp_sat_splits_four_units_or_incompatible_recipes_across_cycles() -> None:
    four = [
        _task(
            f"FURNACE_{index}",
            duration=10.0,
            resources=("FURNACE",),
            task_type="RUN_FURNACE",
            payload={"recipe": "CAB_A"},
        )
        for index in range(4)
    ]
    incompatible = [
        _task(
            "FURNACE_A",
            duration=10.0,
            resources=("FURNACE",),
            task_type="RUN_FURNACE",
            payload={"recipe": "CAB_A"},
        ),
        _task(
            "FURNACE_B",
            duration=10.0,
            resources=("FURNACE",),
            task_type="RUN_FURNACE",
            payload={"recipe": "CAB_B"},
        ),
    ]

    assert CpSatReferencePlanner(time_limit_s=1.0).solve(_snapshot(four)).makespan_s == pytest.approx(20.0)
    assert CpSatReferencePlanner(time_limit_s=1.0).solve(_snapshot(incompatible)).makespan_s == pytest.approx(
        20.0
    )
