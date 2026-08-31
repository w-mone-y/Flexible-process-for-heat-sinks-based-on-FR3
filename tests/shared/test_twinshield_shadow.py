from __future__ import annotations

from brazing_sim.events import EventType
from brazing_sim.manufacturing_runtime import ManufacturingRuntime
from brazing_sim.optimization import PlanStatus
from brazing_sim.planning import ManufacturingTask, TaskStatus, TaskType
from brazing_sim.scheduling.twinshield_shadow import TwinShieldShadowScheduler
from brazing_sim.twin import DigitalTwinSnapshot


def _task(
    task_id: str,
    *,
    duration: float = 2.0,
    resources: tuple[str, ...] = ("ARM1",),
    predecessors: tuple[str, ...] = (),
    status: str = "READY",
    assigned_resource: str | None = None,
    station: str | None = None,
    zones: tuple[str, ...] = (),
    priority: int = 1,
) -> dict:
    return {
        "task_id": task_id,
        "task_type": TaskType.INSTALL_FIN.value,
        "order_id": "ORDER",
        "unit_id": "UNIT",
        "status": status,
        "estimated_duration": duration,
        "eligible_resources": list(resources),
        "assigned_resource": assigned_resource,
        "started_at": 0.0 if status == "RUNNING" else None,
        "predecessors": list(predecessors),
        "station_id": station,
        "required_zones": list(zones),
        "required_tool": None,
        "priority": priority,
        "payload": {},
    }


def _snapshot(tasks: list[dict], *, sim_time: float = 0.0) -> DigitalTwinSnapshot:
    return DigitalTwinSnapshot.from_mapping(
        {
            "schema_version": 2,
            "sim_time": sim_time,
            "tasks": tasks,
            "orders": [],
            "resources_v2": {
                "ARM1": {
                    "status": "IDLE",
                    "current_task_id": None,
                    "current_tool": "parallel_gripper",
                    "available_tools": ["vacuum", "parallel_gripper"],
                    "capabilities": [TaskType.INSTALL_FIN.value],
                    "estimated_available_time": 0.0,
                },
                "ARM3": {
                    "status": "IDLE",
                    "current_task_id": None,
                    "current_tool": "parallel_gripper",
                    "available_tools": ["parallel_gripper"],
                    "capabilities": [TaskType.INSTALL_FIN.value],
                    "estimated_available_time": 0.0,
                },
            },
        },
        source_name="test",
    )


def test_shadow_scheduler_selects_parallel_ready_actions_and_explains_rejections() -> None:
    snapshot = _snapshot(
        [
            _task("A", resources=("ARM1",)),
            _task("B", resources=("ARM3",)),
            _task("C", resources=("ARM2",)),
        ]
    )

    proposal = TwinShieldShadowScheduler().plan(snapshot)

    assert proposal.status is PlanStatus.FEASIBLE
    assert {item.task_id for item in proposal.selected} == {"A", "B"}
    assert proposal.selected_count == 2
    assert any(item.task_id == "C" and item.reason_code == "UNKNOWN_RESOURCE" for item in proposal.rejected)
    assert proposal.candidate_count == 3
    assert proposal.snapshot_fingerprint == snapshot.fingerprint


def test_shadow_greedy_plan_respects_precedence_station_and_zone() -> None:
    snapshot = _snapshot(
        [
            _task("FIRST", duration=3.0, resources=("ARM1",), station="S1", zones=("Z",)),
            _task(
                "SECOND",
                duration=2.0,
                resources=("ARM1",),
                predecessors=("FIRST",),
                station="S1",
                zones=("Z",),
            ),
        ]
    )

    proposal = TwinShieldShadowScheduler().plan(snapshot)
    first = next(item for item in proposal.operations if item.task_id == "FIRST")
    second = next(item for item in proposal.operations if item.task_id == "SECOND")

    assert first.end_s <= second.start_s
    assert proposal.estimated_makespan_s == 5.0
    assert proposal.validation is not None and proposal.validation.valid


def test_shadow_schedule_does_not_release_queue_held_wip() -> None:
    held = _task("HELD", resources=("ARM1",))
    held["payload"] = {"queue_held": True}
    proposal = TwinShieldShadowScheduler().plan(_snapshot([held]))

    assert proposal.operations == ()
    assert proposal.selected_count == 0
    assert any(item.reason_code == "WIP_ADMISSION" for item in proposal.rejected)
    assert proposal.validation is not None and proposal.validation.valid


def test_commit_window_mode_skips_full_horizon_greedy_expansion(monkeypatch) -> None:
    scheduler = TwinShieldShadowScheduler()

    def unexpected_full_horizon(*args, **kwargs):  # type: ignore[no-untyped-def]
        raise AssertionError("authority commit path must not expand the full horizon")

    monkeypatch.setattr(scheduler, "_greedy_operations", unexpected_full_horizon)

    proposal = scheduler.plan(
        _snapshot([_task("FAST", resources=("ARM1",))]),
        commit_window_only=True,
    )

    assert proposal.status is PlanStatus.FEASIBLE
    assert [operation.task_id for operation in proposal.operations] == ["FAST"]
    assert proposal.validation is not None and proposal.validation.valid


def test_runtime_shadow_plan_is_observational_and_resettable() -> None:
    runtime = ManufacturingRuntime(enable_motion_planning=False)
    runtime.graph.add_task(
        ManufacturingTask(
            task_id="SHADOW_TASK",
            task_type=TaskType.INSTALL_FIN,
            order_id="ORDER",
            unit_id="UNIT",
            eligible_resources=["ARM1"],
            estimated_duration=2.0,
            status=TaskStatus.READY,
        )
    )
    before = [(task.task_id, task.status.value, task.assigned_resource) for task in runtime.graph]
    before_fingerprint = runtime.capture_digital_twin().fingerprint
    proposal = runtime.compute_shadow_schedule(emit_event=True)
    after = [(task.task_id, task.status.value, task.assigned_resource) for task in runtime.graph]

    assert proposal.status is PlanStatus.FEASIBLE
    assert before == after
    assert runtime.capture_digital_twin().fingerprint == before_fingerprint
    assert runtime.snapshot()["shadow_schedule"]["selected_count"] == 1
    assert runtime.events.history[-1].event_type is EventType.REPLAN_COMPLETED

    runtime.reset()
    assert runtime.snapshot()["shadow_schedule"] is None
