from __future__ import annotations

from dataclasses import replace

from brazing_sim.execution import SkillRegistry, TimedSkill
from brazing_sim.manufacturing_runtime import ManufacturingRuntime
from brazing_sim.optimization import PlanStatus, PlanValidation
from brazing_sim.planning import ManufacturingTask, TaskStatus, TaskType
from brazing_sim.scheduling import (
    ShadowCandidate,
    ShadowScheduleProposal,
    Assignment,
    TwinShieldAuthority,
)


def _task(
    task_id: str,
    resource: str,
    *,
    zone: str = "",
    task_type: TaskType = TaskType.INSTALL_FIN,
) -> ManufacturingTask:
    return ManufacturingTask(
        task_id=task_id,
        task_type=task_type,
        order_id="ORDER",
        unit_id="UNIT",
        eligible_resources=[resource],
        required_zones=[] if not zone else [zone],
        estimated_duration=2.0,
        status=TaskStatus.READY,
    )


def _proposal(runtime: ManufacturingRuntime, *candidates: ShadowCandidate) -> ShadowScheduleProposal:
    snapshot = runtime.capture_digital_twin(0.0)
    return ShadowScheduleProposal(
        status=PlanStatus.FEASIBLE,
        snapshot_fingerprint=snapshot.fingerprint,
        sim_time=0.0,
        horizon_seconds=60.0,
        selected=tuple(candidates),
        candidates=tuple(candidates),
        validation=PlanValidation(valid=True),
    )


def _candidate(task_id: str, resource: str, cost: float = 1.0) -> ShadowCandidate:
    return ShadowCandidate(task_id, resource, task_id, 0.0, 2.0, cost, {}, True)


def test_authority_accepts_one_conflict_free_commit_window() -> None:
    runtime = ManufacturingRuntime(enable_motion_planning=False)
    first = _task("FIRST", "ARM1", zone="ZONE_TABLE1")
    second = _task(
        "SECOND",
        "ARM2",
        zone="ZONE_TABLE2_CORE",
        task_type=TaskType.DISPENSE_BRAZING,
    )
    runtime.graph.add_task(first)
    runtime.graph.add_task(second)
    proposal = _proposal(runtime, _candidate("FIRST", "ARM1"), _candidate("SECOND", "ARM2"))

    decision = TwinShieldAuthority().decide(
        proposal,
        snapshot=runtime.capture_digital_twin(0.0),
        ready_tasks=(first, second),
        resources=runtime.resources.states,
        zone_leases=runtime.zones.snapshot(),
    )

    assert decision.accepted
    assert [item.task_id for item in decision.assignments] == ["FIRST", "SECOND"]
    assert decision.fallback_reason == ""


def test_authority_rejects_stale_snapshot_and_requests_deterministic_fallback() -> None:
    runtime = ManufacturingRuntime(enable_motion_planning=False)
    task = _task("STALE", "ARM1")
    runtime.graph.add_task(task)
    proposal = _proposal(runtime, _candidate("STALE", "ARM1"))
    current = runtime.capture_digital_twin(0.0)
    stale = replace(proposal, snapshot_fingerprint="old-fingerprint")

    decision = TwinShieldAuthority().decide(
        stale,
        snapshot=current,
        ready_tasks=(task,),
        resources=runtime.resources.states,
        zone_leases=runtime.zones.snapshot(),
    )

    assert not decision.accepted
    assert decision.source == "CURRENT_SCHEDULER"
    assert decision.rejections[0].reason_code == "STALE_SNAPSHOT"


def test_authority_rejects_entire_window_when_one_candidate_became_invalid() -> None:
    runtime = ManufacturingRuntime(enable_motion_planning=False)
    valid = _task("VALID", "ARM1")
    invalid = _task("INVALID", "ARM2")
    runtime.graph.add_task(valid)
    runtime.graph.add_task(invalid)
    proposal = _proposal(runtime, _candidate("VALID", "ARM1"), _candidate("INVALID", "ARM2"))
    invalid.status = TaskStatus.RUNNING
    current = runtime.capture_digital_twin(0.0)
    proposal = replace(proposal, snapshot_fingerprint=current.fingerprint)

    decision = TwinShieldAuthority().decide(
        proposal,
        snapshot=current,
        ready_tasks=(valid,),
        resources=runtime.resources.states,
        zone_leases=runtime.zones.snapshot(),
    )

    assert not decision.accepted
    assert decision.assignments == ()
    assert any(item.task_id == "INVALID" for item in decision.rejections)


def test_authority_rejects_resource_and_zone_conflicts_inside_commit_window() -> None:
    runtime = ManufacturingRuntime(enable_motion_planning=False)
    first = _task("FIRST", "ARM1", zone="ZONE_TABLE2_CORE")
    second = _task("SECOND", "ARM1", zone="ZONE_TABLE2_CORE")
    runtime.graph.add_task(first)
    runtime.graph.add_task(second)
    proposal = _proposal(runtime, _candidate("FIRST", "ARM1"), _candidate("SECOND", "ARM1"))

    decision = TwinShieldAuthority().decide(
        proposal,
        snapshot=runtime.capture_digital_twin(0.0),
        ready_tasks=(first, second),
        resources=runtime.resources.states,
        zone_leases=runtime.zones.snapshot(),
    )

    assert not decision.accepted
    assert {item.reason_code for item in decision.rejections} & {
        "RESOURCE_CONFLICT",
        "ZONE_CONFLICT",
    }


def test_authority_never_rebinds_an_already_running_task() -> None:
    runtime = ManufacturingRuntime(enable_motion_planning=False)
    task = _task("RUNNING", "ARM1")
    runtime.graph.add_task(task)
    task.reserve("ARM1")
    task.mark_running(0.0)
    proposal = _proposal(runtime, _candidate("RUNNING", "ARM2"))

    decision = TwinShieldAuthority().decide(
        proposal,
        snapshot=runtime.capture_digital_twin(0.0),
        ready_tasks=(),
        resources=runtime.resources.states,
        zone_leases=runtime.zones.snapshot(),
    )

    assert not decision.accepted
    assert decision.rejections[0].reason_code == "TASK_NOT_READY"


def test_runtime_falls_back_when_twinshield_proposal_is_stale() -> None:
    runtime = ManufacturingRuntime(enable_motion_planning=False, twinshield_mode="AUTHORITY")
    task = _task("FALLBACK", "ARM1")
    runtime.graph.add_task(task)
    stale = replace(_proposal(runtime, _candidate("FALLBACK", "ARM1")), snapshot_fingerprint="stale")
    runtime.compute_shadow_schedule = lambda *args, **kwargs: stale  # type: ignore[method-assign]

    runtime.tick(0.0)

    state = runtime.snapshot(0.0)
    assert state["twinshield"]["last_source"] == "CURRENT_SCHEDULER"
    assert state["twinshield"]["fallback_count"] == 1
    assert task.status is TaskStatus.RUNNING


def test_operator_fallback_mode_bypasses_twinshield_and_uses_current_scheduler() -> None:
    runtime = ManufacturingRuntime(enable_motion_planning=False, twinshield_mode="FALLBACK")
    task = _task("ROLLBACK", "ARM1")
    runtime.graph.add_task(task)

    def unexpected_shadow(*args, **kwargs):  # type: ignore[no-untyped-def]
        raise AssertionError("fallback mode must not invoke TwinShield")

    runtime.compute_shadow_schedule = unexpected_shadow  # type: ignore[method-assign]

    runtime.tick(0.0)

    assert task.status is TaskStatus.RUNNING
    assert runtime.snapshot(0.0)["twinshield"]["last_source"] == "CURRENT_SCHEDULER"


def test_authority_replans_only_when_the_dispatch_boundary_changes() -> None:
    runtime = ManufacturingRuntime(enable_motion_planning=False, twinshield_mode="AUTHORITY")
    task = _task("EVENT_BOUNDARY", "ARM1")
    runtime.graph.add_task(task)
    calls = 0

    def proposal(*args, **kwargs):  # type: ignore[no-untyped-def]
        nonlocal calls
        calls += 1
        now = float(args[0])
        snapshot = runtime.capture_digital_twin(now)
        return replace(
            _proposal(runtime, _candidate("EVENT_BOUNDARY", "ARM1")),
            snapshot_fingerprint=snapshot.fingerprint,
            sim_time=now,
        )

    runtime.compute_shadow_schedule = proposal  # type: ignore[method-assign]

    first = runtime._twinshield_decision([task], 0.0)  # noqa: SLF001
    repeated = runtime._twinshield_decision([task], 0.05)  # noqa: SLF001
    runtime.resources.reserve("ARM1", "EXTERNAL", 0.1)
    changed = runtime._twinshield_decision([task], 0.1)  # noqa: SLF001

    assert first is not None and first.accepted
    assert repeated is None
    assert changed is None or not changed.accepted
    assert calls == 2


def test_authority_solver_exception_falls_back_without_failing_the_order() -> None:
    runtime = ManufacturingRuntime(enable_motion_planning=False, twinshield_mode="AUTHORITY")
    task = _task("SOLVER_FAILURE", "ARM1")
    runtime.graph.add_task(task)

    def broken_solver(*args, **kwargs):  # type: ignore[no-untyped-def]
        raise RuntimeError("solver timeout")

    runtime.compute_shadow_schedule = broken_solver  # type: ignore[method-assign]

    runtime.tick(0.0)

    state = runtime.snapshot(0.0)
    assert task.status is TaskStatus.RUNNING
    assert state["twinshield"]["fallback_count"] == 1
    assert "solver timeout" in state["twinshield"]["last_fallback_reason"]


def test_authority_snapshot_reports_decision_latency_distribution() -> None:
    runtime = ManufacturingRuntime(enable_motion_planning=False, twinshield_mode="AUTHORITY")
    task = _task("LATENCY", "ARM1")
    runtime.graph.add_task(task)

    runtime._twinshield_decision([task], 0.0)  # noqa: SLF001

    latency = runtime.snapshot(0.0)["twinshield"]["decision_latency_ms"]
    assert latency["sample_count"] == 1
    assert latency["p95"] >= 0.0
    assert latency["maximum"] >= latency["p95"]


class _ExplodingSkill(TimedSkill):
    def start(self, task, resource_id, context, now):  # type: ignore[no-untyped-def]
        raise RuntimeError("SECOND_START_FAILED")


def test_atomic_commit_rolls_back_every_task_when_one_skill_cannot_start() -> None:
    registry = SkillRegistry()
    for task_type in TaskType:
        registry.register_factory(task_type, TimedSkill)
    registry.register_factory(TaskType.DISPENSE_BRAZING, _ExplodingSkill, replace=True)
    runtime = ManufacturingRuntime(
        enable_motion_planning=False,
        skill_registry=registry,
        twinshield_mode="AUTHORITY",
    )
    first = _task("FIRST", "ARM1", zone="ZONE_TABLE1")
    second = _task(
        "SECOND",
        "ARM2",
        zone="ZONE_TABLE2_CORE",
        task_type=TaskType.DISPENSE_BRAZING,
    )
    runtime.graph.add_task(first)
    runtime.graph.add_task(second)

    committed, reason = runtime._commit_assignments_atomically(  # noqa: SLF001
        (Assignment("FIRST", "ARM1"), Assignment("SECOND", "ARM2")),
        0.0,
        source="TWINSHIELD_RH",
    )

    assert not committed
    assert reason == "SECOND_START_FAILED"
    assert first.status is TaskStatus.READY
    assert second.status is TaskStatus.READY
    assert runtime.executor.active == {}
    assert all(resource.status.value == "IDLE" for resource in runtime.resources.states.values())
    assert all(lease is None for lease in runtime.zones.snapshot().values())
    assert runtime.assignment_history == []
