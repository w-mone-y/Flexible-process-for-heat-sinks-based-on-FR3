from __future__ import annotations

from brazing_sim.events import EventType
from brazing_sim.manufacturing_runtime import ManufacturingRuntime
from brazing_sim.optimization import PlanStatus
from brazing_sim.planning import ManufacturingTask, TaskStatus, TaskType
from brazing_sim.flexible import build_preset_plan
from brazing_sim.dual_line.unified_runtime import UnifiedV2Runtime


def test_runtime_computes_reference_plan_as_shadow_event() -> None:
    runtime = ManufacturingRuntime(enable_motion_planning=False)
    runtime.graph.add_task(
        ManufacturingTask(
            task_id="REFERENCE_TASK",
            task_type=TaskType.INSTALL_FIN,
            order_id="ORDER",
            unit_id="UNIT",
            eligible_resources=["ARM1"],
            estimated_duration=2.0,
            status=TaskStatus.READY,
        )
    )

    before = runtime.capture_digital_twin().fingerprint
    plan = runtime.compute_reference_plan(time_limit_s=1.0, emit_event=True)
    after = runtime.capture_digital_twin().fingerprint

    assert plan.status is PlanStatus.OPTIMAL
    assert plan.operations[0].task_id == "REFERENCE_TASK"
    assert before == after
    assert runtime.events.history[-1].event_type is EventType.PLAN_PROPOSED
    assert runtime.snapshot()["reference_plan"]["status"] == "OPTIMAL"


def test_runtime_reset_discards_derived_reference_plan() -> None:
    runtime = ManufacturingRuntime(enable_motion_planning=False)
    runtime.compute_reference_plan(time_limit_s=1.0)

    runtime.reset(0.0)

    assert runtime.snapshot()["reference_plan"] is None


def test_runtime_order_snapshot_preserves_simulation_due_date_for_reference_solver() -> None:
    runtime = ManufacturingRuntime(enable_motion_planning=False)
    runtime.submit_plan(build_preset_plan("A"), now=4.0, due_at=120.0)

    order = runtime.capture_digital_twin(4.0).orders[0]

    assert order["due_at_sim_time"] == 120.0


def test_unified_v2_runtime_exposes_reference_solver_without_dispatching_plan() -> None:
    runtime = UnifiedV2Runtime(fast=True)
    runtime.submit_order("A", order_id="REFERENCE_V2", due_at=300.0)
    before = len(runtime.manufacturing_runtime.assignment_history)

    plan = runtime.compute_reference_plan(time_limit_s=1.0, random_seed=17, emit_event=False)

    assert plan.status in {PlanStatus.OPTIMAL, PlanStatus.FEASIBLE}
    assert plan.validation is not None and plan.validation.valid
    assert plan.metadata["random_seed"] == 17
    assert len(runtime.manufacturing_runtime.assignment_history) == before


def test_unified_v2_runtime_exposes_shadow_schedule_without_dispatching_plan() -> None:
    runtime = UnifiedV2Runtime(fast=True)
    runtime.submit_order("A", order_id="SHADOW_V2")
    before = len(runtime.manufacturing_runtime.assignment_history)

    proposal = runtime.compute_shadow_schedule(include_reference=False, emit_event=False)

    assert proposal.snapshot_fingerprint
    assert proposal.validation is not None
    assert len(runtime.manufacturing_runtime.assignment_history) == before


def test_unified_v2_runtime_can_use_twinshield_authority_at_a_safe_boundary() -> None:
    runtime = UnifiedV2Runtime(fast=True, twinshield_mode="AUTHORITY")
    runtime.submit_order("A", order_id="AUTHORITY_V2")

    runtime.tick(0.05)

    state = runtime.manufacturing_runtime.snapshot(runtime.sim_time)
    assert state["twinshield"]["mode"] == "AUTHORITY"
    assert state["twinshield"]["authority_count"] + state["twinshield"]["fallback_count"] > 0
    assert all(task["status"] != "ERROR" for task in state["tasks"])
