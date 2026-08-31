from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import pytest

from brazing_sim.dual_line.application import V2BrazingApplication
from brazing_sim.dual_line.cli import parse_args
from brazing_sim.dual_line import DualLineRuntime, FurnacePhase, InstallBranch, TrayOwner, UnitStage
from brazing_sim.dual_line.unified_runtime import UnifiedV2Runtime
from brazing_sim.flexible import build_custom_plan
from brazing_sim.planning import TaskType

ROOT = Path(__file__).resolve().parents[2]


class _LegacyBranchPreferenceGate:
    """Compatibility gate whose obsolete preference must not own dispatch."""

    def tray_ready(self, *_args) -> bool:
        return True

    def owner_available(self, *_args) -> bool:
        return True

    def operation_complete(self, *_args) -> bool:
        return True

    def operation_start_allowed(self, *_args) -> bool:
        return True

    def preferred_install_resource(self, *_args) -> str:
        return "ARM3"


def test_physical_branch_threshold_is_not_overridden_by_a_legacy_dag_preference() -> None:
    runtime = DualLineRuntime(fast=True)
    runtime.dispatcher.minimum_arm3_net_gain_s = 10_000.0
    runtime.set_execution_gate(_LegacyBranchPreferenceGate())
    order = runtime.submit_order("A", order_id="BRANCH_AUTHORITY")
    unit = runtime.units[order.unit_ids[0]]

    selected = runtime._assign_branch(unit)

    assert selected is InstallBranch.ARM1_A
    event = next(item for item in reversed(runtime.events) if item["type"] == "INSTALL_ASSIGNED")
    assert event["branch"] == InstallBranch.ARM1_A.value
    assert event["arm3_activated"] is False
    assert runtime.snapshot()["rolling_horizon_scheduler"]["selected_branch"] == "ARM1_A"


def _custom_plan(*, order_id: str = "CUSTOM_RUNTIME", quantity: int = 1):
    return build_custom_plan(
        order_id=order_id,
        quantity=quantity,
        priority=17,
        product={
            "base_size_m": [0.36, 0.22, 0.008],
            "fin_size_m": [0.30, 0.002, 0.06],
            "fin_count": 6,
            "fin_pitch_m": 0.02,
            "path_margin_m": 0.015,
            "path_width_m": 0.004,
            "nozzle_spacing_m": 0.005,
            "nozzle_tip_height_m": 0.004,
            "material_speed_m_s": 0.04,
            "target_clamping_force_n": 24.0,
            "recipe": "demo_brazing",
        },
    )


def test_v2_runtime_executes_a_custom_process_plan_to_completion() -> None:
    runtime = DualLineRuntime(fast=True)
    runtime.submit_plan(_custom_plan())

    snapshot = runtime.run_until_complete(max_sim_time=180.0, dt=0.05)

    assert snapshot["complete"]
    unit = snapshot["units"][0]
    assert unit["preset"] == "CUSTOM"
    assert unit["product_id"] == "CUSTOM_CUSTOM_RUNTIME"
    assert unit["fin_count"] == 6
    assert unit["comb_module"] == "comb_insert_20mm"
    assert unit["path_count"] == 12
    assert snapshot["orders"][0]["mode"] == "custom"


def test_v2_runtime_accepts_the_configured_d_product_family() -> None:
    runtime = DualLineRuntime(fast=True)

    order = runtime.submit_order("D", order_id="PRODUCT_D_RUNTIME")

    unit = runtime.units[order.unit_ids[0]]
    assert unit.preset == "D"
    assert unit.product_id == "PRODUCT_D"
    assert unit.comb_module == "comb_insert_15mm"


def test_compatible_custom_and_preset_orders_share_one_v2_furnace_batch() -> None:
    runtime = DualLineRuntime(fast=True)
    runtime.submit_plan(_custom_plan(order_id="CUSTOM_MIX"))
    runtime.submit_order("A", order_id="PRESET_MIX")

    snapshot = runtime.run_until_complete(max_sim_time=180.0, dt=0.05)

    assert snapshot["complete"]
    assert snapshot["furnace"]["completed_batches"] == 1
    queued = [event for event in snapshot["events"] if event["type"] == "ORDER_QUEUED"]
    assert {event["preset"] for event in queued} == {"CUSTOM", "A"}


def test_v2_runtime_schedules_three_mixed_orders_across_both_install_branches() -> None:
    runtime = DualLineRuntime(fast=True)
    runtime.submit_order("A", order_id="V2_A", priority=10)
    runtime.submit_order("B", order_id="V2_B", priority=20)
    runtime.submit_order("C", order_id="V2_C", priority=30)

    snapshot = runtime.run_until_complete(max_sim_time=180.0, dt=0.05)

    assert snapshot["complete"]
    assert snapshot["execution_mode"] == "CONTROL_PLANE_REHEARSAL"
    assert snapshot["physical_execution_complete"] is False
    assert snapshot["completed_orders"] == ["V2_A", "V2_B", "V2_C"]
    assert set(snapshot["install_branch_counts"]) == {"ARM1_A", "ARM3_B"}
    assert all(value >= 1 for value in snapshot["install_branch_counts"].values())
    assert snapshot["scheduled_parallel_install_seconds"] >= 0.5
    assert snapshot["furnace"]["completed_batches"] == 1
    assert snapshot["furnace"]["last_batch"]["real_equivalent_cycle_s"] == pytest.approx(3600.0)
    assert all(tray["stage"] == "EMPTY_BUFFER" for tray in snapshot["trays"])
    assert all(unit["stage"] == UnitStage.COMPLETE.value for unit in snapshot["units"])
    scheduling = snapshot["rolling_horizon_scheduler"]
    assert scheduling["horizon_seconds"] == 45.0
    assert scheduling["maximum_candidates"] == 4
    assert scheduling["selected_branch"] in {"ARM1_A", "ARM3_B"}
    assert scheduling["arm3_activation"]["reason_zh"]
    assert scheduling["rolling_horizon"]["candidates"]
    assignments = [event for event in snapshot["events"] if event["type"] == "INSTALL_ASSIGNED"]
    assert assignments
    assert all("arm3_net_gain_s" in event for event in assignments)
    batch_started = next(
        event["time"] for event in snapshot["events"] if event["type"] == "FURNACE_BATCH_STARTED"
    )
    door_opened = next(
        event["time"] for event in snapshot["events"] if event["type"] == "FURNACE_FRONT_DOOR_OPENED"
    )
    thermal_started = next(
        event["time"] for event in snapshot["events"] if event["type"] == "FURNACE_THERMAL_CYCLE_STARTED"
    )
    assert batch_started < door_opened < thermal_started
    post_scan_starts = [
        event
        for event in snapshot["events"]
        if event["type"] == "OPERATION_STARTED" and event["kind"] == "POST_BRAZE_INSPECTION"
    ]
    assert post_scan_starts
    assert {event["resource"] for event in post_scan_starts} == {"POST_CAMERA"}
    assert any(
        event["type"] == "TRAY_HANDOFF" and event["target"] == "MERGE_B_WAIT" for event in snapshot["events"]
    )
    for unit in snapshot["units"]:
        stages = [
            event["stage"]
            for event in snapshot["events"]
            if event["type"] == "UNIT_STAGE" and event["unit_id"] == unit["unit_id"]
        ]
        assert stages.index("PRODUCT_REMOVED") < stages.index("VIRTUAL_RETURN")


def test_unified_v2_admits_fourth_order_into_the_next_upstream_wip_cohort() -> None:
    runtime = UnifiedV2Runtime(fast=True)
    for preset in ("A", "B", "C"):
        runtime.submit_order(preset, order_id=f"LAYER_FULL_{preset}")

    runtime.submit_order("A", order_id="LAYER_WAITING_D")

    entry = runtime.manufacturing_runtime.orders["LAYER_WAITING_D"]
    assert entry.status.value == "RELEASED"
    assert entry.admitted_unit_ids == {"LAYER_WAITING_D_UNIT_01"}
    assert entry.tray_assignments == {"tray_01": "tray_04"}
    assert runtime.manufacturing_runtime._active_wip() == 4
    assert len(runtime.manufacturing_runtime.tray_routes) == 6
    assert runtime.physical_runtime._active_batch_units == []


def test_unified_v2_sizes_the_base_wave_from_its_two_parallel_install_branches() -> None:
    runtime = UnifiedV2Runtime(fast=True)
    for index, preset in enumerate("ABCABC", start=1):
        runtime.submit_order(preset, order_id=f"BRANCH_WAVE_{index}")
    runtime.tick(0.05)

    policy = runtime.manufacturing_runtime.arm1_tool_policy.config
    assert policy.max_base_microbatch == 6
    assert policy.drain_admitted_base_wave is True

    snapshot = runtime.manufacturing_runtime.arm1_tool_policy.snapshot()
    assert snapshot["parallel_fin_branches"] == 2


def test_six_order_viewer_pipeline_releases_fin_work_after_a_three_base_wave() -> None:
    args = parse_args(
        (
            "--headless",
            "--fast",
            "--no-ui",
            "--orders",
            "A,B,C,A,B,C",
            "--max-sim-time",
            "80",
        )
    )
    application = V2BrazingApplication(args)
    application.submit_cli_orders()
    runtime = application.runtime.physical_runtime
    first_arm1_fin_started = False
    arm2_ready_idle_streak = 0
    longest_arm2_ready_idle_streak = 0
    try:
        for _ in range(2_000):
            application.advance_frame()
            arm2_ready_work = any(
                unit.stage is UnitStage.DISPENSING and runtime._tray_ready(unit)
                for unit in runtime.units.values()
            )
            if arm2_ready_work and "ARM2" not in runtime.operations:
                arm2_ready_idle_streak += 1
                longest_arm2_ready_idle_streak = max(
                    longest_arm2_ready_idle_streak,
                    arm2_ready_idle_streak,
                )
            else:
                arm2_ready_idle_streak = 0
            arm1 = runtime.operations.get("ARM1")
            if arm1 is not None and arm1.kind == "INSTALL_FIN":
                completed_bases = {
                    event["unit_id"]
                    for event in runtime.events
                    if event["type"] == "OPERATION_COMPLETED" and event.get("kind") == "BASE_LOADING"
                }
                assert len(completed_bases) >= 3
                first_arm1_fin_started = True
                break
    finally:
        application.scene.close()

    assert first_arm1_fin_started
    assert longest_arm2_ready_idle_streak <= 1


def test_unified_v2_physical_admission_follows_global_release_order() -> None:
    runtime = UnifiedV2Runtime(fast=True)
    for preset in ("A", "B", "C", "D"):
        runtime.submit_order(preset, order_id=f"AUTHORITY_{preset}")

    runtime.tick(0.05)
    runtime.tick(0.05)

    stages = {unit.unit_id: unit.stage for unit in runtime.physical_runtime.units.values()}
    released = {
        unit_id
        for entry in runtime.manufacturing_runtime.orders.values()
        for unit_id in entry.admitted_unit_ids
    }
    assert len(released) == 4
    assert sum(stages[unit_id] is UnitStage.BASE_LOADING for unit_id in released) == 1
    assert runtime.physical_runtime.operations["ARM1"].unit_id == "AUTHORITY_A_UNIT_01"
    assert stages["AUTHORITY_D_UNIT_01"] is UnitStage.QUEUED


def test_unified_v2_binds_fin_tasks_to_the_selected_physical_install_branch() -> None:
    runtime = UnifiedV2Runtime(fast=True)
    runtime.submit_order("A", order_id="BRANCH_BOUND")
    unit = runtime.physical_runtime.units["BRANCH_BOUND_UNIT_01"]
    unit.branch = InstallBranch.ARM3_B

    runtime.tick(0.05)

    fin_tasks = [
        task
        for task in runtime.manufacturing_runtime.graph
        if task.unit_id == unit.unit_id and task.task_type in {TaskType.PICK_FIN, TaskType.INSTALL_FIN}
    ]
    assert fin_tasks
    assert all(task.eligible_resources == ["ARM3"] for task in fin_tasks)


def test_unified_v2_furnace_guard_ignores_the_queued_next_batch() -> None:
    runtime = UnifiedV2Runtime(fast=True)
    for preset in ("A", "B", "C", "D"):
        runtime.submit_order(preset, order_id=f"BATCH_GUARD_{preset}")

    active_batch = [f"BATCH_GUARD_{preset}_UNIT_01" for preset in ("A", "B", "C")]
    next_batch_unit = "BATCH_GUARD_D_UNIT_01"
    runtime.physical_runtime._active_batch_units = list(active_batch)
    for unit_id in active_batch:
        runtime.physical_runtime.units[unit_id].stage = UnitStage.BRAZING
    active_task = next(
        candidate
        for candidate in runtime.manufacturing_runtime.graph
        if candidate.task_type is TaskType.RUN_FURNACE and candidate.unit_id in active_batch
    )
    next_batch_task = next(
        candidate
        for candidate in runtime.manufacturing_runtime.graph
        if candidate.task_type is TaskType.RUN_FURNACE and candidate.unit_id == next_batch_unit
    )

    allowed, reason = runtime.bridge.task_dispatch_allowed(active_task)
    next_allowed, next_reason = runtime.bridge.task_dispatch_allowed(next_batch_task)

    assert allowed
    assert reason == ""
    assert not next_allowed
    assert "所属物理炉批" in next_reason
    assert runtime.physical_runtime.units[next_batch_unit].stage is UnitStage.QUEUED


def test_unified_v2_furnace_guard_waits_for_compatible_buffered_batch_member() -> None:
    runtime = UnifiedV2Runtime(fast=True)
    for preset in ("A", "B", "C"):
        runtime.submit_order(preset, order_id=f"OPEN_BATCH_{preset}")

    physical = runtime.physical_runtime
    loaded = ["OPEN_BATCH_A_UNIT_01", "OPEN_BATCH_B_UNIT_01"]
    buffered = "OPEN_BATCH_C_UNIT_01"
    physical._active_batch_units = list(loaded)
    physical._furnace_load_queue = list(loaded)
    physical._furnace_load_position = len(loaded)
    physical.furnace.state.phase = FurnacePhase.LOADING
    for unit_id in loaded:
        physical.units[unit_id].stage = UnitStage.BRAZING
    physical.units[buffered].stage = UnitStage.FURNACE_BUFFER
    task = next(
        candidate
        for candidate in runtime.manufacturing_runtime.graph
        if candidate.task_type is TaskType.RUN_FURNACE and candidate.unit_id in loaded
    )

    allowed, reason = runtime.bridge.task_dispatch_allowed(task)

    assert not allowed
    assert "兼容托盘" in reason


def test_unified_v2_unload_guard_resolves_reused_tray_from_the_active_batch() -> None:
    runtime = UnifiedV2Runtime(fast=True)
    for index, preset in enumerate(("A", "B", "C", "D", "A"), start=1):
        runtime.submit_order(preset, order_id=f"REUSED_TRAY_{index}")

    physical = runtime.physical_runtime
    old_unit = physical.units["REUSED_TRAY_2_UNIT_01"]
    lower_unit = physical.units["REUSED_TRAY_4_UNIT_01"]
    top_unit = physical.units["REUSED_TRAY_5_UNIT_01"]
    old_unit.tray_id = "V2_TRAY_02"
    old_unit.stage = UnitStage.COMPLETE
    lower_unit.tray_id = "V2_TRAY_01"
    lower_unit.stage = UnitStage.BRAZING
    top_unit.tray_id = "V2_TRAY_02"
    top_unit.stage = UnitStage.BRAZING
    physical._active_batch_units = [lower_unit.unit_id, top_unit.unit_id]
    for layer in physical.furnace.state.layers:
        layer.tray_id = None
        layer.locked = False
    physical.furnace.state.layers[1].tray_id = lower_unit.tray_id
    physical.furnace.state.layers[1].locked = True
    physical.furnace.state.layers[2].tray_id = top_unit.tray_id
    physical.furnace.state.layers[2].locked = True

    top_task = next(
        task
        for task in runtime.manufacturing_runtime.graph
        if task.unit_id == top_unit.unit_id and task.task_type is TaskType.UNLOAD_RACK_LAYER
    )
    lower_task = next(
        task
        for task in runtime.manufacturing_runtime.graph
        if task.unit_id == lower_unit.unit_id and task.task_type is TaskType.UNLOAD_RACK_LAYER
    )

    assert runtime.bridge.task_dispatch_allowed(top_task) == (True, "")
    assert runtime.bridge.task_dispatch_allowed(lower_task)[0] is False


def test_v2_five_order_burst_completes_in_two_physical_furnace_batches() -> None:
    args = parse_args(("--headless", "--fast", "--no-ui", "--orders", "A,B,C,D,A"))
    application = V2BrazingApplication(args)
    try:
        application.submit_cli_orders()
        for _ in range(7000):
            application.advance_frame()
            if application.runtime.complete:
                break
        state = application.publish(viewer_running=False)
        physical_events = list(application.runtime.physical_runtime.events)
    finally:
        application.close()

    assert state["complete"] is True
    assert state["physical_execution_complete"] is True
    assert state["completed_orders"] == [
        "V2_A_01",
        "V2_B_02",
        "V2_C_03",
        "V2_D_04",
        "V2_A_05",
    ]
    assert state["furnace"]["completed_batches"] == 2
    assert not [task for task in state["tasks"] if task["status"] not in {"SUCCEEDED", "CANCELLED"}]

    first_arm1_fin_unit = next(
        str(event["unit_id"])
        for event in physical_events
        if event["type"] == "OPERATION_STARTED"
        and event.get("resource") == "ARM1"
        and event.get("kind") == "INSTALL_FIN"
    )
    first_fin_unit_complete = max(
        float(event["time"])
        for event in physical_events
        if event["type"] == "FIN_INSTALLED"
        and event.get("unit_id") == first_arm1_fin_unit
        and int(event["fin_index"]) == int(event["fin_count"])
    )
    next_base_start = min(
        float(event["time"])
        for event in physical_events
        if event["type"] == "OPERATION_STARTED"
        and event.get("resource") == "ARM1"
        and event.get("kind") == "BASE_LOADING"
        and float(event["time"]) > first_fin_unit_complete
    )
    assert next_base_start - first_fin_unit_complete <= 3.0

    arm1_intervals: list[tuple[float, float]] = []
    active_arm1_start: float | None = None
    for event in physical_events:
        if event.get("resource") != "ARM1":
            continue
        if event["type"] == "OPERATION_STARTED":
            active_arm1_start = float(event["time"])
        elif event["type"] == "OPERATION_COMPLETED" and active_arm1_start is not None:
            arm1_intervals.append((active_arm1_start, float(event["time"])))
            active_arm1_start = None
    install_a_handoffs = [
        event
        for event in physical_events
        if event["type"] == "TRAY_HANDOFF" and event.get("target") == "INSTALL_A"
    ]
    for handoff in install_a_handoffs:
        arrived_at = float(handoff["time"])
        started_at = min(
            float(event["time"])
            for event in physical_events
            if event["type"] == "OPERATION_STARTED"
            and event.get("resource") == "ARM1"
            and event.get("kind") == "INSTALL_FIN"
            and event.get("unit_id") == handoff.get("unit_id")
            and float(event["time"]) >= arrived_at
        )
        occupied_until = max(
            (finished_at for began_at, finished_at in arm1_intervals if began_at <= arrived_at < finished_at),
            default=arrived_at,
        )
        assert started_at - occupied_until <= 1.0


def test_v2_runtime_keeps_processing_the_next_batch_while_the_furnace_runs() -> None:
    runtime = DualLineRuntime(fast=True)
    for index, preset in enumerate(("A", "B", "C", "A", "B", "C"), start=1):
        runtime.submit_order(preset, order_id=f"V2_{index}", priority=index)

    snapshot = runtime.run_until_complete(max_sim_time=300.0, dt=0.05)

    assert snapshot["complete"]
    assert snapshot["furnace"]["completed_batches"] == 2
    assert snapshot["metrics"]["upstream_work_during_brazing_s"] > 0.0
    assert snapshot["metrics"]["maximum_wip"] == 6


def test_s1_stages_the_next_tray_while_arm1_is_busy_installing_fins() -> None:
    runtime = DualLineRuntime(fast=True)
    runtime.submit_order("A", order_id="INSTALL_BRANCH_B")
    runtime.submit_order("B", order_id="INSTALL_IN_PROGRESS")

    for _ in range(2_000):
        runtime.tick(0.05)
        operation = runtime.operations.get("ARM1")
        if operation is not None and operation.kind == "INSTALL_FIN":
            break
    else:
        raise AssertionError("Arm1 did not reach fin installation")

    next_order = runtime.submit_order("B", order_id="S1_PRESTAGED")
    next_unit = runtime.units[next_order.unit_ids[0]]

    assert runtime.operations["ARM1"].kind == "INSTALL_FIN"
    assert next_unit.stage is UnitStage.BASE_LOADING
    assert next_unit.tray_id is not None
    assert runtime.flow.get(next_unit.tray_id).owner is TrayOwner.S1


def test_v2_loads_each_arriving_tray_top_down_before_the_batch_is_full() -> None:
    """The furnace is a live accumulator, not a three-place parking queue."""

    runtime = DualLineRuntime(fast=True)
    runtime.submit_order("A", order_id="INCREMENTAL", quantity=3)

    first_load_started_at = None
    third_buffer_arrived_at = None
    first_locked_layer = None
    for _ in range(4_000):
        runtime.tick(0.05)
        if third_buffer_arrived_at is None:
            buffer_arrivals = [
                event
                for event in runtime.events
                if event["type"] == "UNIT_STAGE" and event["stage"] == "FURNACE_BUFFER"
            ]
            if len(buffer_arrivals) >= 3:
                third_buffer_arrived_at = float(buffer_arrivals[2]["time"])
        if first_load_started_at is None:
            starts = [
                event
                for event in runtime.events
                if event["type"] == "OPERATION_STARTED" and event["kind"] == "FURNACE_LOAD_TRAY"
            ]
            if starts:
                first_load_started_at = float(starts[0]["time"])
        locked = [
            layer for layer in runtime.furnace.state.layers if layer.tray_id is not None and layer.locked
        ]
        if first_locked_layer is None and locked:
            first_locked_layer = max(locked, key=lambda layer: layer.index).index
        if runtime.furnace.state.phase in {
            FurnacePhase.PREHEAT,
            FurnacePhase.RAMP,
            FurnacePhase.SOAK,
        }:
            break

    assert first_load_started_at is not None
    assert third_buffer_arrived_at is not None
    assert first_load_started_at < third_buffer_arrived_at
    assert first_locked_layer == 2
    assert all(layer.locked for layer in runtime.furnace.state.layers)
    thermal_start = next(
        event["time"] for event in runtime.events if event["type"] == "FURNACE_THERMAL_CYCLE_STARTED"
    )
    load_completions = [
        event["time"]
        for event in runtime.events
        if event["type"] == "OPERATION_COMPLETED" and event["kind"] == "FURNACE_LOAD_TRAY"
    ]
    assert len(load_completions) == 3
    assert thermal_start > max(load_completions)


def test_v2_rear_door_stays_open_until_the_last_physical_unload_finishes() -> None:
    runtime = DualLineRuntime(fast=True)
    runtime.submit_order("A", order_id="REAR_DOOR", quantity=3)

    last_unload_started = False
    for _ in range(5_000):
        runtime.tick(0.05)
        transfer = runtime.operations.get("FURNACE_TRANSFER")
        occupied = [layer for layer in runtime.furnace.state.layers if layer.tray_id is not None]
        if transfer is not None and transfer.kind == "FURNACE_UNLOAD_TRAY" and not occupied:
            last_unload_started = True
            assert runtime.furnace.state.rear_door_open
            assert runtime.furnace.state.phase is FurnacePhase.UNLOADING
            assert not runtime.furnace.state.complete
            break

    assert last_unload_started


def test_v2_output_gate_wraps_each_delivery_and_closes_before_completion() -> None:
    runtime = DualLineRuntime(fast=True)
    runtime.submit_order("A", order_id="OUTPUT_GATE")
    snapshot = runtime.run_until_complete(max_sim_time=180.0, dt=0.05)

    events = snapshot["events"]
    opened = next(index for index, event in enumerate(events) if event["type"] == "OUTPUT_GATE_OPENED")
    output_handoff = next(
        index
        for index, event in enumerate(events)
        if event["type"] == "TRAY_HANDOFF" and event["target"] == "OUTPUT"
    )
    delivery_complete = next(
        index
        for index, event in enumerate(events)
        if event["type"] == "OPERATION_COMPLETED" and event["kind"] == "OUTPUT_DELIVERY"
    )
    closed = next(index for index, event in enumerate(events) if event["type"] == "OUTPUT_GATE_CLOSED")
    unit_complete = next(index for index, event in enumerate(events) if event["type"] == "UNIT_COMPLETED")
    assert opened < output_handoff < delivery_complete < closed < unit_complete
    assert snapshot["output"]["gate_open"] is False


def test_v2_pause_continue_and_reset_preserve_then_release_ownership() -> None:
    runtime = DualLineRuntime(fast=True)
    runtime.submit_order("C", order_id="PAUSE_TEST")
    for _ in range(30):
        runtime.tick(0.05)
    before = runtime.snapshot()
    runtime.pause()
    for _ in range(20):
        runtime.tick(0.05)
    paused = runtime.snapshot()
    assert paused["sim_time"] == before["sim_time"]
    assert paused["trays"] == before["trays"]

    runtime.continue_run()
    runtime.tick(0.05)
    assert runtime.snapshot()["sim_time"] > paused["sim_time"]
    runtime.reset()
    reset = runtime.snapshot()
    assert reset["sim_time"] == 0.0
    assert not reset["orders"]
    assert all(tray["stage"] == "EMPTY_BUFFER" for tray in reset["trays"])


def test_v2_urgent_order_wins_the_next_free_tray_without_preempting_current_work() -> None:
    runtime = DualLineRuntime(fast=True)
    runtime.submit_order("A", order_id="ACTIVE", priority=10)
    runtime.submit_order("B", order_id="NORMAL_WAIT", priority=99)
    runtime.submit_order(
        "C",
        order_id="URGENT_NEXT",
        priority=1,
        urgent=True,
    )

    for _ in range(300):
        runtime.tick(0.05)
        if runtime.units["URGENT_NEXT_UNIT_01"].stage is not UnitStage.QUEUED:
            break

    assert runtime.units["URGENT_NEXT_UNIT_01"].stage is not UnitStage.QUEUED
    assert runtime.units["NORMAL_WAIT_UNIT_01"].stage is UnitStage.QUEUED


def test_v2_due_risk_releases_a_partial_furnace_batch() -> None:
    runtime = DualLineRuntime(fast=True)
    runtime.submit_order(
        "A",
        order_id="DUE_FIRST",
        priority=30,
        due_at=1.0,
    )
    runtime.submit_order("C", order_id="LATER_UNIT", priority=1)

    for _ in range(1_200):
        runtime.tick(0.05)
        starts = [event for event in runtime.events if event["type"] == "FURNACE_BATCH_STARTED"]
        if starts:
            break

    assert starts[0]["unit_ids"] == ["DUE_FIRST_UNIT_01"]
    assert runtime.units["LATER_UNIT_UNIT_01"].stage not in {
        UnitStage.BRAZING,
        UnitStage.FURNACE_LOADING,
    }


def test_v2_headless_launcher_is_independent_and_emits_json_state() -> None:
    completed = subprocess.run(
        [
            sys.executable,
            str(ROOT / "brazing_line_v2.py"),
            "--headless",
            "--orders",
            "A,B,C",
            "--fast",
            "--max-sim-time",
            "180",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=90,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr + completed.stdout
    payload = json.loads(completed.stdout[completed.stdout.index("{") :])
    assert payload["schema_version"] == 2
    assert payload["line"] == "V2_DUAL_INSTALL"
    assert payload["complete"]
    assert payload["scene"]["compiled"]
    assert Path(payload["scene"]["xml"]).as_posix().endswith("scenes/production/brazing_line_v2.xml")
