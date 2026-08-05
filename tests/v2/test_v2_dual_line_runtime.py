from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import pytest

from brazing_sim.dual_line import DualLineRuntime, FurnacePhase, UnitStage
from brazing_sim.flexible import build_custom_plan

ROOT = Path(__file__).resolve().parents[2]


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


def test_v2_runtime_keeps_processing_the_next_batch_while_the_furnace_runs() -> None:
    runtime = DualLineRuntime(fast=True)
    for index, preset in enumerate(("A", "B", "C", "A", "B", "C"), start=1):
        runtime.submit_order(preset, order_id=f"V2_{index}", priority=index)

    snapshot = runtime.run_until_complete(max_sim_time=300.0, dt=0.05)

    assert snapshot["complete"]
    assert snapshot["furnace"]["completed_batches"] == 2
    assert snapshot["metrics"]["upstream_work_during_brazing_s"] > 0.0
    assert snapshot["metrics"]["maximum_wip"] == 6


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
        occupied = [layer for layer in runtime.furnace.state.layers if layer.tray_id is not None]
        if first_locked_layer is None and occupied and occupied[0].locked:
            first_locked_layer = occupied[0].index
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
