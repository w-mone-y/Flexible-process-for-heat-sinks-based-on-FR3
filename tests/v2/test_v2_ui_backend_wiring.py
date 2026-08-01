"""Every V2 console control must reach real backend logic.

The console is shared with V1, so a V2 session used to show tabs and buttons
backed by hardcoded stubs: the gantt tab was permanently empty, the metrics tab
showed four zeros, the fixture row read "待接入", and several order fields were
accepted over HTTP and then silently dropped.  These tests pin the fix and, just
as importantly, pin the *honest* behaviour where V2 genuinely cannot comply.
"""

from __future__ import annotations

import pytest

from brazing_sim.api import validate_http_command
from brazing_sim.dual_line.application import V2ControlSurface
from brazing_sim.dual_line.presentation import V2StatePresenter
from brazing_sim.dual_line.runtime import DualLineRuntime
from brazing_sim.dual_line.tray_flow import TrayOwner
from brazing_sim.ui import line_ui_profile, manual_review_popup_state


def _custom_product() -> dict[str, object]:
    return {
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
    }


def _state(runtime: DualLineRuntime) -> dict:
    return V2StatePresenter().present(runtime.snapshot(), simulation_speed=1.0, actual_rtf=1.0)


def test_manual_review_popup_changes_from_waiting_to_success() -> None:
    waiting = manual_review_popup_state(
        {
            "sim_time": 3.0,
            "manual_review_notices": [
                {
                    "recovery_id": "R1",
                    "fault_label_zh": "机械臂暂时离线",
                    "status": "MANUAL_REVIEW",
                    "message": "发生机械臂暂时离线故障❌，需进行人工审核🔩🔧，请稍作等待⏰",
                    "complete_at": 10.0,
                }
            ],
        }
    )
    assert waiting == {
        "recovery_id": "R1",
        "status": "MANUAL_REVIEW",
        "message": "发生机械臂暂时离线故障❌，需进行人工审核🔩🔧，请稍作等待⏰\n预计剩余 7.0 秒",
    }

    succeeded = manual_review_popup_state(
        {
            "sim_time": 10.0,
            "manual_review_notices": [
                {
                    "recovery_id": "R1",
                    "status": "SUCCEEDED",
                    "message": "修改成功✅",
                    "complete_at": 10.0,
                }
            ],
        }
    )
    assert succeeded == {"recovery_id": "R1", "status": "SUCCEEDED", "message": "修改成功✅"}


def test_manual_review_dialog_has_full_message_and_does_not_close_control_panel(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Exercise the real Qt widget lifecycle that previously left only MuJoCo visible."""

    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    pytest.importorskip("PySide6")
    from PySide6.QtWidgets import QApplication, QWidget

    from brazing_sim.qt_manual_review import ManualReviewDialog

    application = QApplication.instance() or QApplication([])
    control_panel = QWidget()
    control_panel.show()
    dialog = ManualReviewDialog(control_panel)

    dialog.apply_popup(
        {
            "recovery_id": "R1",
            "status": "MANUAL_REVIEW",
            "message": "发生机械臂暂时离线故障❌，需进行人工审核🔩🔧，请稍作等待⏰\n预计剩余 9.8 秒",
        }
    )
    application.processEvents()
    assert dialog.isVisible()
    assert "发生机械臂暂时离线故障" in dialog.message_label.text()
    assert "预计剩余 9.8 秒" in dialog.message_label.text()
    assert not dialog.confirm_button.isVisible(), "等待阶段不应出现默认 OK 按钮"

    dialog.apply_popup({"recovery_id": "R1", "status": "SUCCEEDED", "message": "修改成功✅"})
    application.processEvents()
    assert dialog.confirm_button.isVisible()
    assert dialog.confirm_button.text() == "确定"
    assert dialog.message_label.text() == "修改成功✅"

    dialog.confirm_button.click()
    application.processEvents()
    assert not dialog.isVisible()
    assert control_panel.isVisible(), "确认修复结果后控制台不得随弹窗一起消失"


@pytest.fixture(scope="module")
def running_state():
    runtime = DualLineRuntime(fast=True)
    runtime.submit_order("A", order_id="WIRED", quantity=2)
    for _ in range(4000):
        runtime.tick(0.05)
        if runtime.complete:
            break
    return runtime, _state(runtime)


# ------------------------------------------------------------------ gantt tab


def test_gantt_tab_is_no_longer_empty(running_state):
    _runtime, state = running_state
    events = state["gantt_events"]
    assert events, "甘特图页此前恒为空表"
    row = events[0]
    assert {
        "task_id",
        "display_name_zh",
        "station_id",
        "status",
        "planned_duration",
        "actual_start",
        "actual_end",
    } <= set(row)
    assert row["display_name_zh"] != row["task_type"], "应显示中文工序名"
    assert all(item["actual_start"] is not None for item in events)


def test_gantt_durations_are_derived_from_real_event_times(running_state):
    _runtime, state = running_state
    finished = [item for item in state["gantt_events"] if item["actual_end"] is not None]
    assert finished
    for item in finished:
        assert item["planned_duration"] == pytest.approx(item["actual_end"] - item["actual_start"])


def test_cancelled_operations_are_visible_with_a_reason():
    runtime = DualLineRuntime(fast=True)
    runtime.submit_order("A", order_id="CANCEL", quantity=1)
    runtime.inject_fault("BRAZING_MISSING", target="slot_02_left")
    for _ in range(600):
        runtime.tick(0.05)
    cancelled = [item for item in _state(runtime)["gantt_events"] if item["status"] == "CANCELLED"]
    assert cancelled, "恢复回滚取消的操作应在甘特图中可见"
    assert cancelled[0]["blockers"]


# ---------------------------------------------------------------- metrics tab


def test_metrics_tab_keys_are_all_populated(running_state):
    """The console reads exactly these four; V2 produced none of them."""

    _runtime, state = running_state
    metrics = state["experiment_metrics"]
    for key in (
        "makespan",
        "throughput_per_sim_second",
        "average_robot_utilization",
        "recovery_rate",
    ):
        assert key in metrics, f"指标页读取的 {key} 缺失"
    assert metrics["makespan"] > 0.0
    assert metrics["throughput_per_sim_second"] > 0.0
    assert 0.0 < metrics["average_robot_utilization"] <= 1.0


def test_utilization_is_reported_per_arm(running_state):
    _runtime, state = running_state
    utilization = state["experiment_metrics"]["robot_utilization"]
    assert set(utilization) == {"ARM1", "ARM2", "ARM3"}
    assert all(0.0 <= value <= 1.0 for value in utilization.values())


def test_recovery_rate_follows_real_faults():
    runtime = DualLineRuntime(fast=True)
    runtime.submit_order("A", order_id="RATE", quantity=1)
    runtime.inject_fault("BRAZING_MISSING", target="slot_02_left")
    for _ in range(8000):
        runtime.tick(0.05)
        if runtime.complete:
            break
    metrics = _state(runtime)["experiment_metrics"]
    assert metrics["fault_count"] == 1
    assert metrics["recovery_rate"] == pytest.approx(1.0)


# -------------------------------------------------------------- placeholders


def test_fixture_row_reports_the_real_comb_module():
    runtime = DualLineRuntime(fast=True)
    idle = _state(runtime)["fixture"]
    assert idle["active_comb_module"] is None
    assert "待接入" not in idle["status"]

    runtime.submit_order("C", order_id="FIX_C", quantity=1)
    for _ in range(2000):
        runtime.tick(0.05)
        fixture = _state(runtime)["fixture"]
        if fixture["active_comb_module"]:
            break
    # C uses the 15 mm comb; the row must say so rather than "待接入".
    assert fixture["active_comb_module"] == "comb_insert_15mm"
    assert "C" in fixture["status"]


def test_zone_locks_reflect_tray_ownership(running_state):
    _runtime, state = running_state
    runtime = DualLineRuntime(fast=True)
    runtime.submit_order("A", order_id="ZONE", quantity=2)
    for _ in range(600):
        runtime.tick(0.05)
    locks = _state(runtime)["zone_locks"]
    assert locks, "V2 通过托盘唯一归属实现互斥，占用工位即为持锁区域"
    assert all(item["holder"] for item in locks.values())


def test_every_tray_owner_maps_to_a_station():
    """An unmapped owner would silently disappear from the zone view."""

    from brazing_sim.dual_line.presentation import _OWNER_TO_STATION

    missing = [
        owner.value
        for owner in TrayOwner
        if owner is not TrayOwner.EMPTY_BUFFER and owner.value not in _OWNER_TO_STATION
    ]
    assert missing == []


def test_peak_parallel_arms_is_measured(running_state):
    _runtime, state = running_state
    parallelism = state["async_line"]["parallelism"]
    # Two install branches mean the peak must exceed one.
    assert parallelism["max_parallel_arms"] >= 2


# ---------------------------------------------------- honest command refusals


def test_dropped_order_fields_are_reported_not_swallowed():
    """Silently ignoring a setting is worse than refusing it."""

    runtime = DualLineRuntime(fast=True)
    surface = V2ControlSurface(runtime)
    command = validate_http_command(
        "/orders/insert",
        {
            "mode": "preset",
            "preset": "A",
            "order_id": "DROP",
            "quantity": 1,
            "priority": 10,
            "preferred_rack_layer": 2,
        },
    )
    with pytest.raises(ValueError) as excinfo:
        surface.process(command)
    assert "未生效" in str(excinfo.value)
    assert surface.ignored_order_fields
    # The order itself is still accepted; only the unsupported field is refused.
    assert runtime.orders


def test_supported_order_fields_do_not_warn():
    runtime = DualLineRuntime(fast=True)
    surface = V2ControlSurface(runtime)
    surface.process(
        {
            "type": "order_insert",
            "preset": "B",
            "order_id": "CLEAN",
            "quantity": 2,
            "priority": 5,
            "urgent": True,
        }
    )
    assert not surface.ignored_order_fields
    assert runtime.orders["CLEAN"].urgent is True
    assert len(runtime.orders["CLEAN"].unit_ids) == 2


def test_custom_order_insert_reaches_the_v2_runtime() -> None:
    runtime = DualLineRuntime(fast=True)
    surface = V2ControlSurface(runtime)

    surface.process(
        {
            "type": "order_insert",
            "mode": "custom",
            "preset": "CUSTOM",
            "order_id": "CUSTOM_WIRED",
            "quantity": 1,
            "priority": 21,
            "route_strategy": "STANDARD",
            "custom_product": _custom_product(),
        }
    )

    order = runtime.orders["CUSTOM_WIRED"]
    unit = runtime.units[order.unit_ids[0]]
    assert order.product_id == "CUSTOM_CUSTOM_WIRED"
    assert unit.preset == "CUSTOM"
    assert unit.fin_count == 6
    assert len(unit.process_geometry.fin_targets) == 6
    assert len(unit.process_geometry.brazing_paths) == 12
    assert unit.comb_module == "comb_insert_20mm"
    assert unit.target_clamping_force_n == pytest.approx(24.0)


def test_v2_ui_profile_exposes_custom_order_mode() -> None:
    assert line_ui_profile("V2").supports_custom_orders


def test_custom_fixture_parameters_are_visible_in_the_v2_console_state() -> None:
    runtime = DualLineRuntime(fast=True)
    V2ControlSurface(runtime).process(
        {
            "type": "order_insert",
            "mode": "custom",
            "order_id": "CUSTOM_FIXTURE",
            "quantity": 1,
            "priority": 10,
            "custom_product": _custom_product(),
        }
    )
    fixture = _state(runtime)["fixture"]
    for _ in range(2_000):
        runtime.tick(0.05)
        fixture = _state(runtime)["fixture"]
        if fixture["active_comb_module"]:
            break

    assert fixture["active_comb_module"] == "comb_insert_20mm"
    assert "CUSTOM" in fixture["status"]


def test_batch_command_is_supported():
    """The backend accepted ``batch`` all along; only the UI lacked an entry."""

    runtime = DualLineRuntime(fast=True)
    V2ControlSurface(runtime).process({"type": "batch", "preset": "A", "layers": 3})
    assert len(next(iter(runtime.orders.values())).unit_ids) == 3


def test_speed_and_lifecycle_commands_still_work():
    runtime = DualLineRuntime(fast=True)
    surface = V2ControlSurface(runtime)
    surface.process({"type": "speed", "action": "accelerate"})
    assert surface.simulation_speed == pytest.approx(2.0)
    surface.process({"type": "speed", "action": "decelerate"})
    assert surface.simulation_speed == pytest.approx(1.0)
    surface.process({"type": "stop"})
    assert runtime.paused
    surface.process({"type": "continue"})
    assert not runtime.paused
    surface.process({"type": "reset"})
    assert not runtime.units


def test_unsupported_segment_still_refuses_clearly():
    surface = V2ControlSurface(DualLineRuntime(fast=True))
    with pytest.raises(RuntimeError) as excinfo:
        surface.process({"type": "segment", "segment": "v2_base_loading"})
    assert "尚未接通" in str(excinfo.value)
