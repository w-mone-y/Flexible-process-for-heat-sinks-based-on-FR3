from __future__ import annotations

from datetime import datetime, timedelta

import pytest

from brazing_sim.dual_line import DualLineRuntime
from brazing_sim.dual_line.application import V2BrazingApplication, V2ControlSurface
from brazing_sim.dual_line.cli import parse_args
from brazing_sim.dual_line.presentation import V2StatePresenter
from brazing_sim.api import validate_http_command
from brazing_sim.ui import initial_control_panel_size, line_ui_profile


def test_v2_viewer_starts_idle_and_only_explicit_headless_orders_are_submitted() -> None:
    viewer = parse_args([])
    explicit_headless = parse_args(["--headless", "--orders", "A,B"])

    assert viewer.order_presets == ()
    assert explicit_headless.order_presets == ("A", "B")


def test_v2_cli_exposes_twinshield_authority_and_operator_rollback_modes() -> None:
    default = parse_args(["--headless"])
    rollback = parse_args(["--headless", "--twinshield-mode", "fallback"])

    assert default.twinshield_mode == "AUTHORITY"
    assert rollback.twinshield_mode == "FALLBACK"


def test_v2_reuses_the_full_v1_console_with_dual_branch_controls() -> None:
    profile = line_ui_profile("V2_DUAL_INSTALL")

    assert profile.tab_titles == (
        "运行总览",
        # Six-dimension flexibility evidence, shared by both lines.
        "柔性总览",
        "订单规划",
        "任务图 / 调度",
        "异步流水工位",
        "实时甘特图",
        "产品工程图规划",
        "资源与区域",
        "故障与恢复规划",
        "批次与物流",
        "指标与实验",
    )
    assert [action.segment for action in profile.segment_actions] == [
        "v2_base_loading",
        "v2_dispensing",
        "v2_material_inspection",
        "v2_install_a",
        "v2_install_b",
        "v2_parallel_install",
        "v2_merge_inspection",
        "v2_furnace_batch",
        "v2_post_braze_delivery",
    ]
    assert profile.station_titles["S3A_ARM1_INSTALL"] == "S3A Arm1 翅片安装"
    assert profile.station_titles["S3B_ARM3_INSTALL"] == "S3B Arm3 翅片安装"


def test_control_panel_uses_nearly_the_full_available_desktop_width() -> None:
    """The initial console should expose horizontal content without left-right scrolling."""

    assert initial_control_panel_size(1512, 982) == (1451, 785)
    width, height = initial_control_panel_size(1920, 1080)
    assert (width, height) == (1843, 850)
    assert width <= 1920
    assert height <= 1080


def test_v2_state_presenter_matches_the_shared_ui_contract_without_fake_capabilities() -> None:
    runtime = DualLineRuntime(fast=True)
    presenter = V2StatePresenter()

    state = presenter.present(runtime.snapshot(), simulation_speed=1.0, actual_rtf=0.0)

    assert state["line_profile"] == "V2_DUAL_INSTALL"
    assert state["status"] == "idle"
    assert state["stage"] == "IDLE"
    assert state["orders"] == []
    assert set(state["arms"]) == {"arm1", "arm2", "arm3"}
    assert {
        "S1_BASE_LOADING",
        "S2A_DISPENSING",
        "S2B_MATERIAL_INSPECTION",
        "S3A_ARM1_INSTALL",
        "S3B_ARM3_INSTALL",
        "S4_PRE_BRAZE_INSPECTION",
    }.issubset(state["workstations"])
    assert not any(state["ui_capabilities"]["segments"].values())
    assert not state["ui_capabilities"]["custom_orders"]
    # V2 now owns a real fault controller, so the console must advertise it.
    # The remaining gap is MuJoCo fault *visuals*, not the capability itself.
    assert state["ui_capabilities"]["fault_injection"]
    assert state["faults_v2"] == []
    assert state["recoveries"] == []
    assert state["manual_fault_requests"] == []
    assert state["physical_execution_complete"] is False


def test_v2_task_graph_uses_chinese_micro_tasks_and_updates_each_fin_immediately() -> None:
    runtime = DualLineRuntime(fast=True)
    presenter = V2StatePresenter()
    runtime.submit_order("B", order_id="TASK_GRAPH_B")

    for _ in range(2_000):
        runtime.tick(0.05)
        unit = runtime.units["TASK_GRAPH_B_UNIT_01"]
        if unit.fins_installed >= 1:
            break
    else:
        raise AssertionError("first fin was not installed")

    state = presenter.present(
        runtime.snapshot(),
        simulation_speed=1.0,
        actual_rtf=1.0,
    )
    fin_tasks = [task for task in state["tasks"] if task["task_type"] == "INSTALL_FIN"]

    assert fin_tasks[0]["display_name_zh"] == "安装第 1 / 4 片翅片"
    assert fin_tasks[0]["status"] == "SUCCEEDED"
    assert fin_tasks[0]["status_zh"] == "已完成"
    assert fin_tasks[1]["status"] in {"READY", "RUNNING"}
    assert state["resources_v2"]["ARM2"]["current_tool"] == "固定双喷嘴焊料枪"


def test_v2_task_graph_does_not_claim_arm3_before_install_branch_assignment() -> None:
    runtime = DualLineRuntime(fast=True)
    presenter = V2StatePresenter()
    runtime.submit_order("A", order_id="PENDING_BRANCH")

    state = presenter.present(
        runtime.snapshot(),
        simulation_speed=1.0,
        actual_rtf=1.0,
    )
    fin_task = next(task for task in state["tasks"] if task["task_type"] == "INSTALL_FIN")

    assert fin_task["station_id"] == "INSTALL_BRANCH_PENDING"
    assert fin_task["eligible_resources"] == ["ARM1_OR_ARM3"]
    assert fin_task["assigned_resource"] is None
    assert "等待支路分配" in fin_task["display_detail_zh"]


def test_v2_control_surface_drives_orders_speed_pause_continue_and_reset() -> None:
    runtime = DualLineRuntime(fast=True)
    controls = V2ControlSurface(runtime)

    controls.process(
        {
            "type": "order_insert",
            "order_id": "UI_V2_A",
            "preset": "A",
            "quantity": 1,
            "priority": 20,
        }
    )
    assert list(runtime.orders) == ["UI_V2_A"]

    for _ in range(6):
        controls.process({"type": "speed", "action": "accelerate"})
    assert controls.simulation_speed == 32.0
    controls.process({"type": "stop"})
    assert runtime.paused
    controls.process({"type": "continue"})
    assert not runtime.paused

    with pytest.raises(RuntimeError, match="尚未接通"):
        controls.process({"type": "segment", "segment": "v2_install_a"})

    controls.process({"type": "reset"})
    assert runtime.orders == {}
    assert controls.simulation_speed == 1.0


def test_v2_control_surface_preserves_urgent_and_iso_due_time_semantics() -> None:
    runtime = DualLineRuntime(fast=True)
    controls = V2ControlSurface(runtime)
    due_time = datetime.now().astimezone() + timedelta(minutes=30)

    controls.process(
        {
            "type": "order_insert",
            "order_id": "UI_V2_URGENT",
            "preset": "C",
            "quantity": 1,
            "priority": 25,
            "due_time": due_time.isoformat(),
            "urgent": True,
        }
    )

    unit = runtime.units["UI_V2_URGENT_UNIT_01"]
    order = runtime.snapshot()["orders"][0]
    presented = V2StatePresenter().present(
        runtime.snapshot(),
        simulation_speed=1.0,
        actual_rtf=1.0,
    )
    assert unit.urgent
    assert unit.due_at == pytest.approx(1_800.0, abs=2.0)
    assert order["urgent"] is True
    assert presented["orders"][0]["urgent"] is True


def test_v2_presenter_exposes_install_selection_candidates_and_explanation() -> None:
    runtime = DualLineRuntime(fast=True)
    presenter = V2StatePresenter()
    runtime.submit_order("A", order_id="EXPLAIN_BRANCH")

    for _ in range(400):
        runtime.tick(0.05)
        if any(event["type"] == "INSTALL_ASSIGNED" for event in runtime.events):
            break

    state = presenter.present(
        runtime.snapshot(),
        simulation_speed=1.0,
        actual_rtf=1.0,
    )
    scheduler = state["scheduler"]
    assert scheduler["selected"]
    assert {item["resource_id"] for item in scheduler["candidates"]} == {
        "ARM1_A",
        "ARM3_B",
    }
    assert scheduler["selected"][0]["explanation_zh"]


def test_shared_http_api_accepts_v2_segment_names_for_the_v2_backend() -> None:
    for segment in (
        "v2_base_loading",
        "v2_dispensing",
        "v2_material_inspection",
        "v2_install_a",
        "v2_install_b",
        "v2_parallel_install",
        "v2_merge_inspection",
        "v2_furnace_batch",
        "v2_post_braze_delivery",
    ):
        assert validate_http_command("/segment", {"segment": segment}) == {
            "type": "segment",
            "segment": segment,
        }


def test_v2_application_publishes_real_carrier_transport_to_the_shared_ui() -> None:
    args = parse_args(["--headless", "--orders", "A", "--fast"])
    application = V2BrazingApplication(args)
    try:
        application.submit_cli_orders()
        application.advance_frame()
        state = application.publish(viewer_running=False)

        assert state["line_profile"] == "V2_DUAL_INSTALL"
        assert state["transfers"]["V2_TRAY_01"]["moving"]
        assert state["transfers"]["V2_TRAY_01"]["target"] == "S1"
        assert state["async_line"]["physical_tray_owners"]["V2_TRAY_01"] == "EMPTY_BUFFER"
        assert state["tray_routes"]["V2_TRAY_01"]["physical_owner"] == "EMPTY_BUFFER"
        assert state["prepositioning"]["ARM1"]["operation_kind"] == "BASE_LOADING"
        assert state["arms"]["arm1"]["status"] == "prepositioning"
        assert state["arms"]["arm1"]["preposition_for"] == "BASE_LOADING"
        assert state["ui_capabilities"]["orders"]
        assert state["twinshield"]["mode"] == "AUTHORITY"
        assert "decision_latency_ms" in state["twinshield"]
    finally:
        application.close()


def test_v2_application_publishes_planned_motion_and_space_time_reservations() -> None:
    """V2 must expose the scheduler's PRM/SIPP authority, not only arm playback."""

    args = parse_args(["--headless", "--orders", "A", "--fast"])
    application = V2BrazingApplication(args)
    try:
        application.submit_cli_orders()
        for _ in range(40):
            application.advance_frame()
            state = application.publish(viewer_running=False)
            planned = [
                item for item in state["motion_plans"] if item.get("planner") in {"PRM_ASTAR", "RRT_CONNECT"}
            ]
            if planned and state["space_time_reservations"]:
                break
        else:
            raise AssertionError("V2 did not publish an active motion reservation")

        reservation = state["space_time_reservations"][0]
        assert planned[0]["reservation_id"] == reservation["reservation_id"]
        assert planned[0]["resource_id"] == "ARM1"
        assert any(cell.startswith("CELL_V2_S1") for cell, _start, _end in reservation["occupied_cells"])
    finally:
        application.close()


def test_v2_comb_configuration_is_stage_driven_without_visible_changeover_gantry() -> None:
    args = parse_args(["--headless", "--orders", "D", "--fast"])
    application = V2BrazingApplication(args)
    try:
        assert (
            application.scene.mujoco.mj_name2id(
                application.scene.model,
                application.scene.mujoco.mjtObj.mjOBJ_BODY,
                "v2_shared_changeover_gantry",
            )
            == -1
        )
        application.submit_cli_orders()
        for _ in range(1_200):
            application.advance_frame()
            state = application.publish(viewer_running=False)
            configured = next(task for task in state["tasks"] if task["task_type"] == "CONFIGURE_COMB")
            if configured["status"] == "SUCCEEDED":
                break
        else:
            raise AssertionError("V2 stage-driven comb configuration did not complete")

        unit = application.runtime.units["V2_D_01_UNIT_01"]
        assert unit.tray_id is not None
        assert application.scene.component_visible(unit.tray_id, "front_comb_base")
        assert "changeover" not in state
        assert "module_ownership" not in state
        assert state["ui_capabilities"]["flexibility_actions"]["physical_changeover"] is False
    finally:
        application.close()


def test_v2_physical_execution_complete_requires_every_terminal_gate() -> None:
    args = parse_args(["--headless", "--orders", "A", "--fast"])
    application = V2BrazingApplication(args)
    try:
        initial = application.publish(viewer_running=False)
        assert initial["physical_execution_complete"] is False
        assert initial["physical_completion_gates"]["passed"] is False
        assert "manufacturing_terminal" in initial["physical_completion_gates"]["failed_checks"]

        application.submit_cli_orders()
        for _ in range(4_500):
            application.advance_frame()
            if application.runtime.complete:
                final = application.publish(viewer_running=False)
                break
        else:
            raise AssertionError("single A did not physically complete")

        gates = final["physical_completion_gates"]
        assert final["physical_execution_complete"] is True
        assert gates["passed"] is True
        assert gates["failed_checks"] == []
        assert all(item["passed"] for item in gates["checks"].values())
        assert {
            "manufacturing_terminal",
            "task_outcomes",
            "physical_runtime_terminal",
            "operations_idle",
            "tray_transport_settled",
            "motion_reservations_released",
            "furnace_and_output_safe",
            "tray_ownership_safe",
            "faults_resolved",
            "robots_settled",
            "physical_gate_bound",
        }.issubset(gates["checks"])
        assert "changeover_safe" not in gates["checks"]
    finally:
        application.close()
