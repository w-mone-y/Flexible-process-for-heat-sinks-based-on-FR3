"""Realtime V2 task projection and camera-discovered physical faults."""

from __future__ import annotations

import pytest

from brazing_sim.api import validate_http_command
from brazing_sim.dual_line.application import V2ControlSurface
from brazing_sim.dual_line.presentation import V2StatePresenter
from brazing_sim.dual_line.runtime import DualLineRuntime


def _present(runtime: DualLineRuntime) -> dict:
    return V2StatePresenter().present(
        runtime.snapshot(),
        simulation_speed=1.0,
        actual_rtf=1.0,
    )


def test_running_task_publishes_continuous_progress_without_stage_change() -> None:
    runtime = DualLineRuntime(fast=True)
    runtime.submit_order("A", order_id="LIVE_TASK")

    before = next(task for task in _present(runtime)["tasks"] if task["task_type"] == "LOAD_BASE")
    assert before["status"] == "RUNNING"
    assert before["progress"] == pytest.approx(0.0)

    runtime.tick(0.10)
    after = next(task for task in _present(runtime)["tasks"] if task["task_id"] == before["task_id"])
    assert after["status"] == "RUNNING"
    assert after["progress"] > before["progress"]
    assert after["updated_at"] > before["updated_at"]
    assert "执行进度" in after["display_detail_zh"]


def test_brazing_defect_manifests_before_camera_detection_and_rework() -> None:
    runtime = DualLineRuntime(fast=True)
    runtime.submit_order("A", order_id="LATENT_BRAZE")
    runtime.inject_fault("BRAZING_MISSING", target="path_02")

    for _ in range(300):
        runtime.tick(0.02)
        request = next(iter(runtime.faults.pending.values()))
        if request.status == "MANIFESTED":
            break
    else:
        raise AssertionError("漏涂应在涂覆工序中先形成物理缺陷")

    snapshot = runtime.snapshot()
    request = snapshot["manual_fault_requests"][0]
    defect = snapshot["physical_faults"][0]
    assert request["status"] == "MANIFESTED"
    assert request["manifested_at"] is not None
    assert request["detected_at"] is None
    assert defect["status"] == "MANIFESTED"
    assert defect["target"] == "path_02"
    assert snapshot["faults_v2"] == [], "相机尚未检测时不得提前生成恢复计划"

    for _ in range(600):
        runtime.tick(0.02)
        request = next(iter(runtime.faults.pending.values()))
        if request.detected_at is not None:
            break
    else:
        raise AssertionError("焊料检测相机应发现已经存在的漏涂")

    assert request.detected_at > request.manifested_at
    assert runtime.faults.faults
    manifested = next(event for event in runtime.events if event["type"] == "FAULT_MANIFESTED")
    detected = next(event for event in runtime.events if event["type"] == "FAULT_DETECTED")
    assert manifested["time"] < detected["time"]


def test_brazing_rework_is_scoped_to_the_exact_path_not_the_whole_board() -> None:
    runtime = DualLineRuntime(fast=True)
    runtime.submit_order("A", order_id="LOCAL_SCOPE")
    runtime.inject_fault("BRAZING_MISSING", target="path_02")

    repair = None
    for _ in range(2_000):
        runtime.tick(0.02)
        repair = next(
            (
                operation
                for operation in runtime.operations.values()
                if operation.recovery and operation.kind == "DISPENSING"
            ),
            None,
        )
        if repair is not None:
            break
    assert repair is not None
    assert repair.recovery_strategy == "LOCAL_BRAZING_REWORK"
    assert repair.recovery_target_index == 2
    assert repair.duration_s < runtime.durations.dispensing


def test_exact_fin_pose_target_does_not_manifest_on_an_earlier_fin() -> None:
    runtime = DualLineRuntime(fast=True)
    runtime.submit_order("A", order_id="TARGET_FIN")
    runtime.inject_fault("FIN_POSE", target="fin_03")

    for _ in range(2_000):
        runtime.tick(0.02)
        request = next(iter(runtime.faults.pending.values()))
        if request.status == "MANIFESTED":
            break
    else:
        raise AssertionError("第三片翅片故障未形成")

    unit = runtime.units["TARGET_FIN_UNIT_01"]
    defect = next(iter(runtime.faults.physical_faults.values()))
    assert unit.fins_installed >= 2
    assert defect.target == "fin_03"
    assert defect.operation_index == 3


@pytest.mark.parametrize(
    "fault,target,expected_source,expected_target",
    (
        ("BRAZING_MISSING", "path_02", "S2B", "S2A"),
        ("BRAZING_PATH_DEVIATION", "path_03", "S2B", "S2A"),
        ("FIN_POSE", "fin_04", "S4", None),
    ),
)
def test_quality_rework_physically_returns_to_the_process_station(
    fault: str,
    target: str,
    expected_source: str,
    expected_target: str | None,
) -> None:
    runtime = DualLineRuntime(fast=True)
    runtime.submit_order("A", order_id=f"RETURN_{fault}")
    runtime.inject_fault(fault, target=target)
    for _ in range(10_000):
        runtime.tick(0.02)
        if runtime.complete:
            break
    assert runtime.complete

    returned = next(event for event in runtime.events if event["type"] == "RECOVERY_RETURN_STARTED")
    assert returned["source"] == expected_source
    if expected_target is None:
        assert returned["target"] in {"INSTALL_A", "INSTALL_B"}
    else:
        assert returned["target"] == expected_target
    repaired = next(event for event in runtime.events if event["type"] == "FAULT_REPAIRED")
    detected = next(event for event in runtime.events if event["type"] == "FAULT_DETECTED")
    assert detected["time"] < repaired["time"]
    assert runtime.snapshot()["manual_fault_requests"][0]["status"] == "RECOVERED"


def test_v2_catalog_aliases_are_executable_not_dead_buttons() -> None:
    runtime = DualLineRuntime(fast=True)
    surface = V2ControlSurface(runtime)
    runtime.submit_order("A", order_id="ALIASES")

    fin_pose = validate_http_command(
        "/faults/inject",
        {"fault_type": "FIN_POSE", "target": "fin_01"},
    )
    surface.process(fin_pose)
    assert next(iter(runtime.faults.pending.values())).visual_type == "FIN_POSE"

    furnace = validate_http_command(
        "/faults/inject",
        {"fault_type": "FURNACE_PROFILE", "target": "furnace", "severity": "severe"},
    )
    surface.process(furnace)
    assert any(item.visual_type == "FURNACE_PROFILE" for item in runtime.faults.pending.values())


def test_flexibility_demo_commands_mutate_the_real_v2_runtime() -> None:
    command = validate_http_command("/flexibility/demo", {"demo": "product_mix"})
    runtime = DualLineRuntime(fast=True)
    V2ControlSurface(runtime).process(command)
    assert [order.preset for order in runtime.orders.values()] == ["A", "B", "C"]

    disturbed = DualLineRuntime(fast=True)
    V2ControlSurface(disturbed).process(validate_http_command("/flexibility/demo", {"demo": "fault_loop"}))
    assert disturbed.orders
    assert any(item.visual_type == "BRAZING_MISSING" for item in disturbed.faults.pending.values())


def test_v2_rejects_unsupported_route_strategy_before_accepting_order() -> None:
    runtime = DualLineRuntime(fast=True)
    surface = V2ControlSurface(runtime)
    with pytest.raises(ValueError, match="不支持"):
        surface.process(
            {
                "type": "order_insert",
                "preset": "A",
                "quantity": 1,
                "route_strategy": "TURBO",
            }
        )
    assert runtime.orders == {}
