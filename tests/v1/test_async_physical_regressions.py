"""Focused regressions for visible asynchronous line motion."""

from __future__ import annotations

import math
import time
from types import SimpleNamespace

import numpy as np
import pytest

from brazing_line import BrazingApplication, parse_args
from brazing_sim.execution.async_line_skills import AsyncLinePhysicalSkill, _TransferState
from brazing_sim.flexible import build_inline_plan
from brazing_sim.layout import SHALLOW_U_LAYOUT
from brazing_sim.planning import build_task_graph
from brazing_sim.planning.task_models import TaskStatus, TaskType


class _ImmediateTransferRegistry:
    """Minimal actuator double that follows each commanded setpoint exactly."""

    def __init__(self) -> None:
        self.position = 0.0
        self.commands: list[float] = []

    def async_transfer_limit(self, _transfer_id: str) -> float:
        return 1.0

    def set_async_transfer_target(self, _transfer_id: str, position: float) -> None:
        self.position = float(position)
        self.commands.append(self.position)

    def async_transfer_position(self, _transfer_id: str) -> float:
        return self.position

    def async_transfer_velocity(self, _transfer_id: str) -> float:
        return 0.0

    def finish_batch_tray_async_transfer(
        self,
        _tray_id: str,
        _transfer_id: str,
        _destination: str,
    ) -> None:
        return None


class _NoopHttpServer:
    server_address = ("127.0.0.1", 0)

    def shutdown(self) -> None:
        return None

    def server_close(self) -> None:
        return None


def test_loaded_transfer_uses_gradual_quintic_commands_instead_of_endpoint_jump() -> None:
    registry = _ImmediateTransferRegistry()
    skill = AsyncLinePhysicalSkill(TaskType.TRANSFER_S1_S2A)
    skill.context = SimpleNamespace(scene=SimpleNamespace(registry=registry))
    skill.task = SimpleNamespace(tray_id="tray_01")
    skill.transfer = _TransferState(
        transfer_id="s1_s2a",
        destination="s2a",
        started_at=0.0,
        start_position=0.0,
        duration_s=4.0,
        return_duration_s=2.5,
    )

    progress = []
    for now in (0.0, 0.5, 1.0, 2.0, 3.0):
        result = skill._update_transfer(now)
        progress.append(float(result.metrics["progress"]))

    assert registry.commands[0] == pytest.approx(0.0)
    assert registry.commands[1] < 0.05
    assert registry.commands[-1] == pytest.approx(0.896484375)
    assert registry.commands == sorted(registry.commands)
    assert progress == sorted(progress)


def test_s3_fin_install_and_output_camera_share_their_real_swept_zone() -> None:
    graph = build_task_graph(
        build_inline_plan(
            preset="B",
            order_id="INTERARM_ZONE",
            quantity=1,
            priority=10,
        ),
        flexible_cell=True,
    )
    installs = [task for task in graph if task.task_type is TaskType.INSTALL_FIN]
    output_views = [
        task
        for task in graph
        if task.task_type in {TaskType.POST_BRAZE_INSPECTION, TaskType.SECOND_POST_BRAZE_VIEW}
    ]

    assert installs
    assert output_views
    assert all("ZONE_S3_OUTPUT_INTERARM" in task.required_zones for task in installs)
    assert all("ZONE_S3_OUTPUT_INTERARM" in task.required_zones for task in output_views)


def test_first_fin_pick_overlaps_comb_install_but_insertion_waits_for_lock() -> None:
    graph = build_task_graph(
        build_inline_plan(
            preset="A",
            order_id="COMB_FIN_PIPELINE",
            quantity=1,
            priority=10,
        ),
        flexible_cell=True,
    )
    configure = next(task for task in graph if task.task_type is TaskType.CONFIGURE_COMB)
    first_pick = next(
        task
        for task in graph
        if task.task_type is TaskType.PICK_FIN and task.payload.get("fin_id") == "fin_01"
    )
    first_install = next(
        task
        for task in graph
        if task.task_type is TaskType.INSTALL_FIN and task.payload.get("fin_id") == "fin_01"
    )

    assert configure.task_id not in first_pick.predecessors
    assert configure.task_id in first_install.predecessors
    assert first_pick.task_id in first_install.predecessors


def test_comb_completion_has_no_ui_block_before_first_fin_install(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """First-fin IK is bounded and its pickup is prefetched during comb motion."""

    monkeypatch.setattr(
        "brazing_line.start_http_server",
        lambda *_args, **_kwargs: _NoopHttpServer(),
    )
    application = BrazingApplication(
        parse_args(
            [
                "--headless",
                "--dt",
                "0.02",
                "--no-ui",
                "--no-terminal-commands",
                "--port",
                "0",
            ]
        )
    )
    try:
        application.process_command(
            {
                "type": "order_insert",
                "mode": "preset",
                "preset": "A",
                "order_id": "COMB_FIN_NO_GAP",
                "quantity": 1,
                "priority": 10,
                "route_strategy": "STANDARD",
                "urgent": False,
            }
        )
        first_planning_wall_s: float | None = None
        configure = first_pick = first_install = None
        for _ in range(6_000):
            application.tick()
            configure = next(
                (
                    task
                    for task in application.manufacturing_runtime.graph
                    if task.task_type is TaskType.CONFIGURE_COMB
                ),
                None,
            )
            first_pick = next(
                (
                    task
                    for task in application.manufacturing_runtime.graph
                    if task.task_type is TaskType.PICK_FIN and task.payload.get("fin_id") == "fin_01"
                ),
                None,
            )
            first_install = next(
                (
                    task
                    for task in application.manufacturing_runtime.graph
                    if task.task_type is TaskType.INSTALL_FIN and task.payload.get("fin_id") == "fin_01"
                ),
                None,
            )
            execution = next(
                (
                    item
                    for item in application.manufacturing_runtime.executor.active.values()
                    if item.task.task_type is TaskType.PICK_FIN
                    and item.task.payload.get("fin_id") == "fin_01"
                ),
                None,
            )
            if (
                first_planning_wall_s is None
                and execution is not None
                and execution.skill.arm_stages
                and execution.skill.arm_stages[0].joint7_value is None
            ):
                started = time.perf_counter()
                application.tick()
                first_planning_wall_s = time.perf_counter() - started
            if first_install is not None and first_install.status is TaskStatus.RUNNING:
                break

        assert configure is not None and configure.finished_at is not None
        assert first_pick is not None and first_pick.finished_at is not None
        assert first_install is not None and first_install.started_at is not None
        assert first_planning_wall_s is not None
        assert first_planning_wall_s < 0.50
        assert first_pick.finished_at <= configure.finished_at
        assert first_install.started_at - configure.finished_at <= 0.020001
    finally:
        application.close()


def test_loaded_transfers_reserve_both_endpoint_junctions() -> None:
    graph = build_task_graph(
        build_inline_plan(
            preset="C",
            order_id="TRANSFER_JUNCTIONS",
            quantity=1,
            priority=10,
        ),
        flexible_cell=True,
    )
    transfers = {
        task.task_type: set(task.required_zones)
        for task in graph
        if task.task_type
        in {
            TaskType.TRANSFER_S1_S2A,
            TaskType.TRANSFER_S2A_S2B,
            TaskType.TRANSFER_S2B_S3,
            TaskType.TRANSFER_S3_RACK,
        }
    }

    assert "ZONE_S1_ARM1" in transfers[TaskType.TRANSFER_S1_S2A]
    assert "ZONE_S2A_ARM2" in (transfers[TaskType.TRANSFER_S1_S2A] & transfers[TaskType.TRANSFER_S2A_S2B])
    assert "ZONE_S2B_ARM3" in (transfers[TaskType.TRANSFER_S2A_S2B] & transfers[TaskType.TRANSFER_S2B_S3])
    assert "ZONE_S3_SHARED" in (transfers[TaskType.TRANSFER_S2B_S3] & transfers[TaskType.TRANSFER_S3_RACK])


def test_s3_and_furnace_junction_pallet_envelopes_have_real_clearance() -> None:
    centre_distance = SHALLOW_U_LAYOUT.rack_infeed_xy[0] - SHALLOW_U_LAYOUT.station_s3_xy[0]
    edge_clearance = centre_distance - 2.0 * SHALLOW_U_LAYOUT.output_pallet_half_width_m

    assert edge_clearance >= 0.040


def test_two_adjacent_orders_do_not_switch_to_gripper_before_both_bases_are_placed() -> None:
    application = BrazingApplication(
        parse_args(
            [
                "--headless",
                "--dt",
                "0.02",
                "--no-ui",
                "--no-terminal-commands",
                "--port",
                "0",
            ]
        )
    )
    gripper_before_all_bases = False
    tool_change_joint_samples: list[np.ndarray] = []
    try:
        for index, preset in enumerate(("A", "B"), start=1):
            application.process_command(
                {
                    "type": "order_insert",
                    "mode": "preset",
                    "preset": preset,
                    "order_id": f"GROUPED_TOOL_{index}",
                    "quantity": 1,
                    "priority": 10,
                    "route_strategy": "STANDARD",
                    "urgent": False,
                }
            )

        places = []
        for _ in range(16_000):
            application.tick()
            places = [
                task
                for task in application.manufacturing_runtime.graph
                if task.task_type is TaskType.PLACE_BASE_PLATE
            ]
            all_bases_placed = len(places) == 2 and all(
                task.status is TaskStatus.SUCCEEDED for task in places
            )
            current_tool = application.scene.arm1_tools.current_tool
            gripper_before_all_bases |= current_tool == "parallel_gripper" and not all_bases_placed
            preparing = any(
                task.task_type is TaskType.PREPARE_FIN_TOOL and task.status is TaskStatus.RUNNING
                for task in application.manufacturing_runtime.graph
            )
            if preparing:
                controller = application.scene.arms["arm1"]
                tool_change_joint_samples.append(
                    np.asarray(application.scene.data.qpos[controller.qpos_ids], dtype=float).copy()
                )
            if all_bases_placed and current_tool == "parallel_gripper":
                break
        else:
            raise AssertionError("Arm1 did not finish grouped base loading and gripper preparation")

        assert not gripper_before_all_bases
        assert len(places) == 2
        assert all(task.status is TaskStatus.SUCCEEDED for task in places)
        assert len(tool_change_joint_samples) > 5
        joint_span = np.ptp(np.stack(tool_change_joint_samples), axis=0)
        # A direct rack approach may reconfigure the elbow, but no individual
        # joint may execute the near-2pi branch wrap seen in the old motion.
        assert float(np.max(joint_span)) < math.pi
    finally:
        application.close()


def test_b_second_fin_is_committed_at_its_own_slot() -> None:
    application = BrazingApplication(
        parse_args(
            [
                "--headless",
                "--dt",
                "0.02",
                "--no-ui",
                "--no-terminal-commands",
                "--port",
                "0",
            ]
        )
    )
    try:
        application.process_command(
            {
                "type": "order_insert",
                "mode": "preset",
                "preset": "B",
                "order_id": "B_FIN_02_REGRESSION",
                "quantity": 1,
                "priority": 10,
                "route_strategy": "STANDARD",
                "urgent": False,
            }
        )
        second = None
        for _ in range(12_000):
            application.tick()
            second = next(
                (
                    task
                    for task in application.manufacturing_runtime.graph
                    if task.task_type is TaskType.INSTALL_FIN and task.payload.get("fin_id") == "fin_02"
                ),
                None,
            )
            if second is not None and second.status in {
                TaskStatus.SUCCEEDED,
                TaskStatus.FAILED,
            }:
                break

        assert second is not None
        assert second.status is TaskStatus.SUCCEEDED
        geom = application.scene.model.geom("batch_tray_01_fin_02")
        assert float(geom.rgba[3]) == pytest.approx(1.0)
        assert float(geom.pos[1]) == pytest.approx(-0.015)
        assert application.manufacturing_runtime.recovery.plans == {}
    finally:
        application.close()


def test_a_fourth_fin_selects_an_insertion_capable_wrist_branch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The penultimate A fin must not discover a bad q7 branch above the comb."""

    monkeypatch.setattr(
        "brazing_line.start_http_server",
        lambda *_args, **_kwargs: _NoopHttpServer(),
    )
    application = BrazingApplication(
        parse_args(
            [
                "--headless",
                "--dt",
                "0.002",
                "--no-ui",
                "--no-terminal-commands",
                "--port",
                "0",
            ]
        )
    )
    try:
        application.process_command(
            {
                "type": "order_insert",
                "mode": "preset",
                "preset": "A",
                "order_id": "A_FIN_04_BRANCH_REGRESSION",
                "quantity": 1,
                "priority": 10,
                "route_strategy": "STANDARD",
                "urgent": False,
            }
        )
        fourth = None
        for _ in range(85_000):
            application.tick()
            fourth = next(
                (
                    task
                    for task in application.manufacturing_runtime.graph
                    if task.task_type is TaskType.INSTALL_FIN and task.payload.get("fin_id") == "fin_04"
                ),
                None,
            )
            if fourth is not None and fourth.status in {
                TaskStatus.SUCCEEDED,
                TaskStatus.FAILED,
            }:
                break

        assert fourth is not None
        assert fourth.status is TaskStatus.SUCCEEDED, fourth.failure_reason
        assert float(application.scene.model.geom("batch_tray_01_fin_04").rgba[3]) == pytest.approx(1.0)
        assert not [
            fault
            for fault in application.manufacturing_runtime.faults.values()
            if fault.related_task_id == fourth.task_id
        ]
    finally:
        application.close()


def test_c_all_seven_fins_use_progressive_grip_and_are_committed() -> None:
    application = BrazingApplication(
        parse_args(
            [
                "--headless",
                "--dt",
                "0.02",
                "--no-ui",
                "--no-terminal-commands",
                "--port",
                "0",
            ]
        )
    )
    samples: list[float] = []
    maximum_carry_path_joint_step_rad = 0.0
    try:
        application.process_command(
            {
                "type": "order_insert",
                "mode": "preset",
                "preset": "C",
                "order_id": "C_FIN_01_REGRESSION",
                "quantity": 1,
                "priority": 10,
                "route_strategy": "STANDARD",
                "urgent": False,
            }
        )
        first = None
        installs = []
        for _ in range(24_000):
            application.tick()
            for execution in application.manufacturing_runtime.executor.active.values():
                if execution.task.task_type is not TaskType.INSTALL_FIN:
                    continue
                skill = execution.skill
                if not skill.arm_stages or len(skill.arm_stages) < 2:
                    continue
                carry_stage = skill.arm_stages[1]
                if carry_stage.joint_path is None:
                    continue
                maximum_carry_path_joint_step_rad = max(
                    maximum_carry_path_joint_step_rad,
                    float(np.max(np.abs(np.diff(carry_stage.joint_path, axis=0)))),
                )
            running_first = any(
                task.task_type in {TaskType.PICK_FIN, TaskType.INSTALL_FIN}
                and task.payload.get("fin_id") == "fin_01"
                and task.status is TaskStatus.RUNNING
                for task in application.manufacturing_runtime.graph
            )
            if running_first:
                samples.append(application.scene.registry.arm1_gripper_closed_fraction())
            first = next(
                (
                    task
                    for task in application.manufacturing_runtime.graph
                    if task.task_type is TaskType.INSTALL_FIN and task.payload.get("fin_id") == "fin_01"
                ),
                None,
            )
            installs = [
                task
                for task in application.manufacturing_runtime.graph
                if task.task_type is TaskType.INSTALL_FIN
            ]
            if len(installs) == 7 and all(
                task.status in {TaskStatus.SUCCEEDED, TaskStatus.FAILED} for task in installs
            ):
                break

        assert first is not None
        assert first.status is TaskStatus.SUCCEEDED, [
            fault.details for fault in application.manufacturing_runtime.faults.values()
        ]
        assert len(installs) == 7
        assert all(task.status is TaskStatus.SUCCEEDED for task in installs)
        for index in range(1, 8):
            assert float(
                application.scene.model.geom(f"batch_tray_01_fin_{index:02d}").rgba[3]
            ) == pytest.approx(1.0)
        assert any(0.10 < value < 0.85 for value in samples)
        assert samples[-1] == pytest.approx(0.90, abs=0.03)
        assert maximum_carry_path_joint_step_rad < 0.20
        assert application.manufacturing_runtime.recovery.plans == {}
    finally:
        application.close()
