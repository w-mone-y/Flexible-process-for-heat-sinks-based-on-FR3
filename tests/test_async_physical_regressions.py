"""Focused regressions for visible asynchronous line motion."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from brazing_line import BrazingApplication, parse_args
from brazing_sim.execution.async_line_skills import AsyncLinePhysicalSkill, _TransferState
from brazing_sim.flexible import build_inline_plan
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
