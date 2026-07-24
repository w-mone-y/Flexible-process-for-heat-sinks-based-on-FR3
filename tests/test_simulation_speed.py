from __future__ import annotations

from types import SimpleNamespace

import pytest

from brazing_line import (
    BrazingApplication,
    RealTimeFactorTracker,
    SimulationRate,
    ViewerFramePacer,
    ViewerRenderScheduler,
)
from brazing_sim.ui import (
    TASK_GRAPH_NODE_SIZE,
    _paint_task_graph_node_text,
    _place_task_graph_node,
)


def test_simulation_rate_doubles_and_halves_current_process_speed() -> None:
    rate = SimulationRate()
    assert rate.multiplier == pytest.approx(1.0)
    assert rate.steps_for_frame() == 1

    assert rate.adjust("accelerate") == pytest.approx(2.0)
    assert rate.steps_for_frame() == 2

    assert rate.adjust("decelerate") == pytest.approx(1.0)
    assert rate.adjust("decelerate") == pytest.approx(0.5)
    assert [rate.steps_for_frame() for _ in range(4)] == [0, 1, 0, 1]


def test_simulation_rate_has_safe_quarter_to_thirty_two_times_bounds() -> None:
    rate = SimulationRate()
    for _ in range(8):
        rate.adjust("accelerate")
    assert rate.multiplier == pytest.approx(32.0)
    assert rate.steps_for_frame() == 32

    for _ in range(12):
        rate.adjust("decelerate")
    assert rate.multiplier == pytest.approx(0.25)

    with pytest.raises(ValueError, match="speed action"):
        rate.adjust("invalid")


def test_application_executes_thirty_two_real_ticks_per_frame_at_maximum_speed() -> None:
    application = object.__new__(BrazingApplication)
    application.simulation_rate = SimulationRate()
    tick_count = 0

    def tick() -> None:
        nonlocal tick_count
        tick_count += 1

    application.tick = tick
    for _ in range(5):
        application.simulation_rate.adjust("accelerate")

    assert application.simulation_rate.multiplier == pytest.approx(32.0)
    assert application.advance_simulation_frame() == 32
    assert tick_count == 32


def test_high_speed_frame_yields_after_small_physics_chunks() -> None:
    application = object.__new__(BrazingApplication)
    application.simulation_rate = SimulationRate()
    application.tick = lambda: None
    for _ in range(5):
        application.simulation_rate.adjust("accelerate")
    chunks: list[tuple[int, int]] = []

    application.advance_simulation_frame(
        lambda completed, total: chunks.append((completed, total)),
        max_chunk_steps=4,
    )

    assert chunks == [(4, 32), (8, 32), (12, 32), (16, 32), (20, 32), (24, 32), (28, 32), (32, 32)]


def test_high_speed_viewer_uses_one_large_physics_chunk() -> None:
    application = object.__new__(BrazingApplication)
    application.simulation_rate = SimulationRate()
    for _ in range(5):
        application.simulation_rate.adjust("accelerate")
    assert application.viewer_step_chunk() == 32
    assert application.viewer_sync_fps() == pytest.approx(10.0)
    assert application.state_publish_interval() == pytest.approx(0.16)


def test_viewer_redraw_is_independent_of_physics_chunks() -> None:
    pacer = ViewerFramePacer(30.0)
    assert pacer.due(0.0)
    pacer.mark_synced(0.0)
    assert not pacer.due(0.010)
    assert not pacer.due(0.030)
    assert pacer.due(0.034)


def test_v2_scheduler_is_rate_limited_independently_of_physics() -> None:
    application = object.__new__(BrazingApplication)
    application._last_v2_tick_sim = float("-inf")
    application.scene = SimpleNamespace(time=0.0)
    tick_times: list[float] = []
    application.manufacturing_runtime = SimpleNamespace(tick=lambda now: tick_times.append(now))

    for index in range(21):
        application.scene.time = index * 0.002
        application._tick_v2_runtime_if_due()

    assert tick_times == pytest.approx([0.0, 0.02, 0.04])


def test_manual_physical_fault_routes_without_typed_identifiers() -> None:
    application = object.__new__(BrazingApplication)
    recorded_physical: list[tuple[str, str, str]] = []
    recorded_runtime: list[dict[str, object]] = []
    application.coordinator = SimpleNamespace(
        inject_fault=lambda kind, target, severity: recorded_physical.append((kind, target, severity))
        or SimpleNamespace(fault_type=kind, target=target, severity=severity)
    )
    application.manufacturing_runtime = SimpleNamespace(
        arm_manual_fault=lambda kind, **kwargs: recorded_runtime.append({"kind": kind, **kwargs})
        or SimpleNamespace(request_id="MANUAL_0001")
    )
    application.scene = SimpleNamespace(time=12.5)
    application._physical_fault_sequence = 0
    application._physical_fault_holds = []

    application.inject_manual_fault(
        {
            "fault_type": "BRAZING_MISSING",
            "target": "slot_02_left",
            "severity": "recoverable",
            "auto_recover": True,
            "duration_s": 8.0,
        }
    )

    assert recorded_physical == [("brazing_gap", "slot_02_left", "recoverable")]
    assert recorded_runtime[0]["kind"] == "BRAZING_MISSING"
    assert recorded_runtime[0]["details"]["path_ids"] == ["slot_02_left"]


def test_actual_rtf_uses_simulation_time_over_wall_time() -> None:
    tracker = RealTimeFactorTracker(sample_window_s=0.25, smoothing=1.0)
    assert tracker.observe(0.0, 10.0) == pytest.approx(0.0)
    assert tracker.observe(0.5, 10.25) == pytest.approx(2.0)
    assert tracker.observe(0.75, 10.5) == pytest.approx(1.0)


def test_task_graph_node_uses_local_geometry_and_scene_position() -> None:
    class GraphicsItem:
        def __init__(self) -> None:
            self.position: tuple[float, float] | None = None
            self.z = 0.0

        def setPos(self, x: float, y: float) -> None:  # noqa: N802
            self.position = (x, y)

        def setZValue(self, value: float) -> None:  # noqa: N802
            self.z = value

    node = GraphicsItem()
    _place_task_graph_node(node, 380.0, 82.0)
    assert TASK_GRAPH_NODE_SIZE == (190.0, 72.0)
    assert node.position == (380.0, 82.0)


def test_task_graph_custom_painter_emits_visible_text(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    pytest.importorskip("PySide6")
    from PySide6.QtCore import QRectF
    from PySide6.QtGui import QColor, QImage, QPainter
    from PySide6.QtWidgets import QApplication

    application = QApplication.instance() or QApplication([])
    assert application is not None
    image = QImage(190, 72, QImage.Format_ARGB32)
    image.fill(QColor("#238636"))
    painter = QPainter(image)
    _paint_task_graph_node_text(
        painter,
        QRectF(0.0, 0.0, 190.0, 72.0),
        "吸取基板",
        "工件1",
        "正在执行",
    )
    painter.end()

    bright_pixels = 0
    for y in range(image.height()):
        for x in range(image.width()):
            color = image.pixelColor(x, y)
            bright_pixels += int(color.red() > 200 and color.green() > 200 and color.blue() > 200)
    assert bright_pixels > 40


def test_secondary_camera_yields_while_user_moves_main_view() -> None:
    scheduler = ViewerRenderScheduler(
        5.0,
        standby_fps=2.0,
        interaction_cooldown_s=0.35,
    )
    camera = SimpleNamespace(
        azimuth=90.0,
        elevation=-32.0,
        distance=2.9,
        lookat=[-0.05, 0.62, 0.25],
    )

    assert not scheduler.observe_view(camera, 0.0)
    assert scheduler.camera_due(0.0, inspection_active=True)
    scheduler.mark_camera_rendered(0.0)
    assert not scheduler.camera_due(0.10, inspection_active=True)
    assert scheduler.camera_due(0.20, inspection_active=True)

    scheduler.mark_camera_rendered(0.20)
    camera.azimuth = 91.0
    assert scheduler.observe_view(camera, 0.30)
    assert not scheduler.camera_due(0.55, inspection_active=True)
    assert scheduler.camera_due(0.66, inspection_active=True)

    scheduler.mark_camera_rendered(0.66)
    assert not scheduler.camera_due(1.00, inspection_active=False)
    assert scheduler.camera_due(1.17, inspection_active=False)


def test_secondary_camera_scheduler_rejects_invalid_rates() -> None:
    with pytest.raises(ValueError, match="camera render rates"):
        ViewerRenderScheduler(0.0)


def test_secondary_camera_is_capped_at_high_simulation_speed() -> None:
    scheduler = ViewerRenderScheduler(5.0, standby_fps=2.0, interaction_cooldown_s=0.0)
    assert scheduler.camera_due(0.0, inspection_active=True, speed_multiplier=32.0)
    scheduler.mark_camera_rendered(0.0)
    assert not scheduler.camera_due(0.5, inspection_active=True, speed_multiplier=32.0)
    assert scheduler.camera_due(1.0, inspection_active=True, speed_multiplier=32.0)
