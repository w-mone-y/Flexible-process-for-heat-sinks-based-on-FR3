from __future__ import annotations

from types import SimpleNamespace

import pytest

from brazing_line import BrazingApplication, SimulationRate, ViewerRenderScheduler


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
