from __future__ import annotations

import pytest

from brazing_sim.dual_line.windows_viewer import CameraController


def test_camera_zoom_positive_is_in_and_stays_clamped() -> None:
    camera = CameraController()
    initial = camera.distance

    camera.zoom(1.0)
    assert camera.distance < initial

    camera.zoom(-100.0)
    assert camera.distance == pytest.approx(camera.MAX_DISTANCE)

    camera.zoom(100.0)
    assert camera.distance == pytest.approx(camera.MIN_DISTANCE)


def test_camera_pan_moves_on_the_ground_plane() -> None:
    camera = CameraController()
    initial = tuple(camera.lookat)

    camera.pan(40.0, -20.0)

    assert tuple(camera.lookat[:2]) != initial[:2]
    assert camera.lookat[2] == pytest.approx(initial[2])


def test_camera_orbit_clamps_elevation() -> None:
    camera = CameraController()

    camera.orbit(10.0, -10_000.0)
    assert camera.elevation == pytest.approx(camera.MAX_ELEVATION)

    camera.orbit(0.0, 10_000.0)
    assert camera.elevation == pytest.approx(camera.MIN_ELEVATION)
