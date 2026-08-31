from __future__ import annotations

import numpy as np

from brazing_sim.planning.motion_planner import TimedJointSample
from brazing_sim.safety import GeometrySafetyBarrier


def _samples() -> list[TimedJointSample]:
    return [
        TimedJointSample(0.0, (0.0, 0.0), safe_wait=True),
        TimedJointSample(0.05, (1.0, 0.0)),
        TimedJointSample(0.10, (2.0, 0.0)),
    ]


def test_force_barrier_checks_every_time_sample_and_records_clearance() -> None:
    barrier = GeometrySafetyBarrier(mode="FORCE", minimum_clearance_m=0.04)

    report = barrier.evaluate(
        _samples(),
        clearance=lambda q: 0.05 if np.asarray(q)[0] < 1.5 else 0.02,
    )

    assert not report.allowed
    assert report.reason_code == "CLEARANCE_BELOW_THRESHOLD"
    assert report.sample_count >= 3
    assert report.minimum_clearance_m == 0.02
    assert barrier.snapshot()["blocked_count"] == 1


def test_shadow_barrier_reports_without_blocking_and_requires_certified_wait() -> None:
    barrier = GeometrySafetyBarrier(mode="SHADOW", minimum_clearance_m=0.04)

    report = barrier.evaluate(_samples(), clearance=lambda _q: 0.01)

    assert report.allowed
    assert report.shadow_violation
    assert barrier.can_wait(np.asarray((0.0, 0.0)), node_name="CURRENT_CERTIFIED_WAIT")
    assert not barrier.can_wait(np.asarray((1.0, 0.0)))


def test_barrier_reset_clears_incidents() -> None:
    barrier = GeometrySafetyBarrier(mode="FORCE")
    barrier.evaluate(_samples(), clearance=lambda _q: 0.01)
    assert barrier.snapshot()["checked_count"] == 1
    barrier.reset()
    assert barrier.snapshot()["checked_count"] == 0
