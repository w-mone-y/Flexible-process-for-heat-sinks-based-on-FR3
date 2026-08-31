from __future__ import annotations

import pytest

from brazing_sim.twin_duration import ShadowDurationEstimator


def test_shadow_duration_estimator_updates_only_from_completed_samples() -> None:
    estimator = ShadowDurationEstimator(prior_seconds=10.0)

    estimator.observe_started("TASK-1", 2.0)
    assert estimator.sample_count("INSTALL_FIN", "ARM1") == 0
    assert estimator.observe_completed("TASK-1", task_type="INSTALL_FIN", resource_id="ARM1", finished_at=8.0)

    estimate = estimator.predict("INSTALL_FIN", "ARM1")
    assert estimate.mean_s == pytest.approx(6.0)
    assert estimate.sample_count == 1
    assert estimate.confidence == "LOW"


def test_shadow_duration_estimator_uses_ewma_and_is_serialisable() -> None:
    estimator = ShadowDurationEstimator(prior_seconds=10.0, alpha=0.5)
    for index, duration in enumerate((6.0, 8.0), start=1):
        task_id = f"TASK-{index}"
        estimator.observe_started(task_id, float(index))
        estimator.observe_completed(
            task_id,
            task_type="INSTALL_FIN",
            resource_id="ARM1",
            finished_at=float(index) + duration,
        )

    estimate = estimator.predict("INSTALL_FIN", "ARM1")
    assert estimate.mean_s == pytest.approx(7.0)
    assert estimate.sample_count == 2
    assert estimator.snapshot()["INSTALL_FIN|ARM1"]["mean_s"] == pytest.approx(7.0)


def test_unknown_completion_and_invalid_sample_are_ignored() -> None:
    estimator = ShadowDurationEstimator()
    assert not estimator.observe_completed(
        "UNKNOWN", task_type="INSTALL_FIN", resource_id="ARM1", finished_at=1.0
    )
    estimator.observe_started("TASK", 4.0)
    assert not estimator.observe_completed(
        "TASK", task_type="INSTALL_FIN", resource_id="ARM1", finished_at=3.0
    )
    assert estimator.sample_count("INSTALL_FIN", "ARM1") == 0
