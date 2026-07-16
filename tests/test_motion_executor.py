from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from brazing_sim.motion import (
    ExecutionState,
    PolylineTrajectory,
    Pose,
    ReachabilityResult,
    TrajectoryExecutor,
)


class TrackingController:
    """Minimal controller double with a configurable steady TCP offset."""

    def __init__(self, offset_m: float) -> None:
        self.offset_m = float(offset_m)
        self.target = Pose(np.zeros(3))
        self.config = SimpleNamespace(
            position_tolerance_m=0.003,
            orientation_tolerance_rad=np.deg2rad(3.0),
            path_rmse_limit_m=0.003,
            path_max_error_limit_m=0.005,
        )

    def validate_trajectory(self, trajectory: PolylineTrajectory):
        return tuple(ReachabilityResult(True, 0.0, 0.0, np.zeros(7), 1) for _ in trajectory.samples())

    def set_target(self, pose: Pose, *, tcp: bool = False) -> None:
        assert tcp
        self.target = pose

    def stop(self, reason: str = "") -> None:
        self.stop_reason = reason

    def current_tcp_pose(self) -> Pose:
        position = self.target.position.copy()
        position[0] -= self.offset_m
        return Pose(position, self.target.quaternion)

    @property
    def at_target(self) -> bool:
        return self.offset_m <= self.config.position_tolerance_m


def test_executor_advances_through_2_7_mm_steady_tracking_error() -> None:
    controller = TrackingController(0.0027)
    executor = TrajectoryExecutor(controller)  # type: ignore[arg-type]
    trajectory = PolylineTrajectory(
        (Pose(np.zeros(3)), Pose(np.asarray([0.020, 0.0, 0.0]))),
        speed_m_s=0.010,
    )
    executor.start(trajectory, 0.0, timeout_s=5.0)

    state = ExecutionState.RUNNING
    for index in range(1, 301):
        state = executor.tick(index * 0.01)
        if state is not ExecutionState.RUNNING:
            break

    assert state is ExecutionState.COMPLETE
    assert np.isclose(executor.command_distance_m, trajectory.length_m)


def test_executor_reports_one_stall_event_after_ten_seconds() -> None:
    controller = TrackingController(0.004)
    executor = TrajectoryExecutor(controller)  # type: ignore[arg-type]
    trajectory = PolylineTrajectory(
        (Pose(np.zeros(3)), Pose(np.asarray([0.020, 0.0, 0.0]))),
        speed_m_s=0.010,
    )
    executor.start(trajectory, 0.0, timeout_s=30.0)

    assert executor.tick(10.01) is ExecutionState.ERROR
    assert "tracking stalled" in executor.error
    assert [event.kind for event in executor.events].count("failed") == 1
    executor.tick(20.0)
    assert [event.kind for event in executor.events].count("failed") == 1
