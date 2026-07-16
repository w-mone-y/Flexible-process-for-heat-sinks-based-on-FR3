"""KPI collection for the brazing-line coordinator.

The collector deliberately has no MuJoCo or Qt dependency.  Simulation code can
feed it either wall-clock or fake-clock timestamps, which keeps headless tests
deterministic and makes the furnace cycle non-blocking.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict, dataclass, is_dataclass
from enum import Enum
import math
import time
from typing import Any, Callable, Mapping


def _name(value: Any) -> str:
    if isinstance(value, Enum):
        return str(value.value)
    return str(value)


def _json_value(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if is_dataclass(value):
        return {key: _json_value(item) for key, item in asdict(value).items()}
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_value(item) for item in value]
    return value


@dataclass(frozen=True)
class KpiView:
    """Stable fallback snapshot used when the domain snapshot is unavailable."""

    order_elapsed: float
    phase_durations: dict[str, float]
    actor_busy: dict[str, float]
    actor_waiting: dict[str, float]
    resource_waits: int
    resource_conflicts: int
    rework_counts: dict[str, int]
    path_rmse_mm: float
    path_max_error_mm: float
    final_quality_score: float | None


class KpiTracker:
    """Accumulate order, phase, actor, resource, rework and path metrics."""

    def __init__(self, clock: Callable[[], float] = time.monotonic) -> None:
        self.clock = clock
        self.reset()

    def reset(self) -> None:
        self.order_started_at: float | None = None
        self.order_finished_at: float | None = None
        self._stage: str | None = None
        self._stage_started_at: float | None = None
        self.phase_durations: dict[str, float] = defaultdict(float)
        self.actor_busy: dict[str, float] = defaultdict(float)
        self.actor_waiting: dict[str, float] = defaultdict(float)
        self._actor_busy_since: dict[str, float] = {}
        self._actor_wait_since: dict[str, float] = {}
        self.resource_waits = 0
        self.resource_conflicts = 0
        self.rework_counts: dict[str, int] = defaultdict(int)
        self._path_squared_error_m2: list[float] = []
        self._path_max_error_m = 0.0
        self.final_quality_score: float | None = None

    def start_order(self, now: float | None = None) -> None:
        self.reset()
        self.order_started_at = self.clock() if now is None else float(now)

    def enter_stage(self, stage: Any, now: float | None = None) -> None:
        timestamp = self.clock() if now is None else float(now)
        self._close_stage(timestamp)
        self._stage = _name(stage)
        self._stage_started_at = timestamp

    def _close_stage(self, now: float) -> None:
        if self._stage is not None and self._stage_started_at is not None:
            self.phase_durations[self._stage] += max(0.0, now - self._stage_started_at)
        self._stage = None
        self._stage_started_at = None

    def actor_started(self, actor: Any, now: float | None = None) -> None:
        timestamp = self.clock() if now is None else float(now)
        key = _name(actor)
        self.actor_stopped_waiting(key, timestamp)
        self._actor_busy_since.setdefault(key, timestamp)

    def actor_finished(self, actor: Any, now: float | None = None) -> None:
        timestamp = self.clock() if now is None else float(now)
        key = _name(actor)
        started = self._actor_busy_since.pop(key, None)
        if started is not None:
            self.actor_busy[key] += max(0.0, timestamp - started)

    def actor_waiting_for_resource(self, actor: Any, now: float | None = None) -> None:
        timestamp = self.clock() if now is None else float(now)
        key = _name(actor)
        if key not in self._actor_wait_since:
            self.resource_waits += 1
            self._actor_wait_since[key] = timestamp

    def actor_stopped_waiting(self, actor: Any, now: float | None = None) -> None:
        timestamp = self.clock() if now is None else float(now)
        key = _name(actor)
        started = self._actor_wait_since.pop(key, None)
        if started is not None:
            self.actor_waiting[key] += max(0.0, timestamp - started)

    def resource_conflict(self) -> None:
        self.resource_conflicts += 1

    def record_rework(self, kind: Any, count: int = 1) -> None:
        self.rework_counts[_name(kind)] += int(count)

    def record_path_error(self, error_m: float) -> None:
        error = abs(float(error_m))
        if math.isfinite(error):
            self._path_squared_error_m2.append(error * error)
            self._path_max_error_m = max(self._path_max_error_m, error)

    def set_final_quality(self, score: float | None) -> None:
        self.final_quality_score = None if score is None else float(score)

    def finish_order(self, now: float | None = None, quality_score: float | None = None) -> None:
        timestamp = self.clock() if now is None else float(now)
        self._close_stage(timestamp)
        for actor in list(self._actor_busy_since):
            self.actor_finished(actor, timestamp)
        for actor in list(self._actor_wait_since):
            self.actor_stopped_waiting(actor, timestamp)
        self.order_finished_at = timestamp
        if quality_score is not None:
            self.set_final_quality(quality_score)

    def as_dict(self, now: float | None = None) -> dict[str, Any]:
        timestamp = self.clock() if now is None else float(now)
        phase = dict(self.phase_durations)
        if self._stage is not None and self._stage_started_at is not None:
            phase[self._stage] = phase.get(self._stage, 0.0) + max(0.0, timestamp - self._stage_started_at)
        busy = dict(self.actor_busy)
        for actor, started in self._actor_busy_since.items():
            busy[actor] = busy.get(actor, 0.0) + max(0.0, timestamp - started)
        waiting = dict(self.actor_waiting)
        for actor, started in self._actor_wait_since.items():
            waiting[actor] = waiting.get(actor, 0.0) + max(0.0, timestamp - started)
        end = self.order_finished_at if self.order_finished_at is not None else timestamp
        elapsed = 0.0 if self.order_started_at is None else max(0.0, end - self.order_started_at)
        rmse = (
            math.sqrt(sum(self._path_squared_error_m2) / len(self._path_squared_error_m2)) * 1000.0
            if self._path_squared_error_m2
            else 0.0
        )
        view = KpiView(
            order_elapsed=elapsed,
            phase_durations={key: round(value, 6) for key, value in phase.items()},
            actor_busy={key: round(value, 6) for key, value in busy.items()},
            actor_waiting={key: round(value, 6) for key, value in waiting.items()},
            resource_waits=self.resource_waits,
            resource_conflicts=self.resource_conflicts,
            rework_counts=dict(self.rework_counts),
            path_rmse_mm=rmse,
            path_max_error_mm=self._path_max_error_m * 1000.0,
            final_quality_score=self.final_quality_score,
        )
        return _json_value(view)

    def snapshot(self, now: float | None = None) -> Any:
        """Return the domain ``KpiSnapshot`` when compatible, else ``KpiView``.

        Domain types evolved during the refactor.  Supporting both the typed
        representation and a stable fallback lets the UI/API import early
        without coupling this module to one constructor signature.
        """

        values = self.as_dict(now)
        try:
            from .domain import KpiSnapshot  # type: ignore

            annotations = getattr(KpiSnapshot, "__annotations__", {})
            kwargs = {key: value for key, value in values.items() if not annotations or key in annotations}
            return KpiSnapshot(**kwargs)
        except (ImportError, TypeError, ValueError):
            return KpiView(**values)


__all__ = ["KpiTracker", "KpiView"]
