"""Simulation-time task timeout and no-progress monitoring."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(slots=True)
class ExecutionWatch:
    task_id: str
    started_at: float
    timeout: float
    last_progress: float = 0.0
    last_progress_at: float = 0.0
    failed_once: bool = False


class ExecutionMonitor:
    def __init__(self, *, no_progress_timeout: float = 10.0) -> None:
        self.no_progress_timeout = float(no_progress_timeout)
        self.active: dict[str, ExecutionWatch] = {}

    def start(self, task_id: str, now: float, timeout: float) -> None:
        self.active[task_id] = ExecutionWatch(task_id, float(now), float(timeout), 0.0, float(now))

    def update(self, task_id: str, now: float, progress: float | None = None) -> str | None:
        watch = self.active[task_id]
        if progress is not None and float(progress) > watch.last_progress + 1e-9:
            watch.last_progress = float(progress)
            watch.last_progress_at = float(now)
        reason = None
        if float(now) - watch.started_at > watch.timeout:
            reason = "TASK_TIMEOUT"
        elif float(now) - watch.last_progress_at > self.no_progress_timeout:
            reason = "NO_PROGRESS_TIMEOUT"
        if reason and not watch.failed_once:
            watch.failed_once = True
            return reason
        return None

    def finish(self, task_id: str) -> ExecutionWatch | None:
        return self.active.pop(task_id, None)

    def reset(self) -> None:
        self.active.clear()

    def snapshot(self) -> dict[str, dict[str, Any]]:
        return {
            task_id: {
                "started_at": watch.started_at,
                "timeout": watch.timeout,
                "last_progress": watch.last_progress,
                "last_progress_at": watch.last_progress_at,
                "failed_once": watch.failed_once,
            }
            for task_id, watch in self.active.items()
        }


__all__ = ["ExecutionMonitor", "ExecutionWatch"]
