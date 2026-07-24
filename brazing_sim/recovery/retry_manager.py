"""Simulation-time retry delay bookkeeping."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class RetryState:
    task_id: str
    attempt: int
    retry_at: float


class RetryManager:
    def __init__(self, *, default_delay: float = 0.5) -> None:
        self.default_delay = float(default_delay)
        self.waiting: dict[str, RetryState] = {}

    def schedule(self, task_id: str, attempt: int, now: float, delay: float | None = None) -> RetryState:
        state = RetryState(
            str(task_id),
            int(attempt),
            float(now) + (self.default_delay if delay is None else float(delay)),
        )
        self.waiting[state.task_id] = state
        return state

    def due(self, now: float) -> tuple[RetryState, ...]:
        result = tuple(state for state in self.waiting.values() if state.retry_at <= float(now))
        for state in result:
            self.waiting.pop(state.task_id, None)
        return result

    def cancel(self, task_id: str) -> bool:
        return self.waiting.pop(task_id, None) is not None

    def reset(self) -> None:
        self.waiting.clear()


__all__ = ["RetryManager", "RetryState"]
