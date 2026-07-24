"""Lightweight synchronous event bus shared by runtime, UI and logging."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable


class EventType(str, Enum):
    ORDER_RELEASED = "ORDER_RELEASED"
    ORDER_COMPLETED = "ORDER_COMPLETED"
    TASK_READY = "TASK_READY"
    TASK_RESERVED = "TASK_RESERVED"
    TASK_STARTED = "TASK_STARTED"
    TASK_SUCCEEDED = "TASK_SUCCEEDED"
    TASK_FAILED = "TASK_FAILED"
    RESOURCE_FAULTED = "RESOURCE_FAULTED"
    RESOURCE_RECOVERED = "RESOURCE_RECOVERED"
    INSPECTION_FAILED = "INSPECTION_FAILED"
    RACK_LAYER_UNAVAILABLE = "RACK_LAYER_UNAVAILABLE"
    FURNACE_READY = "FURNACE_READY"
    RECOVERY_PLANNED = "RECOVERY_PLANNED"
    RECOVERY_COMPLETED = "RECOVERY_COMPLETED"
    REPLAN_STARTED = "REPLAN_STARTED"
    REPLAN_COMPLETED = "REPLAN_COMPLETED"
    EMERGENCY_STOP = "EMERGENCY_STOP"


@dataclass(frozen=True, slots=True)
class SystemEvent:
    event_type: EventType | str
    sim_time: float
    source: str
    payload: dict[str, Any] = field(default_factory=dict)
    sequence: int = 0

    def __post_init__(self) -> None:
        object.__setattr__(self, "event_type", EventType(self.event_type))

    def as_dict(self) -> dict[str, Any]:
        return {
            "sequence": self.sequence,
            "event_type": self.event_type.value,
            "sim_time": self.sim_time,
            "source": self.source,
            "payload": dict(self.payload),
        }


EventHandler = Callable[[SystemEvent], None]


class EventBus:
    def __init__(self, *, history_limit: int | None = None) -> None:
        self.history_limit = history_limit
        self._sequence = 0
        self._handlers: dict[EventType | None, list[EventHandler]] = defaultdict(list)
        self.history: list[SystemEvent] = []

    def subscribe(self, event_type: EventType | str | None, handler: EventHandler) -> None:
        key = None if event_type is None else EventType(event_type)
        if handler not in self._handlers[key]:
            self._handlers[key].append(handler)

    def unsubscribe(self, event_type: EventType | str | None, handler: EventHandler) -> None:
        key = None if event_type is None else EventType(event_type)
        if handler in self._handlers[key]:
            self._handlers[key].remove(handler)

    def publish(
        self,
        event: SystemEvent | EventType | str,
        *,
        sim_time: float = 0.0,
        source: str = "runtime",
        payload: dict[str, Any] | None = None,
    ) -> SystemEvent:
        self._sequence += 1
        if isinstance(event, SystemEvent):
            emitted = SystemEvent(
                event.event_type,
                event.sim_time,
                event.source,
                dict(event.payload),
                self._sequence,
            )
        else:
            emitted = SystemEvent(event, float(sim_time), str(source), dict(payload or {}), self._sequence)
        self.history.append(emitted)
        if self.history_limit is not None and len(self.history) > self.history_limit:
            del self.history[: len(self.history) - self.history_limit]
        for handler in [*self._handlers[emitted.event_type], *self._handlers[None]]:
            handler(emitted)
        return emitted

    def since(self, sequence: int) -> list[SystemEvent]:
        return [event for event in self.history if event.sequence > sequence]

    def reset(self) -> None:
        self.history.clear()
        self._sequence = 0


__all__ = ["EventBus", "EventType", "SystemEvent"]
