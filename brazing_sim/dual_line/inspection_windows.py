"""Deterministic future reservations for Arm3 inspection work."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Iterable


@dataclass(frozen=True, slots=True)
class InspectionWindowRequest:
    unit_id: str
    inspection_kind: str
    source_stage: str
    ready_at: float
    duration_s: float
    reason_zh: str


@dataclass(frozen=True, slots=True)
class Arm3InspectionWindow:
    unit_id: str
    inspection_kind: str
    source_stage: str
    start_at: float
    end_at: float
    reason_zh: str

    def as_dict(self) -> dict[str, object]:
        return {
            "unit_id": self.unit_id,
            "inspection_kind": self.inspection_kind,
            "source_stage": self.source_stage,
            "start_at": round(self.start_at, 6),
            "end_at": round(self.end_at, 6),
            "reason_zh": self.reason_zh,
        }


def schedule_arm3_inspection_windows(
    requests: Iterable[InspectionWindowRequest],
    *,
    arm3_available_at: float,
) -> tuple[Arm3InspectionWindow, ...]:
    """Serialize predicted arrivals on Arm3 without changing tray ownership."""

    cursor = float(arm3_available_at)
    if not isfinite(cursor):
        return ()
    windows: list[Arm3InspectionWindow] = []
    for request in sorted(requests, key=lambda item: (item.ready_at, item.unit_id)):
        if not isfinite(request.ready_at) or not isfinite(request.duration_s) or request.duration_s <= 0.0:
            continue
        start_at = max(cursor, request.ready_at)
        end_at = start_at + request.duration_s
        windows.append(
            Arm3InspectionWindow(
                unit_id=request.unit_id,
                inspection_kind=request.inspection_kind,
                source_stage=request.source_stage,
                start_at=start_at,
                end_at=end_at,
                reason_zh=request.reason_zh,
            )
        )
        cursor = end_at
    return tuple(windows)


__all__ = [
    "Arm3InspectionWindow",
    "InspectionWindowRequest",
    "schedule_arm3_inspection_windows",
]
