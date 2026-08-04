"""Deterministic single-Arm3-camera coordination for V2."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any


class CameraReviewReason(str, Enum):
    HIGH_RELIABILITY = "HIGH_RELIABILITY"
    FIRST_ARTICLE = "FIRST_ARTICLE"


_REASON_PRIORITY = {
    CameraReviewReason.HIGH_RELIABILITY: 0,
    CameraReviewReason.FIRST_ARTICLE: 0,
}

_KIND_PRIORITY = {
    "PRE_BRAZE_INSPECTION": 0,
    "MATERIAL_INSPECTION": 1,
}


@dataclass(slots=True)
class CameraReviewRequest:
    unit_id: str
    inspection_kind: str
    station_id: str
    reason: CameraReviewReason
    requested_at: float
    status: str = "QUEUED"
    started_at: float | None = None

    @property
    def key(self) -> tuple[str, str]:
        return self.unit_id, self.inspection_kind

    def as_dict(self) -> dict[str, Any]:
        return {
            "unit_id": self.unit_id,
            "inspection_kind": self.inspection_kind,
            "station_id": self.station_id,
            "reason": self.reason.value,
            "status": self.status,
            "requested_at": self.requested_at,
            "started_at": self.started_at,
        }


class CameraCoordinationPolicy:
    """Own idempotent Arm3 review requests and their audit history."""

    def __init__(self) -> None:
        self._pending: dict[tuple[str, str], CameraReviewRequest] = {}
        self._history: list[dict[str, Any]] = []
        self._route_reviews: set[tuple[str, str]] = set()

    def request(
        self,
        *,
        unit_id: str,
        inspection_kind: str,
        station_id: str,
        reason: CameraReviewReason,
        now: float,
    ) -> CameraReviewRequest:
        key = str(unit_id), str(inspection_kind)
        current = self._pending.get(key)
        if current is not None:
            return current
        request = CameraReviewRequest(
            unit_id=key[0],
            inspection_kind=key[1],
            station_id=str(station_id),
            reason=CameraReviewReason(reason),
            requested_at=float(now),
        )
        self._pending[key] = request
        return request

    def pending_for(self, unit_id: str, inspection_kind: str) -> CameraReviewRequest | None:
        return self._pending.get((str(unit_id), str(inspection_kind)))

    def pending_requests(self) -> tuple[CameraReviewRequest, ...]:
        return tuple(self._pending.values())

    def next_request(self) -> CameraReviewRequest | None:
        if not self._pending:
            return None
        return min(
            self._pending.values(),
            key=lambda item: (
                _REASON_PRIORITY[item.reason],
                _KIND_PRIORITY.get(item.inspection_kind, 99),
                item.requested_at,
                item.unit_id,
            ),
        )

    def mark_started(self, unit_id: str, inspection_kind: str, now: float) -> CameraReviewRequest:
        request = self._pending[(str(unit_id), str(inspection_kind))]
        request.status = "RUNNING"
        request.started_at = float(now)
        return request

    def complete(
        self,
        unit_id: str,
        inspection_kind: str,
        *,
        now: float,
        result: str,
    ) -> dict[str, Any] | None:
        request = self._pending.pop((str(unit_id), str(inspection_kind)), None)
        if request is None:
            return None
        record = {
            **request.as_dict(),
            "status": "COMPLETED",
            "completed_at": float(now),
            "result": str(result),
        }
        self._history.append(record)
        if request.reason in {
            CameraReviewReason.HIGH_RELIABILITY,
            CameraReviewReason.FIRST_ARTICLE,
        }:
            self._route_reviews.add(request.key)
        return record

    def cancel(
        self,
        unit_id: str,
        inspection_kind: str,
        *,
        now: float,
        result: str,
    ) -> dict[str, Any] | None:
        request = self._pending.pop((str(unit_id), str(inspection_kind)), None)
        if request is None:
            return None
        record = {
            **request.as_dict(),
            "status": "CANCELLED",
            "completed_at": float(now),
            "result": str(result),
        }
        self._history.append(record)
        return record

    def route_review_completed(self, unit_id: str, inspection_kind: str) -> bool:
        return (str(unit_id), str(inspection_kind)) in self._route_reviews

    def reset(self) -> None:
        self._pending.clear()
        self._history.clear()
        self._route_reviews.clear()

    def snapshot(self) -> dict[str, Any]:
        return {
            "policy": "SINGLE_ARM3_CAMERA_WITH_S3B_CLOSEUPS",
            "pending_reviews": [
                item.as_dict()
                for item in sorted(
                    self._pending.values(),
                    key=lambda value: (value.requested_at, value.unit_id, value.inspection_kind),
                )
            ],
            "review_history": list(self._history),
            "rules": {
                "standard": "Arm3末端相机在S2B和S4执行主检",
                "arm3_camera_offline": "工件原位阻塞，等待Arm3恢复或人工处理",
                "post_rework": "返回对应工序后由Arm3末端相机重新采集确认",
                "high_reliability": "S2B主检后在S3B执行两类近景检测，再于S4终检",
                "non_preemptive_fin": True,
            },
        }


__all__ = [
    "CameraCoordinationPolicy",
    "CameraReviewReason",
    "CameraReviewRequest",
]
