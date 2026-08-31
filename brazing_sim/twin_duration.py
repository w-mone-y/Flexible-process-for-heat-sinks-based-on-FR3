"""Deterministic shadow estimates learned from physical completion events."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite, sqrt
from typing import Any


@dataclass(frozen=True, slots=True)
class DurationEstimate:
    mean_s: float
    deviation_s: float
    sample_count: int
    confidence: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "mean_s": round(self.mean_s, 6),
            "deviation_s": round(self.deviation_s, 6),
            "sample_count": self.sample_count,
            "confidence": self.confidence,
        }


@dataclass(slots=True)
class _DurationStats:
    mean_s: float
    deviation_s: float = 0.0
    sample_count: int = 0

    def update(self, duration_s: float, alpha: float) -> None:
        if self.sample_count == 0:
            self.mean_s = duration_s
            self.deviation_s = 0.0
            self.sample_count = 1
            return
        residual = duration_s - self.mean_s
        self.mean_s += alpha * residual
        self.deviation_s = sqrt(max(0.0, (1.0 - alpha) * (self.deviation_s**2 + alpha * residual**2)))
        self.sample_count += 1

    def estimate(self) -> DurationEstimate:
        confidence = "LOW" if self.sample_count < 3 else "MEDIUM" if self.sample_count < 10 else "HIGH"
        return DurationEstimate(self.mean_s, self.deviation_s, self.sample_count, confidence)


class ShadowDurationEstimator:
    """EWMA duration model that never mutates task estimates or dispatch state."""

    def __init__(self, *, prior_seconds: float = 1.0, alpha: float = 0.35) -> None:
        if not isfinite(float(prior_seconds)) or prior_seconds <= 0.0:
            raise ValueError("prior_seconds must be finite and positive")
        if not 0.0 < float(alpha) <= 1.0:
            raise ValueError("alpha must be in (0, 1]")
        self.prior_seconds = float(prior_seconds)
        self.alpha = float(alpha)
        self._starts: dict[str, float] = {}
        self._stats: dict[str, _DurationStats] = {}

    @staticmethod
    def _key(task_type: str, resource_id: str) -> str:
        return f"{str(task_type).upper()}|{str(resource_id).upper()}"

    def observe_started(self, task_id: str, started_at: float) -> None:
        if not isfinite(float(started_at)) or started_at < 0.0:
            return
        self._starts[str(task_id)] = float(started_at)

    def observe_completed(
        self,
        task_id: str,
        *,
        task_type: str,
        resource_id: str,
        finished_at: float,
    ) -> bool:
        started_at = self._starts.pop(str(task_id), None)
        if started_at is None or not isfinite(float(finished_at)):
            return False
        duration = float(finished_at) - started_at
        if not isfinite(duration) or duration <= 0.0:
            return False
        key = self._key(task_type, resource_id)
        stats = self._stats.setdefault(key, _DurationStats(self.prior_seconds))
        stats.update(duration, self.alpha)
        return True

    def predict(self, task_type: str, resource_id: str) -> DurationEstimate:
        stats = self._stats.get(self._key(task_type, resource_id))
        return (stats or _DurationStats(self.prior_seconds)).estimate()

    def sample_count(self, task_type: str, resource_id: str) -> int:
        return self.predict(task_type, resource_id).sample_count

    def snapshot(self) -> dict[str, dict[str, Any]]:
        return {key: stats.estimate().as_dict() for key, stats in sorted(self._stats.items())}

    def reset(self) -> None:
        self._starts.clear()
        self._stats.clear()


__all__ = ["DurationEstimate", "ShadowDurationEstimator"]
