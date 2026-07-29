"""Explainable earliest-finish dispatcher for the two V2 fin-install lines."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from math import isfinite
from types import MappingProxyType
from typing import Iterable, Mapping


class InstallBranch(str, Enum):
    ARM1_A = "ARM1_A"
    ARM3_B = "ARM3_B"


@dataclass(frozen=True, slots=True)
class InstallRequest:
    tray_id: str
    fin_count: int
    ready_at: float
    due_at: float | None = None
    priority: int = 10

    def __post_init__(self) -> None:
        if not self.tray_id or not 1 <= self.fin_count <= 12:
            raise ValueError("installation request must identify a tray with 1..12 fins")
        values = (self.ready_at,) if self.due_at is None else (self.ready_at, self.due_at)
        if not all(isfinite(value) for value in values):
            raise ValueError("installation timestamps must be finite")
        if self.priority < 0:
            raise ValueError("priority must be non-negative")


@dataclass(frozen=True, slots=True)
class InstallResourceState:
    branch: InstallBranch
    available_at: float
    seconds_per_fin: float
    queued_fins: int = 0
    inspection_reservations: tuple[tuple[float, float], ...] = ()
    enabled: bool = True

    def __post_init__(self) -> None:
        if not isfinite(self.available_at) or self.seconds_per_fin <= 0:
            raise ValueError("resource timing must be finite and positive")
        if self.queued_fins < 0:
            raise ValueError("queued fin count must be non-negative")
        previous_end = float("-inf")
        for start, end in self.inspection_reservations:
            if not isfinite(start) or not isfinite(end) or end <= start or start < previous_end:
                raise ValueError("inspection reservations must be sorted non-overlapping intervals")
            previous_end = end


@dataclass(frozen=True, slots=True)
class InstallCandidate:
    branch: InstallBranch
    start_at: float
    finish_at: float
    processing_s: float
    queue_wait_s: float
    inspection_wait_s: float
    lateness_s: float
    cost: float
    blocked_reason_zh: str = ""


@dataclass(frozen=True, slots=True)
class InstallDecision:
    tray_id: str
    branch: InstallBranch
    start_at: float
    finish_at: float
    candidates: Mapping[InstallBranch, InstallCandidate]
    explanation_zh: str


class DualInstallDispatcher:
    """Assign one whole tray while treating every carried fin as non-preemptible."""

    def __init__(self, *, lateness_weight: float = 2.0, priority_weight: float = 0.02) -> None:
        if lateness_weight < 0 or priority_weight < 0:
            raise ValueError("dispatcher weights must be non-negative")
        self.lateness_weight = float(lateness_weight)
        self.priority_weight = float(priority_weight)

    @staticmethod
    def _schedule_fins(
        start_at: float,
        fin_count: int,
        seconds_per_fin: float,
        reservations: tuple[tuple[float, float], ...],
    ) -> tuple[float, float]:
        cursor = float(start_at)
        inspection_wait = 0.0
        for _ in range(fin_count):
            for reserved_start, reserved_end in reservations:
                # Do not begin a fin if carrying it would collide with an
                # inspection reservation. A fin already in flight remains
                # non-preemptible by construction.
                if cursor < reserved_end and cursor + seconds_per_fin > reserved_start:
                    wait = max(0.0, reserved_end - cursor)
                    cursor += wait
                    inspection_wait += wait
            cursor += seconds_per_fin
        return cursor, inspection_wait

    def _candidate(
        self,
        request: InstallRequest,
        resource: InstallResourceState,
    ) -> InstallCandidate:
        if not resource.enabled:
            return InstallCandidate(
                branch=resource.branch,
                start_at=float("inf"),
                finish_at=float("inf"),
                processing_s=float("inf"),
                queue_wait_s=float("inf"),
                inspection_wait_s=0.0,
                lateness_s=float("inf"),
                cost=float("inf"),
                blocked_reason_zh="资源不可用",
            )
        queue_duration = resource.queued_fins * resource.seconds_per_fin
        start_at = max(request.ready_at, resource.available_at + queue_duration)
        finish_at, inspection_wait = self._schedule_fins(
            start_at,
            request.fin_count,
            resource.seconds_per_fin,
            resource.inspection_reservations if resource.branch is InstallBranch.ARM3_B else (),
        )
        queue_wait = max(0.0, start_at - request.ready_at)
        processing = request.fin_count * resource.seconds_per_fin
        lateness = 0.0 if request.due_at is None else max(0.0, finish_at - request.due_at)
        urgency_credit = request.priority * self.priority_weight
        cost = finish_at + lateness * self.lateness_weight - urgency_credit
        return InstallCandidate(
            branch=resource.branch,
            start_at=start_at,
            finish_at=finish_at,
            processing_s=processing,
            queue_wait_s=queue_wait,
            inspection_wait_s=inspection_wait,
            lateness_s=lateness,
            cost=cost,
        )

    def assign(
        self,
        request: InstallRequest,
        resources: Iterable[InstallResourceState],
    ) -> InstallDecision:
        resource_map = {resource.branch: resource for resource in resources}
        missing = set(InstallBranch).difference(resource_map)
        if missing:
            raise ValueError(f"missing installation resources: {', '.join(item.value for item in missing)}")
        candidates = {branch: self._candidate(request, resource_map[branch]) for branch in InstallBranch}
        selected = min(candidates.values(), key=lambda item: (item.cost, item.branch.value))
        if not isfinite(selected.finish_at):
            raise RuntimeError("no installation branch is currently executable")
        other = next(
            candidate for candidate in candidates.values() if candidate.branch is not selected.branch
        )
        inspection_note = ""
        arm3 = candidates[InstallBranch.ARM3_B]
        if arm3.inspection_wait_s > 0:
            inspection_note = f"；Arm3受检测预约影响等待{arm3.inspection_wait_s:.1f}秒"
        explanation = (
            f"{request.tray_id}分配至{selected.branch.value}：预计{selected.finish_at:.1f}秒完成，"
            f"另一支路预计{other.finish_at:.1f}秒完成{inspection_note}"
        )
        return InstallDecision(
            tray_id=request.tray_id,
            branch=selected.branch,
            start_at=selected.start_at,
            finish_at=selected.finish_at,
            candidates=MappingProxyType(candidates),
            explanation_zh=explanation,
        )


__all__ = [
    "DualInstallDispatcher",
    "InstallBranch",
    "InstallCandidate",
    "InstallDecision",
    "InstallRequest",
    "InstallResourceState",
]
