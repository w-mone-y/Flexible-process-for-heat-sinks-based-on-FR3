"""Explainable earliest-finish dispatcher for the two V2 fin-install lines."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from math import isfinite
from types import MappingProxyType
from typing import Iterable, Mapping

from ..scheduling.rolling_horizon import (
    HorizonAction,
    HorizonDecisionContext,
    RollingHorizonPlanner,
)


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
    downstream_blocking_s: float = 0.0
    enabled: bool = True

    def __post_init__(self) -> None:
        if not isfinite(self.available_at) or self.seconds_per_fin <= 0:
            raise ValueError("resource timing must be finite and positive")
        if self.queued_fins < 0:
            raise ValueError("queued fin count must be non-negative")
        if not isfinite(self.downstream_blocking_s) or self.downstream_blocking_s < 0.0:
            raise ValueError("downstream blocking estimate must be finite and non-negative")
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
    downstream_blocking_s: float
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
    arm3_activated: bool
    arm3_expected_gain_s: float
    arm3_inspection_penalty_s: float
    arm3_blocking_penalty_s: float
    arm3_net_gain_s: float
    activation_reason_zh: str


class DualInstallDispatcher:
    """Assign one whole tray while treating every carried fin as non-preemptible."""

    def __init__(
        self,
        *,
        lateness_weight: float = 2.0,
        priority_weight: float = 0.02,
        minimum_arm3_net_gain_s: float = 0.5,
    ) -> None:
        if lateness_weight < 0 or priority_weight < 0 or minimum_arm3_net_gain_s < 0:
            raise ValueError("dispatcher weights must be non-negative")
        self.lateness_weight = float(lateness_weight)
        self.priority_weight = float(priority_weight)
        self.minimum_arm3_net_gain_s = float(minimum_arm3_net_gain_s)
        self.rolling_horizon = RollingHorizonPlanner(horizon_seconds=45.0, maximum_candidates=4)
        self._last_decision: InstallDecision | None = None

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
                downstream_blocking_s=resource.downstream_blocking_s,
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
            downstream_blocking_s=resource.downstream_blocking_s,
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
        arm1 = candidates[InstallBranch.ARM1_A]
        arm3 = candidates[InstallBranch.ARM3_B]
        if not isfinite(arm1.finish_at) and not isfinite(arm3.finish_at):
            raise RuntimeError("no installation branch is currently executable")
        if isfinite(arm1.finish_at) and isfinite(arm3.finish_at):
            # Compare against Arm3's unconstrained install finish, then charge
            # its reserved inspection window exactly once below. ``finish_at``
            # already includes that wait because it is the physical forecast.
            arm3_unconstrained_finish = arm3.finish_at - arm3.inspection_wait_s
            expected_gain = arm1.finish_at - arm3_unconstrained_finish
        elif isfinite(arm3.finish_at):
            expected_gain = self.rolling_horizon.horizon_seconds
        else:
            expected_gain = -self.rolling_horizon.horizon_seconds
        inspection_penalty = arm3.inspection_wait_s
        blocking_penalty = arm3.downstream_blocking_s
        net_gain = expected_gain - inspection_penalty - blocking_penalty
        next_inspection = next(
            iter(resource_map[InstallBranch.ARM3_B].inspection_reservations),
            None,
        )
        horizon_actions: list[HorizonAction] = []
        if isfinite(arm1.finish_at):
            horizon_actions.append(
                HorizonAction(
                    action_id=InstallBranch.ARM1_A.value,
                    action_zh="由Arm1完成整托盘翅片安装",
                    duration_s=arm1.processing_s,
                    projected_completion_s=arm1.finish_at,
                    due_date_penalty_s=arm1.lateness_s,
                )
            )
        if isfinite(arm3.finish_at):
            horizon_actions.append(
                HorizonAction(
                    action_id=InstallBranch.ARM3_B.value,
                    action_zh="启用Arm3空窗完成整托盘翅片安装",
                    duration_s=resource_map[InstallBranch.ARM3_B].seconds_per_fin,
                    projected_completion_s=arm3.finish_at,
                    blocks_resource="ARM3",
                    downstream_blocking_s=blocking_penalty,
                    due_date_penalty_s=arm3.lateness_s,
                )
            )
        horizon = self.rolling_horizon.choose(
            horizon_actions,
            HorizonDecisionContext(
                now=request.ready_at,
                critical_resource=(None if next_inspection is None else "ARM3"),
                critical_work_ready_in_s=(
                    float("inf")
                    if next_inspection is None
                    else max(0.0, next_inspection[0] - request.ready_at)
                ),
                critical_work_duration_s=(
                    0.0 if next_inspection is None else next_inspection[1] - next_inspection[0]
                ),
                critical_path_weight=2.0,
            ),
        )
        arm3_activated = (
            isfinite(arm3.finish_at)
            and horizon.selected_action_id == InstallBranch.ARM3_B.value
            and (not isfinite(arm1.finish_at) or net_gain >= self.minimum_arm3_net_gain_s)
        )
        if arm3_activated:
            selected = arm3
            activation_reason = "Arm3存在完整安装空窗，产线级净收益超过启用阈值"
        else:
            selected = arm1
            activation_reason = (
                "Arm3资源不可用，使用Arm1安装支路"
                if not isfinite(arm3.finish_at)
                else "Arm3局部节拍更快，但检测与下游阻塞代价超过收益"
            )
        other = next(
            candidate for candidate in candidates.values() if candidate.branch is not selected.branch
        )
        inspection_note = ""
        if arm3.inspection_wait_s > 0:
            inspection_note = f"；Arm3受检测预约影响等待{arm3.inspection_wait_s:.1f}秒"
        explanation = (
            f"{request.tray_id}分配至{selected.branch.value}：预计{selected.finish_at:.1f}秒完成，"
            f"另一支路预计{other.finish_at:.1f}秒完成{inspection_note}；"
            f"Arm3净收益{net_gain:.1f}秒（阈值{self.minimum_arm3_net_gain_s:.1f}秒）"
        )
        decision = InstallDecision(
            tray_id=request.tray_id,
            branch=selected.branch,
            start_at=selected.start_at,
            finish_at=selected.finish_at,
            candidates=MappingProxyType(candidates),
            explanation_zh=explanation,
            arm3_activated=arm3_activated,
            arm3_expected_gain_s=expected_gain,
            arm3_inspection_penalty_s=inspection_penalty,
            arm3_blocking_penalty_s=blocking_penalty,
            arm3_net_gain_s=net_gain,
            activation_reason_zh=activation_reason,
        )
        self._last_decision = decision
        return decision

    def reset(self) -> None:
        """Clear run-specific candidates and explanations."""

        self._last_decision = None
        self.rolling_horizon.reset()

    def snapshot(self) -> dict[str, object]:
        decision = self._last_decision
        if decision is None:
            return {
                "selected_branch": None,
                "horizon_seconds": self.rolling_horizon.horizon_seconds,
                "maximum_candidates": self.rolling_horizon.maximum_candidates,
                "minimum_arm3_net_gain_s": self.minimum_arm3_net_gain_s,
                "arm3_activation": {
                    "activated": False,
                    "reason_zh": "等待托盘进入分支决策点",
                },
                "rolling_horizon": dict(self.rolling_horizon.snapshot()),
                "candidates": [],
            }

        def finite_or_none(value: float) -> float | None:
            return value if isfinite(value) else None

        return {
            "selected_branch": decision.branch.value,
            "horizon_seconds": self.rolling_horizon.horizon_seconds,
            "maximum_candidates": self.rolling_horizon.maximum_candidates,
            "minimum_arm3_net_gain_s": self.minimum_arm3_net_gain_s,
            "arm3_activation": {
                "activated": decision.arm3_activated,
                "expected_gain_s": decision.arm3_expected_gain_s,
                "inspection_penalty_s": decision.arm3_inspection_penalty_s,
                "blocking_penalty_s": decision.arm3_blocking_penalty_s,
                "net_gain_s": decision.arm3_net_gain_s,
                "reason_zh": decision.activation_reason_zh,
            },
            "explanation_zh": decision.explanation_zh,
            "rolling_horizon": dict(self.rolling_horizon.snapshot()),
            "candidates": [
                {
                    "branch": candidate.branch.value,
                    "start_at": finite_or_none(candidate.start_at),
                    "finish_at": finite_or_none(candidate.finish_at),
                    "processing_s": finite_or_none(candidate.processing_s),
                    "queue_wait_s": finite_or_none(candidate.queue_wait_s),
                    "inspection_wait_s": candidate.inspection_wait_s,
                    "downstream_blocking_s": candidate.downstream_blocking_s,
                    "lateness_s": finite_or_none(candidate.lateness_s),
                    "cost": finite_or_none(candidate.cost),
                    "blocked_reason_zh": candidate.blocked_reason_zh,
                }
                for candidate in decision.candidates.values()
            ],
        }


__all__ = [
    "DualInstallDispatcher",
    "InstallBranch",
    "InstallCandidate",
    "InstallDecision",
    "InstallRequest",
    "InstallResourceState",
]
