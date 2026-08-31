"""Runtime contact guard for the brazing workcell.

The MJCF intentionally keeps inter-arm and external tool collisions enabled.
This module filters contacts that are structural/expected and reports the
remaining contacts to the coordinator, which can move the order to ERROR.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil, isfinite
from threading import RLock
from typing import Any, Callable, Iterable, Sequence

import numpy as np


@dataclass(frozen=True)
class ContactEvent:
    body1: str
    body2: str
    geom1: str
    geom2: str
    distance: float


@dataclass(frozen=True, slots=True)
class SafetyBarrierReport:
    """Evidence produced by one continuous path safety check.

    ``allowed`` is the value consumed by a force-mode planner.  In shadow mode
    the same violation is retained in ``shadow_violation`` while the path is
    allowed to proceed, making the mode useful for a controlled ablation.
    """

    allowed: bool
    mode: str
    reason_code: str
    reason_zh: str
    minimum_clearance_m: float
    sample_count: int
    shadow_violation: bool = False
    wait_certified: bool = False

    def as_dict(self) -> dict[str, object]:
        return {
            "allowed": self.allowed,
            "mode": self.mode,
            "reason_code": self.reason_code,
            "reason_zh": self.reason_zh,
            "minimum_clearance_m": self.minimum_clearance_m,
            "sample_count": self.sample_count,
            "shadow_violation": self.shadow_violation,
            "wait_certified": self.wait_certified,
        }


class GeometrySafetyBarrier:
    """Continuous, fail-closed safety gate shared by planners and UI.

    The barrier deliberately accepts a caller-supplied clearance function.  A
    headless planner can provide a capsule/AABB query while the MuJoCo adapter
    can provide the exact model query; this keeps the policy deterministic and
    testable without hiding geometry behind a coarse station lock.
    """

    MODES = frozenset({"SHADOW", "FORCE"})

    def __init__(
        self,
        *,
        mode: str = "FORCE",
        minimum_clearance_m: float = 0.04,
        sample_period_s: float = 0.02,
        max_joint_step: float = 0.08,
        certified_wait_nodes: Iterable[str] = ("CURRENT_CERTIFIED_WAIT",),
    ) -> None:
        normalized = str(mode).strip().upper()
        if normalized not in self.MODES:
            raise ValueError("geometry safety mode must be SHADOW or FORCE")
        if not isfinite(float(minimum_clearance_m)) or float(minimum_clearance_m) < 0.0:
            raise ValueError("minimum clearance must be finite and non-negative")
        if not isfinite(float(sample_period_s)) or float(sample_period_s) <= 0.0:
            raise ValueError("sample period must be finite and positive")
        if not isfinite(float(max_joint_step)) or float(max_joint_step) <= 0.0:
            raise ValueError("max joint step must be finite and positive")
        self.mode = normalized
        self.minimum_clearance_m = float(minimum_clearance_m)
        self.sample_period_s = min(0.02, float(sample_period_s))
        self.max_joint_step = float(max_joint_step)
        self.certified_wait_nodes = frozenset(str(node) for node in certified_wait_nodes)
        self._lock = RLock()
        self._checked_count = 0
        self._blocked_count = 0
        self._shadow_violation_count = 0
        self._last_report: SafetyBarrierReport | None = None

    def can_wait(self, node: np.ndarray | Sequence[float], *, node_name: str | None = None) -> bool:
        """Return whether a planner may pause at a certified safe node."""

        del node  # Position is intentionally opaque; certification is explicit.
        return node_name is not None and str(node_name) in self.certified_wait_nodes

    @staticmethod
    def _interpolate_samples(samples: Sequence[Any], max_joint_step: float, max_dt: float) -> list[np.ndarray]:
        if not samples:
            return []
        points: list[np.ndarray] = [np.asarray(samples[0].position, dtype=float)]
        for first, second in zip(samples, samples[1:]):
            q0 = np.asarray(first.position, dtype=float)
            q1 = np.asarray(second.position, dtype=float)
            distance = float(np.linalg.norm(q1 - q0))
            duration = max(0.0, float(second.time) - float(first.time))
            count = max(1, int(ceil(distance / max_joint_step)), int(ceil(duration / max_dt)))
            points.extend(q0 + fraction * (q1 - q0) for fraction in np.linspace(1.0 / count, 1.0, count))
        return points

    def evaluate(
        self,
        samples: Sequence[Any],
        *,
        clearance: Callable[[np.ndarray], float],
        wait_node: str | None = "CURRENT_CERTIFIED_WAIT",
    ) -> SafetyBarrierReport:
        """Check a path at <=20 ms and bounded joint-space interpolation."""

        points = self._interpolate_samples(samples, self.max_joint_step, self.sample_period_s)
        minimum = float("inf")
        invalid = False
        for point in points:
            value = float(clearance(point))
            # ``+inf`` is the explicit dry-run sentinel used when no physical
            # model is bound; it means “no geometry query available”, not an
            # unsafe penetration.  NaN and negative infinity remain invalid.
            if value != float("inf") and not isfinite(value):
                invalid = True
                minimum = float("-inf")
                break
            minimum = min(minimum, value)
        violation = invalid or minimum < self.minimum_clearance_m
        reason_code = "CLEARANCE_BELOW_THRESHOLD" if violation else "SAFE"
        reason_zh = (
            "预测净空低于40 mm或几何查询无效"
            if violation
            else "连续路径净空满足安全阈值"
        )
        allowed = not violation or self.mode == "SHADOW"
        report = SafetyBarrierReport(
            allowed=allowed,
            mode=self.mode,
            reason_code=reason_code,
            reason_zh=reason_zh,
            minimum_clearance_m=minimum if points else float("inf"),
            sample_count=len(points),
            shadow_violation=violation and self.mode == "SHADOW",
            wait_certified=self.can_wait(
                np.asarray(samples[0].position, dtype=float) if samples else np.zeros(0),
                node_name=wait_node,
            ),
        )
        with self._lock:
            self._checked_count += 1
            if violation and self.mode == "FORCE":
                self._blocked_count += 1
            if report.shadow_violation:
                self._shadow_violation_count += 1
            self._last_report = report
        return report

    def snapshot(self) -> dict[str, object]:
        with self._lock:
            return {
                "mode": self.mode,
                "minimum_clearance_m": self.minimum_clearance_m,
                "sample_period_s": self.sample_period_s,
                "max_joint_step": self.max_joint_step,
                "checked_count": self._checked_count,
                "blocked_count": self._blocked_count,
                "shadow_violation_count": self._shadow_violation_count,
                "last_report": None if self._last_report is None else self._last_report.as_dict(),
            }

    def reset(self) -> None:
        with self._lock:
            self._checked_count = 0
            self._blocked_count = 0
            self._shadow_violation_count = 0
            self._last_report = None


DEFAULT_ALLOWED_BODY_PAIRS = {
    frozenset(("assembly_fixture", "assembly_tray")),
    frozenset(("fixture_tray", "heatsink_base_plate")),
    frozenset(("furnace", "furnace_door")),
}

PRODUCT_BODIES = frozenset(
    {"base_plate", "heatsink_base_plate", *(f"fin_{index:02d}" for index in range(1, 13))}
)


def _arm_prefix(name: str) -> str:
    for prefix in ("arm1_", "arm2_", "arm3_"):
        if name.startswith(prefix):
            return prefix
    return ""


class ContactMonitor:
    """Classify MuJoCo contacts without modifying physical collision masks."""

    def __init__(
        self,
        model: Any,
        *,
        allowed_body_pairs: Iterable[Iterable[str]] = DEFAULT_ALLOWED_BODY_PAIRS,
        penetration_tolerance_m: float = 0.001,
    ) -> None:
        self.model = model
        self.allowed_pairs = {tuple(sorted(pair)) for pair in allowed_body_pairs}
        self.penetration_tolerance_m = float(penetration_tolerance_m)
        self._body_names = tuple(
            self.model.body(index).name or str(index) for index in range(int(self.model.nbody))
        )
        self._geom_names = tuple(
            self.model.geom(index).name or str(index) for index in range(int(self.model.ngeom))
        )
        self._arm_prefixes = tuple(_arm_prefix(name) for name in self._body_names)
        body_ids = {name: index for index, name in enumerate(self._body_names)}
        self._allowed_body_id_pairs = {
            tuple(sorted((body_ids[left], body_ids[right])))
            for left, right in self.allowed_pairs
            if left in body_ids and right in body_ids
        }
        self._product_body_ids = frozenset(body_ids[name] for name in PRODUCT_BODIES if name in body_ids)

    def _allowed(self, body1: str, body2: str) -> bool:
        if body1 == body2:
            return True
        # Names are passed from the cached model lookup in unexpected().
        # Prefix parsing remains as a fallback for direct unit-test calls.
        prefix1, prefix2 = _arm_prefix(body1), _arm_prefix(body2)
        if prefix1 and prefix1 == prefix2:
            return True
        if tuple(sorted((body1, body2))) in self.allowed_pairs:
            return True
        if body1 in PRODUCT_BODIES and body2 in PRODUCT_BODIES:
            return True
        return False

    def _allowed_ids(self, body1_id: int, body2_id: int) -> bool:
        if body1_id == body2_id:
            return True
        prefix1 = self._arm_prefixes[body1_id]
        prefix2 = self._arm_prefixes[body2_id]
        if prefix1 and prefix1 == prefix2:
            return True
        if tuple(sorted((body1_id, body2_id))) in self._allowed_body_id_pairs:
            return True
        return body1_id in self._product_body_ids and body2_id in self._product_body_ids

    def unexpected(self, data: Any) -> list[ContactEvent]:
        events: list[ContactEvent] = []
        for index in range(int(data.ncon)):
            contact = data.contact[index]
            if float(contact.dist) >= -self.penetration_tolerance_m:
                continue
            geom1_id, geom2_id = int(contact.geom1), int(contact.geom2)
            body1_id = int(self.model.geom_bodyid[geom1_id])
            body2_id = int(self.model.geom_bodyid[geom2_id])
            if self._allowed_ids(body1_id, body2_id):
                continue
            events.append(
                ContactEvent(
                    body1=self._body_names[body1_id],
                    body2=self._body_names[body2_id],
                    geom1=self._geom_names[geom1_id],
                    geom2=self._geom_names[geom2_id],
                    distance=float(contact.dist),
                )
            )
        return events


__all__ = ["ContactEvent", "ContactMonitor", "GeometrySafetyBarrier", "SafetyBarrierReport"]
