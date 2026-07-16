"""Runtime contact guard for the brazing workcell.

The MJCF intentionally keeps inter-arm and external tool collisions enabled.
This module filters contacts that are structural/expected and reports the
remaining contacts to the coordinator, which can move the order to ERROR.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable


@dataclass(frozen=True)
class ContactEvent:
    body1: str
    body2: str
    geom1: str
    geom2: str
    distance: float


DEFAULT_ALLOWED_BODY_PAIRS = {
    frozenset(("assembly_fixture", "assembly_tray")),
    frozenset(("furnace", "furnace_door")),
}


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
        self.allowed_pairs = {frozenset(pair) for pair in allowed_body_pairs}
        self.penetration_tolerance_m = float(penetration_tolerance_m)

    def _allowed(self, body1: str, body2: str) -> bool:
        if body1 == body2:
            return True
        prefix1, prefix2 = _arm_prefix(body1), _arm_prefix(body2)
        if prefix1 and prefix1 == prefix2:
            return True
        if frozenset((body1, body2)) in self.allowed_pairs:
            return True
        if {body1, body2} <= {"base_plate", *(f"fin_{i:02d}" for i in range(1, 9))}:
            return True
        return False

    def unexpected(self, data: Any) -> list[ContactEvent]:
        events: list[ContactEvent] = []
        for index in range(int(data.ncon)):
            contact = data.contact[index]
            if float(contact.dist) >= -self.penetration_tolerance_m:
                continue
            geom1_id, geom2_id = int(contact.geom1), int(contact.geom2)
            body1_id = int(self.model.geom_bodyid[geom1_id])
            body2_id = int(self.model.geom_bodyid[geom2_id])
            body1 = self.model.body(body1_id).name or str(body1_id)
            body2 = self.model.body(body2_id).name or str(body2_id)
            if self._allowed(body1, body2):
                continue
            events.append(
                ContactEvent(
                    body1=body1,
                    body2=body2,
                    geom1=self.model.geom(geom1_id).name or str(geom1_id),
                    geom2=self.model.geom(geom2_id).name or str(geom2_id),
                    distance=float(contact.dist),
                )
            )
        return events


__all__ = ["ContactEvent", "ContactMonitor"]
