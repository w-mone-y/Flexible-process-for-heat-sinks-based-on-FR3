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


__all__ = ["ContactEvent", "ContactMonitor"]
