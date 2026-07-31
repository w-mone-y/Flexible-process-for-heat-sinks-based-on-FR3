"""Six-pallet V2 ownership and lifecycle state machine."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from math import isfinite


class TrayOwner(str, Enum):
    EMPTY_BUFFER = "EMPTY_BUFFER"
    S1 = "S1"
    S2A = "S2A"
    S2B = "S2B"
    INSTALL_A = "INSTALL_A"
    INSTALL_B = "INSTALL_B"
    MERGE_A_WAIT = "MERGE_A_WAIT"
    MERGE_B_WAIT = "MERGE_B_WAIT"
    MERGE = "MERGE"
    S4 = "S4"
    BUFFER_1 = "BUFFER_1"
    BUFFER_2 = "BUFFER_2"
    BUFFER_3 = "BUFFER_3"
    FURNACE = "FURNACE"
    POST_SCAN = "POST_SCAN"
    OUTPUT = "OUTPUT"
    VIRTUAL_RETURN = "VIRTUAL_RETURN"


class TrayPhase(str, Enum):
    EMPTY_BUFFER = "EMPTY_BUFFER"
    BASE_LOADING = "BASE_LOADING"
    DISPENSING = "DISPENSING"
    MATERIAL_INSPECTION = "MATERIAL_INSPECTION"
    FIN_INSTALLATION = "FIN_INSTALLATION"
    MERGE_WAIT = "MERGE_WAIT"
    MERGING = "MERGING"
    PRE_BRAZE_INSPECTION = "PRE_BRAZE_INSPECTION"
    FURNACE_BUFFER = "FURNACE_BUFFER"
    BRAZING = "BRAZING"
    POST_BRAZE_INSPECTION = "POST_BRAZE_INSPECTION"
    DELIVERED = "DELIVERED"
    PRODUCT_REMOVED = "PRODUCT_REMOVED"
    VIRTUAL_RETURN = "VIRTUAL_RETURN"


@dataclass(slots=True)
class TrayState:
    tray_id: str
    phase: TrayPhase = TrayPhase.EMPTY_BUFFER
    owner: TrayOwner = TrayOwner.EMPTY_BUFFER
    order_id: str | None = None
    unit_id: str | None = None
    last_transition_at: float = 0.0

    def as_dict(self) -> dict[str, object]:
        return {
            "tray_id": self.tray_id,
            "phase": self.phase.value,
            "owner": self.owner.value,
            "order_id": self.order_id,
            "unit_id": self.unit_id,
            "last_transition_at": self.last_transition_at,
        }


_ALLOWED_OWNER_TRANSITIONS: dict[TrayOwner, frozenset[TrayOwner]] = {
    TrayOwner.EMPTY_BUFFER: frozenset({TrayOwner.S1}),
    TrayOwner.S1: frozenset({TrayOwner.S2A}),
    TrayOwner.S2A: frozenset({TrayOwner.S2B}),
    # Reverse edges are used only by camera-confirmed rework.  They are still
    # ownership-checked physical handoffs, never a teleport or free-joint write.
    TrayOwner.S2B: frozenset({TrayOwner.S2A, TrayOwner.INSTALL_A, TrayOwner.INSTALL_B}),
    TrayOwner.INSTALL_A: frozenset({TrayOwner.MERGE_A_WAIT}),
    TrayOwner.INSTALL_B: frozenset({TrayOwner.MERGE_B_WAIT}),
    TrayOwner.MERGE_A_WAIT: frozenset({TrayOwner.MERGE}),
    TrayOwner.MERGE_B_WAIT: frozenset({TrayOwner.MERGE}),
    TrayOwner.MERGE: frozenset({TrayOwner.S4}),
    TrayOwner.S4: frozenset(
        {
            TrayOwner.INSTALL_A,
            TrayOwner.INSTALL_B,
            TrayOwner.BUFFER_1,
            TrayOwner.BUFFER_2,
            TrayOwner.BUFFER_3,
        }
    ),
    TrayOwner.BUFFER_1: frozenset({TrayOwner.FURNACE}),
    TrayOwner.BUFFER_2: frozenset({TrayOwner.FURNACE}),
    TrayOwner.BUFFER_3: frozenset({TrayOwner.FURNACE}),
    TrayOwner.FURNACE: frozenset({TrayOwner.POST_SCAN}),
    TrayOwner.POST_SCAN: frozenset({TrayOwner.OUTPUT}),
    TrayOwner.OUTPUT: frozenset({TrayOwner.VIRTUAL_RETURN}),
    TrayOwner.VIRTUAL_RETURN: frozenset({TrayOwner.EMPTY_BUFFER}),
}


class TrayFlowController:
    """The single writer for V2 pallet ownership."""

    def __init__(self, *, capacity: int = 6) -> None:
        if capacity != 6:
            raise ValueError("V2 physically preallocates exactly six trays")
        self._trays = {
            f"V2_TRAY_{index:02d}": TrayState(f"V2_TRAY_{index:02d}") for index in range(1, capacity + 1)
        }
        self._occupied: dict[TrayOwner, set[str]] = {}
        self._owner_capacity = {owner: 1 for owner in TrayOwner}
        self._owner_capacity[TrayOwner.EMPTY_BUFFER] = capacity
        self._owner_capacity[TrayOwner.FURNACE] = 3
        self._owner_capacity[TrayOwner.VIRTUAL_RETURN] = capacity

    @property
    def trays(self) -> tuple[TrayState, ...]:
        return tuple(self._trays.values())

    def get(self, tray_id: str) -> TrayState:
        try:
            return self._trays[str(tray_id)]
        except KeyError as exc:
            raise KeyError(f"unknown V2 tray: {tray_id}") from exc

    @staticmethod
    def _validate_now(now: float) -> float:
        timestamp = float(now)
        if not isfinite(timestamp):
            raise ValueError("transition time must be finite")
        return timestamp

    def assign_order(self, order_id: str, unit_id: str, *, now: float) -> TrayState:
        if not order_id or not unit_id:
            raise ValueError("order and unit ids are required")
        timestamp = self._validate_now(now)
        if self._occupied.get(TrayOwner.S1):
            raise RuntimeError("S1 is occupied")
        tray = next(
            (item for item in self._trays.values() if item.owner is TrayOwner.EMPTY_BUFFER),
            None,
        )
        if tray is None:
            raise RuntimeError("six-tray V2 pool is exhausted")
        if any(item.unit_id == unit_id for item in self._trays.values()):
            raise RuntimeError(f"unit already owns a tray: {unit_id}")
        tray.order_id = str(order_id)
        tray.unit_id = str(unit_id)
        tray.owner = TrayOwner.S1
        tray.phase = TrayPhase.BASE_LOADING
        tray.last_transition_at = timestamp
        self._occupied.setdefault(TrayOwner.S1, set()).add(tray.tray_id)
        return tray

    def handoff(
        self,
        tray_id: str,
        expected_owner: TrayOwner,
        new_owner: TrayOwner,
        phase: TrayPhase,
        *,
        now: float,
    ) -> TrayState:
        timestamp = self._validate_now(now)
        tray = self.get(tray_id)
        expected_owner = TrayOwner(expected_owner)
        new_owner = TrayOwner(new_owner)
        phase = TrayPhase(phase)
        if tray.owner is not expected_owner:
            raise RuntimeError(
                f"ownership mismatch for {tray_id}: expected {expected_owner.value}, "
                f"actual {tray.owner.value}"
            )
        if new_owner not in _ALLOWED_OWNER_TRANSITIONS[expected_owner]:
            raise ValueError(f"illegal tray owner transition: {expected_owner.value} -> {new_owner.value}")
        occupants = self._occupied.setdefault(new_owner, set())
        if tray_id not in occupants and len(occupants) >= self._owner_capacity[new_owner]:
            raise RuntimeError(f"{new_owner.value} is occupied by {', '.join(sorted(occupants))}")
        previous = self._occupied.get(expected_owner, set())
        previous.discard(tray_id)
        if not previous:
            self._occupied.pop(expected_owner, None)
        occupants.add(tray_id)
        tray.owner = new_owner
        tray.phase = phase
        tray.last_transition_at = timestamp
        return tray

    def start_virtual_return(self, tray_id: str, *, now: float) -> TrayState:
        tray = self.get(tray_id)
        if tray.phase is not TrayPhase.PRODUCT_REMOVED:
            raise RuntimeError("virtual return requires confirmed product removal")
        return self.handoff(
            tray_id,
            TrayOwner.OUTPUT,
            TrayOwner.VIRTUAL_RETURN,
            TrayPhase.VIRTUAL_RETURN,
            now=now,
        )

    def mark_product_removed(self, tray_id: str, *, now: float) -> TrayState:
        timestamp = self._validate_now(now)
        tray = self.get(tray_id)
        if tray.owner is not TrayOwner.OUTPUT or tray.phase is not TrayPhase.DELIVERED:
            raise RuntimeError("product removal requires a delivered tray at the output")
        tray.phase = TrayPhase.PRODUCT_REMOVED
        tray.last_transition_at = timestamp
        return tray

    def complete_virtual_return(self, tray_id: str, *, now: float) -> TrayState:
        tray = self.handoff(
            tray_id,
            TrayOwner.VIRTUAL_RETURN,
            TrayOwner.EMPTY_BUFFER,
            TrayPhase.EMPTY_BUFFER,
            now=now,
        )
        tray.order_id = None
        tray.unit_id = None
        return tray

    def reset(self, *, now: float = 0.0) -> None:
        timestamp = self._validate_now(now)
        self._occupied.clear()
        for tray in self._trays.values():
            tray.phase = TrayPhase.EMPTY_BUFFER
            tray.owner = TrayOwner.EMPTY_BUFFER
            tray.order_id = None
            tray.unit_id = None
            tray.last_transition_at = timestamp

    def as_dict(self) -> dict[str, object]:
        return {
            "capacity": len(self._trays),
            "trays": [tray.as_dict() for tray in self.trays],
        }


__all__ = ["TrayFlowController", "TrayOwner", "TrayPhase", "TrayState"]
