"""Three-layer, front-load/rear-unload V2 batch furnace state machine."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from math import isfinite
from typing import Iterable


@dataclass(frozen=True, slots=True)
class BatchRecipe:
    name: str
    material_system: str
    peak_c: float
    soak_seconds: float
    maximum_product_height_m: float

    def __post_init__(self) -> None:
        if not self.name or not self.material_system:
            raise ValueError("batch recipe requires a name and material system")
        if (
            not all(
                isfinite(value) for value in (self.peak_c, self.soak_seconds, self.maximum_product_height_m)
            )
            or self.peak_c <= 0
            or self.soak_seconds <= 0
            or self.maximum_product_height_m <= 0
        ):
            raise ValueError("batch recipe values must be finite and positive")

    def compatible_with(self, other: "BatchRecipe") -> bool:
        return (
            self.name == other.name
            and self.material_system == other.material_system
            and abs(self.peak_c - other.peak_c) <= 1.0e-9
            and abs(self.soak_seconds - other.soak_seconds) <= 1.0e-9
            and abs(self.maximum_product_height_m - other.maximum_product_height_m) <= 1.0e-9
        )


class FurnacePhase(str, Enum):
    IDLE = "IDLE"
    PLANNED = "PLANNED"
    LOADING = "LOADING"
    READY = "READY"
    PREHEAT = "PREHEAT"
    RAMP = "RAMP"
    SOAK = "SOAK"
    COOLING = "COOLING"
    READY_TO_UNLOAD = "READY_TO_UNLOAD"
    UNLOADING = "UNLOADING"
    COMPLETE = "COMPLETE"


@dataclass(slots=True)
class FurnaceLayer:
    index: int
    tray_id: str | None = None
    locked: bool = False

    def as_dict(self) -> dict[str, object]:
        return {"index": self.index, "tray_id": self.tray_id, "locked": self.locked}


@dataclass(slots=True)
class ThroughBatchFurnaceState:
    phase: FurnacePhase = FurnacePhase.IDLE
    front_door_open: bool = False
    rear_door_open: bool = False
    pusher_retracted: bool = True
    lift_clear: bool = True
    planned_trays: tuple[str, ...] = ()
    layers: list[FurnaceLayer] = field(default_factory=list)
    cycle_started_at: float | None = None
    demo_elapsed_s: float = 0.0
    real_equivalent_elapsed_s: float = 0.0
    complete: bool = False


class ThroughBatchFurnace:
    """Non-blocking furnace whose process time advances only through ``update``."""

    def __init__(
        self,
        *,
        capacity: int = 3,
        demo_cycle_seconds: float = 30.0,
        real_cycle_seconds: float = 3600.0,
        nominal_max_wait_seconds: float = 600.0,
    ) -> None:
        if capacity != 3:
            raise ValueError("the V2 furnace has exactly three physical rack layers")
        if min(demo_cycle_seconds, real_cycle_seconds, nominal_max_wait_seconds) <= 0:
            raise ValueError("furnace timing values must be positive")
        self.capacity = capacity
        self.demo_cycle_seconds = float(demo_cycle_seconds)
        self.real_cycle_seconds = float(real_cycle_seconds)
        self.nominal_max_wait_seconds = float(nominal_max_wait_seconds)
        self.state = ThroughBatchFurnaceState(layers=[FurnaceLayer(index) for index in range(capacity)])
        self._recipe: BatchRecipe | None = None
        self._planned_at = 0.0
        self._last_now = 0.0

    @staticmethod
    def _timestamp(now: float) -> float:
        value = float(now)
        if not isfinite(value):
            raise ValueError("simulation time must be finite")
        return value

    def _check_monotonic(self, now: float) -> float:
        value = self._timestamp(now)
        if value + 1.0e-12 < self._last_now:
            raise ValueError("simulation time must be monotonic")
        self._last_now = max(self._last_now, value)
        return value

    @property
    def ready_to_unload(self) -> bool:
        return self.state.phase in {FurnacePhase.READY_TO_UNLOAD, FurnacePhase.UNLOADING}

    def plan_batch(
        self,
        trays: Iterable[tuple[str, BatchRecipe]],
        *,
        now: float,
    ) -> None:
        timestamp = self._check_monotonic(now)
        values = tuple(trays)
        if not 1 <= len(values) <= self.capacity:
            raise ValueError("a V2 furnace batch must contain one to three trays")
        tray_ids = tuple(tray_id for tray_id, _ in values)
        if any(not tray_id for tray_id in tray_ids) or len(set(tray_ids)) != len(tray_ids):
            raise ValueError("batch tray ids must be unique and non-empty")
        reference = values[0][1]
        if any(not reference.compatible_with(recipe) for _, recipe in values[1:]):
            raise ValueError("incompatible tray recipes cannot share one furnace batch")
        if self.state.phase not in {FurnacePhase.IDLE, FurnacePhase.COMPLETE}:
            raise RuntimeError("furnace already owns an active batch")
        self.reset(now=timestamp)
        self._recipe = reference
        self._planned_at = timestamp
        self.state.planned_trays = tray_ids
        self.state.phase = FurnacePhase.PLANNED

    def open_front(self, *, now: float) -> None:
        self._check_monotonic(now)
        if self.state.phase not in {FurnacePhase.PLANNED, FurnacePhase.LOADING}:
            raise RuntimeError("front door may open only for a planned loading batch")
        if self.state.rear_door_open:
            raise RuntimeError("rear door must be closed before front loading")
        self.state.front_door_open = True
        self.state.phase = FurnacePhase.LOADING

    def append_loading_tray(
        self,
        tray_id: str,
        recipe: BatchRecipe,
        *,
        now: float,
    ) -> None:
        """Reserve one compatible arrival in an already open loading batch."""

        self._check_monotonic(now)
        if self.state.phase not in {FurnacePhase.PLANNED, FurnacePhase.LOADING}:
            raise RuntimeError("trays may be appended only while the front-loading batch is open")
        if self._recipe is None or not self._recipe.compatible_with(recipe):
            raise RuntimeError("incompatible tray recipe cannot join the open furnace batch")
        if not tray_id or tray_id in self.state.planned_trays:
            raise RuntimeError(f"tray is already reserved for this batch: {tray_id}")
        if len(self.state.planned_trays) >= self.capacity:
            raise RuntimeError("furnace loading batch is already full")
        self.state.planned_trays = (*self.state.planned_trays, tray_id)

    def load_front(self, tray_id: str, *, layer: int, now: float) -> None:
        self._check_monotonic(now)
        if not self.state.front_door_open or self.state.phase is not FurnacePhase.LOADING:
            raise RuntimeError("front loading requires the front door to be open")
        if tray_id not in self.state.planned_trays:
            raise RuntimeError(f"tray is not reserved for this batch: {tray_id}")
        if not 0 <= layer < self.capacity:
            raise ValueError("rack layer is outside furnace capacity")
        if any(value.tray_id == tray_id for value in self.state.layers):
            raise RuntimeError(f"tray is already loaded: {tray_id}")
        target = self.state.layers[layer]
        if target.tray_id is not None:
            raise RuntimeError(f"furnace layer {layer} is occupied")
        target.tray_id = tray_id
        target.locked = False

    def lock_layer(self, layer: int, *, now: float) -> None:
        self._check_monotonic(now)
        target = self.state.layers[layer]
        if target.tray_id is None:
            raise RuntimeError("cannot lock an empty furnace layer")
        target.locked = True

    def close_front(self, *, now: float) -> None:
        self._check_monotonic(now)
        loaded = {layer.tray_id for layer in self.state.layers if layer.tray_id is not None}
        if loaded != set(self.state.planned_trays):
            raise RuntimeError("all planned trays must be loaded before closing the front door")
        if any(not layer.locked for layer in self.state.layers if layer.tray_id is not None):
            raise RuntimeError("all loaded furnace layers must be locked")
        if not self.state.pusher_retracted or not self.state.lift_clear:
            raise RuntimeError("transfer mechanism must clear the furnace mouth")
        self.state.front_door_open = False
        self.state.phase = FurnacePhase.READY

    def start_cycle(self, *, now: float) -> None:
        timestamp = self._check_monotonic(now)
        if self.state.front_door_open:
            raise RuntimeError("front door must be closed before the furnace cycle")
        if self.state.rear_door_open:
            raise RuntimeError("rear door must be closed before the furnace cycle")
        if self.state.phase is not FurnacePhase.READY:
            raise RuntimeError("all planned layers must be loaded and locked before cycle start")
        self.state.phase = FurnacePhase.PREHEAT
        self.state.cycle_started_at = timestamp
        self.state.demo_elapsed_s = 0.0
        self.state.real_equivalent_elapsed_s = 0.0

    def update(self, now: float) -> ThroughBatchFurnaceState:
        timestamp = self._check_monotonic(now)
        if self.state.cycle_started_at is None or self.state.phase in {
            FurnacePhase.IDLE,
            FurnacePhase.PLANNED,
            FurnacePhase.LOADING,
            FurnacePhase.READY,
            FurnacePhase.READY_TO_UNLOAD,
            FurnacePhase.UNLOADING,
            FurnacePhase.COMPLETE,
        }:
            return self.state
        elapsed = min(self.demo_cycle_seconds, max(0.0, timestamp - self.state.cycle_started_at))
        fraction = elapsed / self.demo_cycle_seconds
        self.state.demo_elapsed_s = elapsed
        self.state.real_equivalent_elapsed_s = fraction * self.real_cycle_seconds
        if fraction < 0.20:
            self.state.phase = FurnacePhase.PREHEAT
        elif fraction < 0.45:
            self.state.phase = FurnacePhase.RAMP
        elif fraction < 0.75:
            self.state.phase = FurnacePhase.SOAK
        elif fraction < 1.0:
            self.state.phase = FurnacePhase.COOLING
        else:
            self.state.phase = FurnacePhase.READY_TO_UNLOAD
        return self.state

    def open_rear(self, *, now: float) -> None:
        self._check_monotonic(now)
        if not self.ready_to_unload:
            raise RuntimeError("rear door remains interlocked until cooling completes")
        if self.state.front_door_open:
            raise RuntimeError("front door must remain closed during rear unloading")
        self.state.rear_door_open = True
        self.state.phase = FurnacePhase.UNLOADING

    def unload_rear(self, *, now: float) -> str:
        self._check_monotonic(now)
        if not self.state.rear_door_open or self.state.phase is not FurnacePhase.UNLOADING:
            raise RuntimeError("rear unloading requires the rear door to be open")
        occupied = [layer for layer in self.state.layers if layer.tray_id is not None]
        if not occupied:
            raise RuntimeError("furnace rack is empty")
        layer = max(occupied, key=lambda value: value.index)
        assert layer.tray_id is not None
        tray_id = layer.tray_id
        layer.tray_id = None
        layer.locked = False
        return tray_id

    def close_rear(self, *, now: float) -> None:
        """Close the rear door only after every physical unload has settled.

        ``unload_rear`` releases logical rack ownership when an extractor
        starts moving.  Door closure is deliberately separate so the runtime
        cannot close the panel while the final tray is still crossing it.
        """

        self._check_monotonic(now)
        if not self.state.rear_door_open or self.state.phase is not FurnacePhase.UNLOADING:
            raise RuntimeError("rear door may close only during rear unloading")
        if any(layer.tray_id is not None for layer in self.state.layers):
            raise RuntimeError("all furnace layers must be empty before closing the rear door")
        self.state.rear_door_open = False
        self.state.phase = FurnacePhase.COMPLETE
        self.state.complete = True

    def should_release_partial_batch(
        self,
        *,
        now: float,
        earliest_due_at: float | None = None,
        estimated_remaining_s: float = 0.0,
    ) -> bool:
        timestamp = self._check_monotonic(now)
        if not self.state.planned_trays or len(self.state.planned_trays) >= self.capacity:
            return False
        wait_expired = timestamp - self._planned_at >= self.nominal_max_wait_seconds
        due_risk = earliest_due_at is not None and timestamp + max(0.0, estimated_remaining_s) >= float(
            earliest_due_at
        )
        return wait_expired or due_risk

    def reset(self, *, now: float = 0.0) -> None:
        timestamp = self._timestamp(now)
        self._last_now = timestamp
        self._recipe = None
        self._planned_at = timestamp
        self.state = ThroughBatchFurnaceState(layers=[FurnaceLayer(index) for index in range(self.capacity)])

    def as_dict(self) -> dict[str, object]:
        return {
            "phase": self.state.phase.value,
            "front_door_open": self.state.front_door_open,
            "rear_door_open": self.state.rear_door_open,
            "planned_trays": list(self.state.planned_trays),
            "layers": [layer.as_dict() for layer in self.state.layers],
            "demo_elapsed_s": self.state.demo_elapsed_s,
            "real_equivalent_elapsed_s": self.state.real_equivalent_elapsed_s,
            "demo_cycle_seconds": self.demo_cycle_seconds,
            "real_cycle_seconds": self.real_cycle_seconds,
            "complete": self.state.complete,
        }


__all__ = [
    "BatchRecipe",
    "FurnaceLayer",
    "FurnacePhase",
    "ThroughBatchFurnace",
    "ThroughBatchFurnaceState",
]
