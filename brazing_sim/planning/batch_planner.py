"""Compatibility checks and runtime rack reservations for mixed orders."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from ..flexible.models import ProcessPlan


@dataclass(frozen=True, slots=True)
class BatchReservation:
    batch_id: str
    unit_id: str
    order_id: str
    tray_id: str
    layer_index: int


@dataclass(frozen=True, slots=True)
class BatchCandidate:
    plan: ProcessPlan
    queued_at: float


class BatchPlanner:
    def __init__(
        self,
        *,
        maximum_units: int = 3,
        max_wait_time: float = 30.0,
        allow_partial_batch: bool = True,
    ) -> None:
        self.maximum_units = int(maximum_units)
        self.max_wait_time = float(max_wait_time)
        self.allow_partial_batch = bool(allow_partial_batch)
        if not 1 <= self.maximum_units <= 3 or self.max_wait_time < 0:
            raise ValueError("invalid batch planning limits")

    def select_batch(self, candidates: Iterable[BatchCandidate], now: float) -> tuple[ProcessPlan, ...]:
        values = sorted(
            candidates,
            key=lambda item: (-item.plan.order.priority, item.queued_at, item.plan.order.order_id),
        )
        if not values:
            return ()
        selected: list[ProcessPlan] = []
        units = 0
        for candidate in values:
            if units + candidate.plan.quantity > self.maximum_units:
                continue
            if selected and not are_units_batch_compatible((*selected, candidate.plan)):
                continue
            selected.append(candidate.plan)
            units += candidate.plan.quantity
            if units == self.maximum_units:
                return tuple(selected)
        oldest_wait = max(0.0, float(now) - min(item.queued_at for item in values))
        if self.allow_partial_batch and selected and oldest_wait >= self.max_wait_time:
            return tuple(selected)
        return ()


def are_units_batch_compatible(units: Iterable[ProcessPlan]) -> bool:
    plans = tuple(units)
    if not plans:
        return False
    if sum(plan.quantity for plan in plans) > 3:
        return False
    return are_process_plans_compatible(plans)


def are_process_plans_compatible(units: Iterable[ProcessPlan]) -> bool:
    """Check recipe/material/height without applying an order quantity limit."""

    plans = tuple(units)
    if not plans:
        return False
    reference = plans[0]
    reference_recipe = reference.recipe
    reference_material = getattr(reference.product, "material_system", "demo_brazing_material")
    shelf_clearance = 0.12
    for plan in plans:
        recipe = plan.recipe
        material = getattr(plan.product, "material_system", "demo_brazing_material")
        product_height = plan.product.base_size_m[2] + plan.product.fin_size_m[2]
        if (
            recipe.name != reference_recipe.name
            or abs(recipe.peak_c - reference_recipe.peak_c) > 1e-9
            or abs(recipe.soak_seconds - reference_recipe.soak_seconds) > 1e-9
            or material != reference_material
            or product_height > shelf_clearance
        ):
            return False
    return True


__all__ = [
    "BatchCandidate",
    "BatchPlanner",
    "BatchReservation",
    "are_process_plans_compatible",
    "are_units_batch_compatible",
]
