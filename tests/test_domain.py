from __future__ import annotations

import pytest

from brazing_sim.config import create_product_state, derive_product_layout, make_order_spec
from brazing_sim.domain import (
    OrderStage,
    StateTransitionError,
    TaskSpec,
    TaskStatus,
)
from brazing_sim.resources import ResourceManager


def test_a_order_dimensions_and_fixed_scene_allocation() -> None:
    spec = make_order_spec("A")
    layout = derive_product_layout(spec)

    assert spec.base_size == pytest.approx((0.36, 0.22, 0.008))
    assert spec.fin_size == pytest.approx((0.30, 0.002, 0.06))
    assert spec.fin_count == 4
    assert spec.path_count == 8
    assert len(layout.fins) == 8
    assert len(layout.paths) == 16
    assert [fin.fin_id for fin in layout.active_fins] == [f"fin_{index:02d}" for index in range(1, 5)]
    assert [path.path_id for path in layout.active_paths] == [
        f"fin_{fin:02d}_{side}" for fin in range(1, 5) for side in ("left", "right")
    ]
    assert [fin.target_position[1] for fin in layout.active_fins] == pytest.approx((-0.09, -0.03, 0.03, 0.09))
    assert all(not fin.hidden and fin.collision_enabled for fin in layout.active_fins)
    assert all(fin.hidden and not fin.collision_enabled for fin in layout.fins[4:])
    assert all(path.hidden and not path.collision_enabled for path in layout.paths[8:])


@pytest.mark.parametrize(("fin_count", "path_count"), [(6, 12), (8, 16)])
def test_layout_supports_future_fin_counts_without_reallocation(fin_count: int, path_count: int) -> None:
    spec = make_order_spec("A", fin_count=fin_count, fin_pitch=0.025)
    layout = derive_product_layout(spec)
    assert len(layout.fins) == 8
    assert len(layout.paths) == 16
    assert len(layout.active_fins) == fin_count
    assert len(layout.active_paths) == path_count


def test_order_validation_rejects_overallocation_and_geometry_overflow() -> None:
    with pytest.raises(ValueError, match="fin_count"):
        make_order_spec("A", fin_count=9)
    with pytest.raises(ValueError, match="does not fit"):
        make_order_spec("A", fin_count=8, fin_pitch=0.04)
    with pytest.raises(KeyError, match="unknown order preset"):
        make_order_spec("missing")


def test_order_stage_transition_contract() -> None:
    product = create_product_state(order_id="transition")
    product.transition(OrderStage.BASE_LOADING)
    product.transition(OrderStage.FIN_ASSEMBLY)
    product.transition(OrderStage.PRE_INSPECTION)
    product.transition(OrderStage.MATERIAL_APPLICATION)
    with pytest.raises(StateTransitionError):
        product.transition(OrderStage.BRAZING)
    product.fail("actor timeout", now=12.0)
    assert product.stage is OrderStage.ERROR
    assert product.completed_at == 12.0


def test_rework_limit_moves_product_to_manual_review() -> None:
    product = create_product_state(order_id="rework")
    target = product.active_fins[1]
    assert product.record_rework(target.fin_id)
    assert product.record_rework(target.fin_id)
    assert not product.record_rework(target.fin_id)
    assert target.rework_attempts == 2
    assert product.kpi.fin_reworks == 2
    assert product.stage is OrderStage.MANUAL_REVIEW


def test_task_failure_is_idempotent_and_retry_is_bounded() -> None:
    task = TaskSpec(
        "inspect-1",
        "arm3",
        "PRE_INSPECT",
        resource="inspection_zone",
        resources=("assembly_fixture",),
        max_retries=1,
    )
    assert task.resources == ("inspection_zone", "assembly_fixture")
    task.mark_running(1.0)
    task.mark_failed(2.0, "camera")
    task.mark_failed(3.0, "duplicate event")
    assert task.error == "camera"
    assert task.completed_at == 2.0
    assert task.prepare_retry()
    assert task.retries == 1
    assert task.status is TaskStatus.READY
    task.mark_failed(4.0, "camera")
    assert not task.prepare_retry()


def test_resource_leases_are_mutually_exclusive_and_resettable() -> None:
    manager = ResourceManager()
    assert manager.acquire("assembly_fixture", "arm1", now=1.0)
    assert manager.acquire("assembly_fixture", "arm1", now=1.1)
    assert not manager.acquire("assembly_fixture", "arm2", now=1.2)
    assert manager.conflict_count == 1
    assert manager.held_by("assembly_fixture") == "arm1"
    assert not manager.release("assembly_fixture", "arm2")
    assert manager.release("assembly_fixture", "arm1")
    assert manager.acquire_many(("brazing_zone", "furnace_mouth"), "arm2", now=2.0)
    assert manager.release_all("arm2") == 2
    assert all(value is None for value in manager.snapshot().values())


def test_expiring_resource_lease_can_be_acquired_by_next_actor() -> None:
    manager = ResourceManager()
    assert manager.acquire("inspection_zone", "arm3", now=0.0, ttl=2.0)
    assert not manager.acquire("inspection_zone", "arm1", now=1.999)
    assert manager.acquire("inspection_zone", "arm1", now=2.0)
