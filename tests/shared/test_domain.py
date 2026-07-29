from __future__ import annotations

import pytest

from brazing_sim.config import (
    DISPENSER_CONFIG,
    DispenserConfig,
    create_product_state,
    derive_product_layout,
    make_order_spec,
)
from brazing_sim.domain import (
    FixtureStatus,
    OrderStage,
    PressState,
    StateTransitionError,
    TaskSpec,
    TaskStatus,
)
from brazing_sim.resources import ResourceManager


@pytest.mark.parametrize(
    (
        "preset",
        "base_size",
        "fin_size",
        "fin_count",
        "fin_pitch",
        "path_count",
        "comb_module",
        "bead_offset",
        "fin_y",
    ),
    [
        (
            "A",
            (0.36, 0.22, 0.008),
            (0.30, 0.002, 0.06),
            5,
            0.020,
            10,
            "comb_insert_20mm",
            0.0025,
            (-0.040, -0.020, 0.000, 0.020, 0.040),
        ),
        (
            "B",
            (0.36, 0.24, 0.008),
            (0.30, 0.002, 0.06),
            4,
            0.030,
            8,
            "comb_insert_30mm",
            0.0025,
            (-0.045, -0.015, 0.015, 0.045),
        ),
        (
            "C",
            (0.34, 0.20, 0.008),
            (0.28, 0.0018, 0.055),
            7,
            0.015,
            14,
            "comb_insert_15mm",
            0.0022,
            (-0.045, -0.030, -0.015, 0.000, 0.015, 0.030, 0.045),
        ),
    ],
)
def test_product_presets_generate_expected_fins_comb_and_dual_paths(
    preset: str,
    base_size: tuple[float, float, float],
    fin_size: tuple[float, float, float],
    fin_count: int,
    fin_pitch: float,
    path_count: int,
    comb_module: str,
    bead_offset: float,
    fin_y: tuple[float, ...],
) -> None:
    spec = make_order_spec(preset)
    layout = derive_product_layout(spec)

    assert spec.base_size == pytest.approx(base_size)
    assert spec.fin_size == pytest.approx(fin_size)
    assert spec.fin_count == fin_count
    assert spec.fin_pitch == pytest.approx(fin_pitch)
    assert spec.path_count == path_count
    assert spec.comb_module_name == comb_module

    # The MJCF viewer allocation remains fixed while only the order's active
    # subset changes, so switching A/B/C does not require rebuilding MuJoCo.
    assert len(layout.fins) == 12
    assert len(layout.paths) == 24
    assert [fin.fin_id for fin in layout.active_fins] == [
        f"fin_{index:02d}" for index in range(1, fin_count + 1)
    ]
    assert [path.path_id for path in layout.active_paths] == [
        f"slot_{slot:02d}_{side}" for slot in range(1, fin_count + 1) for side in ("left", "right")
    ]
    assert [fin.target_position[1] for fin in layout.active_fins] == pytest.approx(fin_y)

    for fin, left_path, right_path in zip(
        layout.active_fins,
        layout.active_paths[::2],
        layout.active_paths[1::2],
    ):
        assert left_path.slot_id == right_path.slot_id == f"slot_{fin.index + 1:02d}"
        assert left_path.fin_id == right_path.fin_id == fin.fin_id
        assert left_path.local_start[1] == pytest.approx(fin.target_position[1] - bead_offset)
        assert right_path.local_start[1] == pytest.approx(fin.target_position[1] + bead_offset)
        assert left_path.local_start[0] == pytest.approx(-base_size[0] / 2.0 + 0.015)
        assert right_path.local_end[0] == pytest.approx(base_size[0] / 2.0 - 0.015)
        assert left_path.name == f"{left_path.path_id}_brazing_path"
        assert right_path.name == f"{right_path.path_id}_brazing_path"

    assert all(not fin.hidden and fin.collision_enabled for fin in layout.active_fins)
    assert all(fin.hidden and not fin.collision_enabled for fin in layout.fins[fin_count:])
    assert all(path.hidden and not path.collision_enabled for path in layout.paths[path_count:])


def test_dispenser_spacing_is_symmetric_and_outside_the_fin_faces() -> None:
    assert DISPENSER_CONFIG.nozzle_spacing == pytest.approx(
        2.0 * DISPENSER_CONFIG.bead_offset_from_slot_center
    )
    assert DISPENSER_CONFIG.bead_offset_from_slot_center > DISPENSER_CONFIG.fin_thickness / 2.0

    wider_fin = DispenserConfig(
        fin_thickness=0.004,
        bead_offset_from_slot_center=0.005,
        nozzle_spacing=0.010,
    )
    assert wider_fin.required_bead_offset(wider_fin.fin_thickness) == pytest.approx(0.003)

    with pytest.raises(ValueError, match="bead offset"):
        DispenserConfig(
            fin_thickness=0.004,
            bead_offset_from_slot_center=0.0015,
            nozzle_spacing=0.003,
        ).validate_for_fin_thickness(0.004)
    with pytest.raises(ValueError, match=r"bead.?offset"):
        make_order_spec("A", fin_size=(0.30, 0.006, 0.06))


@pytest.mark.parametrize(
    ("fin_count", "fin_pitch", "path_count"),
    [(6, 0.025, 12), (8, 0.025, 16), (12, 0.018, 24)],
)
def test_layout_supports_future_fin_counts_without_reallocation(
    fin_count: int, fin_pitch: float, path_count: int
) -> None:
    spec = make_order_spec("A", fin_count=fin_count, fin_pitch=fin_pitch)
    layout = derive_product_layout(spec)
    assert len(layout.fins) == 12
    assert len(layout.paths) == 24
    assert len(layout.active_fins) == fin_count
    assert len(layout.active_paths) == path_count


def test_order_validation_rejects_overallocation_and_geometry_overflow() -> None:
    with pytest.raises(ValueError, match="fin_count"):
        make_order_spec("A", fin_count=13)
    with pytest.raises(ValueError, match="does not fit"):
        make_order_spec("A", fin_count=12, fin_pitch=0.04)
    with pytest.raises(KeyError, match="unknown order preset"):
        make_order_spec("missing")


def test_order_stage_transition_contract_follows_precoat_then_fin_then_press() -> None:
    product = create_product_state(order_id="transition")

    expected_sequence = (
        OrderStage.BASE_LOADING,
        OrderStage.MATERIAL_APPLICATION,
        OrderStage.MATERIAL_INSPECTION,
        OrderStage.COMB_CONFIGURATION,
        OrderStage.FIN_ASSEMBLY,
        OrderStage.PRE_INSPECTION,
        OrderStage.FIXTURE_PRESSING,
        OrderStage.FIXTURE_LOCKING,
        OrderStage.READY_FOR_TRANSFER,
        OrderStage.FURNACE_LOADING,
        OrderStage.BRAZING,
        OrderStage.UNLOADING,
        OrderStage.POST_INSPECTION,
        OrderStage.PASS,
    )
    for stage in expected_sequence:
        product.transition(stage)
        assert product.stage is stage

    assert product.terminal


def test_order_stage_transition_rejects_installing_fins_before_precoat() -> None:
    product = create_product_state(order_id="invalid-order")
    product.transition(OrderStage.BASE_LOADING)
    with pytest.raises(StateTransitionError):
        product.transition(OrderStage.FIN_ASSEMBLY)
    assert product.stage is OrderStage.BASE_LOADING


def test_material_and_fin_inspection_rework_loops_return_to_their_own_operation() -> None:
    product = create_product_state(order_id="rework-transitions")
    product.transition(OrderStage.BASE_LOADING)
    product.transition(OrderStage.MATERIAL_APPLICATION)
    product.transition(OrderStage.MATERIAL_INSPECTION)
    product.transition(OrderStage.MATERIAL_APPLICATION)
    product.transition(OrderStage.MATERIAL_INSPECTION)
    product.transition(OrderStage.COMB_CONFIGURATION)
    product.transition(OrderStage.FIN_ASSEMBLY)
    product.transition(OrderStage.PRE_INSPECTION)
    product.transition(OrderStage.FIN_ASSEMBLY)
    product.transition(OrderStage.PRE_INSPECTION)
    product.transition(OrderStage.FIXTURE_PRESSING)

    assert product.stage is OrderStage.FIXTURE_PRESSING


def test_nonterminal_stage_can_still_enter_error_once() -> None:
    product = create_product_state(order_id="failure")
    product.transition(OrderStage.BASE_LOADING)
    product.fail("actor timeout", now=12.0)
    assert product.stage is OrderStage.ERROR
    assert product.completed_at == 12.0


def test_fixture_lock_requires_comb_material_fins_and_held_press_force() -> None:
    product = create_product_state(order_id="fixture-gates")
    fixture = product.fixture
    fixture.base_weld_active = True

    with pytest.raises(RuntimeError, match="comb modules"):
        fixture.lock()
    fixture.active_comb_module = product.spec.comb_module_name
    fixture.front_comb_module = product.spec.comb_module_name
    fixture.rear_comb_module = product.spec.comb_module_name
    fixture.comb_configured = True
    fixture.comb_aligned = True

    with pytest.raises(RuntimeError, match="material inspection"):
        fixture.lock()
    fixture.material_passed = True

    with pytest.raises(RuntimeError, match="fin geometry"):
        fixture.lock()
    fixture.fins_passed = True

    with pytest.raises(RuntimeError, match="target force"):
        fixture.lock()
    fixture.press_state = PressState.COMPLETE
    fixture.press_force_held = True
    fixture.lock()

    assert fixture.locked
    assert fixture.cycle_locked
    assert fixture.ready_for_transfer
    assert fixture.status is FixtureStatus.READY_FOR_TRANSFER


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
