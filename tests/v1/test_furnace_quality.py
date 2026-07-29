from __future__ import annotations

import pytest

from brazing_sim.config import create_product_state
from brazing_sim.domain import FurnacePhase, OrderStage, PressState, TerminalDisposition
from brazing_sim.furnace import DemoFurnace, FurnaceInterlockError
from brazing_sim.quality import QualityEvaluator


def _ready_product(fault: str | None = None):
    product = create_product_state(order_id=f"quality-{fault or 'normal'}")
    for fin in product.active_fins:
        fin.inserted = True
        fin.temporary_welded = True
    for path in product.active_paths:
        path.applied = True
        path.coverage_ratio = 1.0
    fixture = product.fixture
    fixture.base_weld_active = True
    fixture.active_comb_module = product.spec.comb_module_name
    fixture.front_comb_module = product.spec.comb_module_name
    fixture.rear_comb_module = product.spec.comb_module_name
    fixture.comb_configured = True
    fixture.comb_aligned = True
    fixture.material_passed = True
    fixture.fins_passed = True
    fixture.press_state = PressState.COMPLETE
    fixture.press_force_held = True
    fixture.lock()
    furnace = DemoFurnace(product.spec.recipe)
    furnace.start(0.0, fault=fault)
    furnace.update(100.0)
    product.furnace = furnace.snapshot()
    return product


def test_demo_furnace_advances_only_with_simulation_clock() -> None:
    furnace = DemoFurnace()
    assert furnace.recipe.process_seconds == pytest.approx(10.0)
    furnace.start(0.0)
    assert furnace.status is FurnacePhase.DOOR_OPENING
    assert not furnace.complete

    furnace.update(1.5)
    assert furnace.status is FurnacePhase.PREHEAT
    furnace.update(3.5)
    assert furnace.status is FurnacePhase.RAMP
    furnace.update(6.5)
    assert furnace.status is FurnacePhase.SOAK
    furnace.update(9.5)
    assert furnace.status is FurnacePhase.COOLING
    furnace.update(12.24)
    assert furnace.status is FurnacePhase.DOOR_OPENING
    furnace.update(12.25)
    assert furnace.complete
    assert furnace.temperature == pytest.approx(80.0)
    assert furnace.state.peak_temperature_c == pytest.approx(600.0)
    with pytest.raises(ValueError, match="monotonic"):
        furnace.update(12.0)


def test_manual_furnace_sequence_enforces_door_and_load_interlocks() -> None:
    furnace = DemoFurnace()
    with pytest.raises(FurnaceInterlockError, match="READY"):
        furnace.start_cycle(0.0)
    with pytest.raises(FurnaceInterlockError, match="fully open"):
        furnace.load_workpiece(0.0)

    furnace.request_open(0.0)
    furnace.update(0.75)
    assert furnace.status is FurnacePhase.LOADING
    furnace.load_workpiece(0.75)
    furnace.request_close(0.75)
    furnace.update(1.5)
    assert furnace.status is FurnacePhase.READY
    furnace.start_cycle(1.5)
    furnace.update(100.0)
    assert furnace.complete


def test_furnace_fault_profiles_are_deterministic() -> None:
    recoverable = DemoFurnace()
    recoverable.start(0.0, fault="recoverable")
    recoverable.update(100.0)
    assert recoverable.state.profile_score == pytest.approx(0.82)
    assert recoverable.state.peak_temperature_c == pytest.approx(560.0)
    assert not recoverable.state.severe_violation

    severe = DemoFurnace()
    severe.start(0.0, fault="severe")
    severe.update(100.0)
    assert severe.state.profile_score == pytest.approx(0.20)
    assert severe.state.peak_temperature_c == pytest.approx(780.0)
    assert severe.state.severe_violation


def test_preinspection_thresholds_and_rework_targets() -> None:
    product = _ready_product()
    evaluator = QualityEvaluator()
    assert evaluator.pre_inspection(product).passed

    product.active_fins[1].position_error_m = product.spec.inspection.fin_position_m + 1e-6
    result = evaluator.pre_inspection(product)
    assert not result.passed
    assert result.rework_targets == ("fin_02",)
    assert "fin_02.position" in result.hard_failures


def test_material_thresholds_are_inclusive_and_fault_targets_one_path() -> None:
    product = _ready_product()
    evaluator = QualityEvaluator()
    path = product.active_paths[2]
    path.coverage_ratio = product.spec.inspection.coverage_ratio
    path.longest_gap_m = product.spec.inspection.longest_material_gap_m
    path.lateral_error_m = product.spec.inspection.lateral_error_m
    path.trajectory_rmse_m = product.spec.inspection.trajectory_rmse_m
    path.trajectory_max_error_m = product.spec.inspection.trajectory_max_error_m
    assert evaluator.material_inspection(product).passed

    path.longest_gap_m += 1e-6
    result = evaluator.material_inspection(product)
    assert not result.passed
    assert result.rework_targets == (path.path_id,)
    assert f"{path.path_id}.gap" in result.hard_failures


@pytest.mark.parametrize(
    ("fault", "expected"),
    [
        (None, TerminalDisposition.PASS),
        ("recoverable", TerminalDisposition.REWORK_REQUIRED),
        ("severe", TerminalDisposition.SCRAPPED),
    ],
)
def test_postinspection_classifies_normal_recoverable_and_severe_profiles(
    fault: str | None, expected: TerminalDisposition
) -> None:
    product = _ready_product(fault)
    product.stage = OrderStage.POST_INSPECTION
    result = QualityEvaluator().post_inspection(product, now=20.0)
    assert result.disposition is expected
    assert product.disposition is expected
    assert product.stage.value == expected.value
    assert product.kpi.final_quality_score == result.score


def test_hard_material_gate_prevents_pass_even_if_weighted_score_is_high() -> None:
    product = _ready_product()
    product.active_paths[0].coverage_ratio = product.spec.inspection.coverage_ratio - 0.001
    result = QualityEvaluator().post_inspection(product)
    assert result.score is not None and result.score >= product.spec.inspection.pass_score
    assert result.disposition is TerminalDisposition.REWORK_REQUIRED


def test_low_weighted_score_scraps_without_a_severe_temperature_fault() -> None:
    product = _ready_product()
    for path in product.active_paths:
        path.applied = False
        path.coverage_ratio = 0.0
    product.fixture.cycle_locked = False
    result = QualityEvaluator().post_inspection(product)
    assert result.score is not None and result.score < product.spec.inspection.rework_score
    assert result.disposition is TerminalDisposition.SCRAPPED
