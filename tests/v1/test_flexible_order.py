from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
import subprocess
import sys

import pytest
import yaml

from brazing_sim.actors import build_scene_actors
from brazing_sim.batch import BatchCoordinator
from brazing_sim.domain import BatchStage, OrderStage, RackShelfState
from brazing_sim.flexible import (
    FlexibleConfigError,
    allocate_rack,
    build_preset_plan,
    build_process_plan,
    fin_y_positions,
    generate_geometry,
    load_fixture_modules,
    load_product,
    load_rack_config,
    validate_process_plan,
)
from brazing_sim.process import ProcessCoordinator
from brazing_sim.scene import BrazingScene

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "config"


@pytest.mark.parametrize(
    ("order_name", "preset", "fins", "paths", "pitch", "force", "length", "layers"),
    [
        ("order_001.yaml", "A", 5, 10, 0.020, 20.0, 0.33, [0, 1, 2]),
        ("order_002.yaml", "B", 4, 8, 0.030, 18.0, 0.33, [0, 1, 2]),
        ("order_003.yaml", "C", 7, 14, 0.015, 22.0, 0.31, [0, 1, 2]),
    ],
)
def test_example_orders_build_complete_process_plans(
    order_name: str,
    preset: str,
    fins: int,
    paths: int,
    pitch: float,
    force: float,
    length: float,
    layers: list[int],
) -> None:
    plan = build_process_plan(CONFIG / "orders" / order_name)
    validate_process_plan(plan)

    assert plan.product.preset == preset
    assert plan.quantity == 3
    assert len(plan.fin_targets) == fins
    assert len(plan.brazing_paths) == paths
    assert plan.execution_spec.fin_pitch == pytest.approx(pitch)
    assert plan.execution_spec.target_clamping_force_n == pytest.approx(force)
    assert all(path.length_m == pytest.approx(length) for path in plan.brazing_paths)
    assert [assignment.layer_index for assignment in plan.rack_assignments] == layers
    assert plan.max_fins == 12
    assert plan.max_paths == 24
    assert plan.path_segment_capacity == 20


@pytest.mark.parametrize(
    ("preset", "expected"),
    [
        ("A", (-0.040, -0.020, 0.0, 0.020, 0.040)),
        ("B", (-0.045, -0.015, 0.015, 0.045)),
        ("C", (-0.045, -0.030, -0.015, 0.0, 0.015, 0.030, 0.045)),
    ],
)
def test_odd_even_and_fifteen_millimetre_arrays_are_centered(
    preset: str, expected: tuple[float, ...]
) -> None:
    product = build_preset_plan(preset).product
    assert fin_y_positions(product) == pytest.approx(expected)

    shifted = replace(product, start_offset_y_m=-0.071)
    assert fin_y_positions(shifted) == pytest.approx(
        tuple(-0.071 + index * product.fin_pitch_m for index in range(product.fin_count))
    )


def test_geometry_rejects_overlap_boundary_and_pool_overflow() -> None:
    product = build_preset_plan("A").product
    with pytest.raises(ValueError, match="重叠"):
        generate_geometry(replace(product, fin_pitch_m=0.001))
    with pytest.raises(ValueError, match="Y边界"):
        generate_geometry(replace(product, start_offset_y_m=0.10))
    with pytest.raises(ValueError, match="对象池上限"):
        generate_geometry(replace(product, fin_count=13, fin_pitch_m=0.010))


def _write_yaml(path: Path, value: object) -> Path:
    path.write_text(yaml.safe_dump(value, allow_unicode=True, sort_keys=False), encoding="utf-8")
    return path


def test_product_loader_reports_file_field_and_reason(tmp_path: Path) -> None:
    source = CONFIG / "products" / "product_a.yaml"
    original = yaml.safe_load(source.read_text(encoding="utf-8"))

    unknown = dict(original, unexpected=True)
    with pytest.raises(FlexibleConfigError, match=r"bad_unknown.yaml.*unexpected.*未知字段"):
        load_product(_write_yaml(tmp_path / "bad_unknown.yaml", unknown))

    missing = dict(original)
    missing.pop("fin_count")
    with pytest.raises(FlexibleConfigError, match=r"bad_missing.yaml.*fin_count.*缺少"):
        load_product(_write_yaml(tmp_path / "bad_missing.yaml", missing))

    wrong_type = dict(original, fin_count="five")
    with pytest.raises(FlexibleConfigError, match=r"bad_type.yaml.*fin_count.*类型错误"):
        load_product(_write_yaml(tmp_path / "bad_type.yaml", wrong_type))

    negative = dict(original, material_speed_m_s=-0.1)
    with pytest.raises(FlexibleConfigError, match=r"bad_negative.yaml.*material_speed_m_s.*大于0"):
        load_product(_write_yaml(tmp_path / "bad_negative.yaml", negative))

    syntax = tmp_path / "bad_syntax.yaml"
    syntax.write_text("product_id: [unterminated\n", encoding="utf-8")
    with pytest.raises(FlexibleConfigError, match=r"bad_syntax.yaml:\d+:\d+.*YAML语法错误"):
        load_product(syntax)


def test_catalog_loader_reports_full_nested_field_path(tmp_path: Path) -> None:
    data = yaml.safe_load((CONFIG / "fixture_modules.yaml").read_text(encoding="utf-8"))
    data["modules"][0]["pitch_m"] = "20 mm"
    path = _write_yaml(tmp_path / "fixtures.yaml", data)
    with pytest.raises(FlexibleConfigError, match=r"modules\[0\]\.pitch_m.*类型错误"):
        load_fixture_modules(path)


def test_rack_allocator_honours_preference_and_skips_occupied_layers() -> None:
    base = build_preset_plan("A", quantity=1)
    rack = load_rack_config(CONFIG / "rack_config.yaml")
    preferred = replace(base.order, quantity=2, preferred_rack_layer=2)
    assignments = allocate_rack(preferred, rack)
    assert [item.layer_index for item in assignments] == [2, 0]

    lowest = replace(base.order, quantity=1, preferred_rack_layer=None)
    assert allocate_rack(lowest, rack, occupied_layers={0})[0].layer_index == 1
    with pytest.raises(ValueError, match="空层不足"):
        allocate_rack(replace(lowest, quantity=2), rack, occupied_layers={0, 1})


def test_dry_run_outputs_chinese_plan_without_starting_a_model() -> None:
    command = [
        sys.executable,
        str(ROOT / "run_flexible_order.py"),
        "--order",
        str(CONFIG / "orders" / "order_002.yaml"),
        "--dry-run",
    ]
    completed = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, timeout=20)
    assert completed.returncode == 0, completed.stderr
    assert "校验结果：通过" in completed.stdout
    payload = json.loads(completed.stdout[completed.stdout.find("{\n") :])
    assert payload["preset"] == "B"
    assert payload["fin_count"] == 4
    assert payload["path_count"] == 8


def test_flexible_mjcf_pool_comb_rack_welds_and_dynamic_c_nozzle() -> None:
    plan = build_preset_plan("C", quantity=1)
    scene = BrazingScene(
        ROOT / "scenes" / "production" / "brazing_line.xml", order=plan.execution_spec, raw=True
    )
    try:
        validate_process_plan(plan, scene)
        for index in range(1, 13):
            scene.model.body(f"fin_{index:02d}")
        for index in range(1, 13):
            for side in ("left", "right"):
                scene.model.body(f"slot_{index:02d}_{side}_brazing_path")
        scene.model.body("front_comb_insert_15mm")
        scene.model.body("rear_comb_insert_15mm")
        for tray in range(1, 4):
            for shelf in range(3):
                scene.model.equality(f"batch_rack_tray_{tray:02d}_shelf_{shelf}_weld")
        left = scene.model.site("arm2_left_nozzle_tip_site").pos
        right = scene.model.site("arm2_right_nozzle_tip_site").pos
        assert float(right[1] - left[1]) == pytest.approx(0.0044)
    finally:
        scene.close()


@pytest.mark.parametrize("quantity", [1, 2])
def test_partial_flexible_batches_braze_all_planned_layers_once(quantity: int) -> None:
    plan = build_preset_plan("B", quantity=quantity)
    scene = BrazingScene(
        ROOT / "scenes" / "production" / "brazing_line.xml", order=plan.execution_spec, raw=True
    )
    holder: dict[str, BatchCoordinator] = {}
    actors = build_scene_actors(scene, lambda: holder["batch"].product, fast=True)
    single = ProcessCoordinator(actors=actors, fast=True)
    batch_coordinator = BatchCoordinator(scene, single, fast=True)
    holder["batch"] = batch_coordinator
    try:
        batch = batch_coordinator.start_process_plan(plan, now=scene.time)
        deadline = scene.time + 40.0
        while scene.time < deadline and not batch_coordinator.terminal:
            batch_coordinator.tick(scene.time)
            scene.step()
        assert batch.stage is BatchStage.COMPLETE, batch_coordinator.snapshot(scene.time)
        assert [unit.product.stage for unit in batch.units] == [OrderStage.PASS] * quantity
        assigned = {item.layer_index for item in plan.rack_assignments}
        assert all(
            (
                shelf.state is RackShelfState.UNLOADED
                if shelf.index in assigned
                else shelf.state is RackShelfState.EMPTY
            )
            for shelf in batch.rack.shelves
        )
        assert batch.furnace.elapsed_seconds == pytest.approx(10.0)
    finally:
        scene.close()
