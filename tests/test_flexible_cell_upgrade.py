from __future__ import annotations

import math
from types import SimpleNamespace

import numpy as np
import pytest

from brazing_line import BrazingApplication, parse_args
from brazing_sim.async_line_router import AsyncLineProcessRouter
from brazing_sim.config import create_product_state
from brazing_sim.domain import OrderStage
from brazing_sim.execution.async_line_skills import AsyncLinePhysicalSkill
from brazing_sim.flexible import build_custom_plan, build_inline_plan
from brazing_sim.layout import SHALLOW_U_LAYOUT
from brazing_sim.manufacturing_runtime import ManufacturingRuntime
from brazing_sim.planning import (
    HybridMotionPlanner,
    MotionRequest,
    SpaceTimeReservationTable,
    build_task_graph,
)
from brazing_sim.planning.task_models import TaskType
from brazing_sim.planning.task_models import ManufacturingTask
from brazing_sim.scene import BrazingScene
from brazing_sim.workcells import validate_async_line_state


def _run_runtime(mode: str, quantity: int = 2) -> tuple[float, ManufacturingRuntime, float]:
    runtime = ManufacturingRuntime(scheduler_mode=mode, flexible_cell=True)
    runtime.submit_plan(
        build_inline_plan(
            preset="A",
            order_id=f"ASYNC_{mode}_{quantity}",
            quantity=quantity,
            priority=10,
        )
    )
    overlap = 0.0
    for index in range(12000):
        now = index * 0.25
        runtime.tick(now)
        active = list(runtime.executor.active.values())
        arm1_s1 = any(
            item.resource_id == "ARM1" and item.task.station_id == "S1_BASE_LOADING" for item in active
        )
        arm2_s2a = any(
            item.resource_id == "ARM2" and item.task.station_id == "S2A_DISPENSING" for item in active
        )
        if arm1_s1 and arm2_s2a:
            overlap += 0.25
        if runtime.terminal:
            return now, runtime, overlap
    raise AssertionError("asynchronous runtime did not terminate")


def test_each_unit_has_four_independent_transfers_and_no_turntable_task() -> None:
    plan = build_inline_plan(preset="A", order_id="ASYNC_GRAPH", quantity=2, priority=10)
    graph = build_task_graph(plan, flexible_cell=True)
    transfer_types = {
        TaskType.TRANSFER_S1_S2A,
        TaskType.TRANSFER_S2A_S2B,
        TaskType.TRANSFER_S2B_S3,
        TaskType.TRANSFER_S3_RACK,
    }
    transfers = [task for task in graph if task.task_type in transfer_types]
    assert len(transfers) == 8
    assert all(task.tray_id is not None for task in transfers)
    assert not any(task.task_type is TaskType.ROTATE_TABLE2 for task in graph)

    second_index = graph.get("ASYNC_GRAPH_U02_INDEX_TRAY")
    assert "ASYNC_GRAPH_U01_TRANSFER_S1_S2A" in second_index.predecessors
    second_dispense = graph.get("ASYNC_GRAPH_U02_DISPENSE")
    first_cross = graph.get("ASYNC_GRAPH_U01_TRANSFER_S2A_S2B")
    assert first_cross.task_id in graph.get("ASYNC_GRAPH_U02_TRANSFER_S1_S2A").predecessors
    assert "ASYNC_GRAPH_U02_TRANSFER_S1_S2A" in second_dispense.predecessors


def test_dynamic_pipeline_overlaps_s1_and_s2a_and_beats_fixed_sequence() -> None:
    _two_time, _runtime, overlap = _run_runtime("dynamic", quantity=2)
    assert overlap >= 1.0
    fixed_time, _, _ = _run_runtime("fixed", quantity=3)
    dynamic_time, runtime, _ = _run_runtime("dynamic", quantity=3)
    assert dynamic_time <= fixed_time * 0.90
    assert runtime.terminal


def test_runtime_allocates_three_unique_wip_trays_and_keeps_one_spare() -> None:
    runtime = ManufacturingRuntime(scheduler_mode="dynamic", flexible_cell=True)
    first = runtime.submit_plan(build_inline_plan(preset="A", order_id="WIP_TWO", quantity=2, priority=10))
    second = runtime.submit_plan(
        build_inline_plan(preset="B", order_id="WIP_ONE", quantity=1, priority=20),
        urgent=True,
    )
    assigned = [*first.tray_assignments.values(), *second.tray_assignments.values()]
    assert len(assigned) == len(set(assigned)) == 3
    snapshot = runtime.snapshot(0.0)
    assert snapshot["async_line"]["active_wip"] == 3
    assert len(snapshot["async_line"]["spare_trays"]) == 1
    validate_async_line_state(
        runtime.tray_routes.values(),
        runtime.workstations.values(),
        runtime.transfers.values(),
        wip_limit=3,
    )


def test_manual_review_keeps_its_wip_and_display_spare_is_never_assigned() -> None:
    runtime = ManufacturingRuntime(scheduler_mode="dynamic", flexible_cell=True)
    held = runtime.submit_plan(build_inline_plan(preset="A", order_id="HELD_REVIEW", quantity=2, priority=10))
    held.status = held.status.__class__.MANUAL_REVIEW
    admitted = runtime.submit_plan(
        build_inline_plan(preset="B", order_id="AFTER_REVIEW", quantity=1, priority=10)
    )
    queued = runtime.submit_plan(
        build_inline_plan(preset="C", order_id="STILL_QUEUED", quantity=1, priority=10)
    )

    assert set(admitted.tray_assignments.values()) == {"tray_03"}
    assert queued.status.value == "QUEUED"
    assert "tray_04" not in {
        tray_id for entry in runtime.orders.values() for tray_id in entry.tray_assignments.values()
    }


def test_runtime_and_physical_scene_share_the_authoritative_station_layout() -> None:
    runtime = ManufacturingRuntime(scheduler_mode="dynamic", flexible_cell=True)
    expected = {
        "S1_BASE_LOADING": SHALLOW_U_LAYOUT.station_s1_xy,
        "S2A_DISPENSING": SHALLOW_U_LAYOUT.station_s2a_xy,
        "S2B_MATERIAL_INSPECTION": SHALLOW_U_LAYOUT.station_s2b_xy,
        "S3_FIN_ASSEMBLY": SHALLOW_U_LAYOUT.station_s3_xy,
        "RACK_INFEED": SHALLOW_U_LAYOUT.rack_infeed_xy,
    }
    assert {
        station_id.value: station.world_xy for station_id, station in runtime.workstations.items()
    } == expected


def test_high_reliability_and_first_article_add_station_specific_checks() -> None:
    high = build_task_graph(
        build_inline_plan(
            preset="A",
            order_id="HIGH",
            quantity=2,
            priority=10,
            route_strategy="HIGH_RELIABILITY",
        ),
        flexible_cell=True,
    )
    first = build_task_graph(
        build_inline_plan(
            preset="A",
            order_id="FIRST",
            quantity=2,
            priority=10,
            route_strategy="FIRST_ARTICLE",
        ),
        flexible_cell=True,
    )
    special = {TaskType.VERIFY_BASE_ALIGNMENT, TaskType.SECOND_POST_BRAZE_VIEW}
    assert sum(task.task_type in special for task in high) == 4
    assert sum(task.task_type in special for task in first) == 2
    assert all(
        task.station_id == "S1_BASE_LOADING"
        for task in high
        if task.task_type is TaskType.VERIFY_BASE_ALIGNMENT
    )


def test_custom_order_rejects_a_pitch_without_a_physical_module() -> None:
    product = {
        "base_size_m": [0.36, 0.22, 0.008],
        "fin_size_m": [0.30, 0.002, 0.06],
        "fin_count": 5,
        "fin_pitch_m": 0.025,
        "path_margin_m": 0.015,
        "path_width_m": 0.004,
        "nozzle_spacing_m": 0.005,
        "nozzle_tip_height_m": 0.004,
        "material_speed_m_s": 0.04,
        "target_clamping_force_n": 20.0,
        "recipe": "demo_brazing",
    }
    with pytest.raises(ValueError, match="实体梳齿模块"):
        build_custom_plan(order_id="BAD_PITCH", quantity=1, priority=10, product=product)


def test_prm_and_sipp_reserve_at_twenty_millisecond_resolution() -> None:
    planner = HybridMotionPlanner(
        [(-1.0, 1.0), (-1.0, 1.0)],
        lambda _q: True,
        roadmap_samples=24,
        neighbours=5,
        seed=7,
    )
    first = planner.plan(MotionRequest("p1", "ARM1", (0.0, 0.0), ((0.8, 0.8),), 0.0))
    second = planner.plan(MotionRequest("p2", "ARM2", (0.0, 0.0), ((0.8, 0.8),), 0.0))
    table = SpaceTimeReservationTable(0.02)

    def occupancy(_q: np.ndarray) -> tuple[str, ...]:
        return ("S3_SHARED_SWEEP",)

    table.reserve(first, occupancy, sample_period_s=0.02, safe_wait=lambda _q: True)
    table.reserve(second, occupancy, sample_period_s=0.02, safe_wait=lambda _q: True)
    assert second.waiting_time > 0.0
    assert second.start_time >= first.end_time


def test_mjcf_async_line_has_one_live_tray_owner() -> None:
    scene = BrazingScene(raw=True)
    try:
        registry = scene.registry
        registry.reset_flexible_cell()
        snapshot = registry.async_line_snapshot()
        assert snapshot["layout"] == "SHALLOW_U"
        assert snapshot["station_owner"] == "S1"
        active_station_welds = [
            station
            for station in ("s1", "s2a", "s2b", "s3", "rack_infeed")
            if bool(scene.data.eq_active[registry.equality_id(f"station_{station}_assembly_tray_weld")])
        ]
        assert active_station_welds == ["s1"]
        assert all(value == pytest.approx(0.0) for value in snapshot["transfer_positions_m"].values())
    finally:
        scene.close()


def test_mjcf_three_wip_trays_can_occupy_independent_stations() -> None:
    scene = BrazingScene(raw=True)
    try:
        registry = scene.registry
        registry.reset_batch_cell()
        registry.dock_batch_tray_to_async_station("tray_01", "s3", snap=True)
        registry.dock_batch_tray_to_async_station("tray_02", "s2a", snap=True)
        registry.dock_batch_tray_to_async_station("tray_03", "s1", snap=True)
        snapshot = registry.async_line_snapshot()
        assert snapshot["physical_tray_owners"] == {
            "tray_01": "S3",
            "tray_02": "S2A",
            "tray_03": "S1",
        }
        for unit in range(1, 4):
            ownership = [
                name
                for name in (
                    *(
                        f"station_{station}_tray_{unit:02d}_weld"
                        for station in ("s1", "s2a", "s2b", "s3", "rack_infeed")
                    ),
                    *(
                        f"transfer_{transfer}_tray_{unit:02d}_weld"
                        for transfer in ("s1_s2a", "s2a_s2b", "s2b_s3", "s3_rack")
                    ),
                )
                if bool(scene.data.eq_active[registry.equality_id(name)])
            ]
            assert len(ownership) == 1
    finally:
        scene.close()


def test_brazing_paths_are_reseated_in_the_current_pallet_frame() -> None:
    product = create_product_state(order_id="PATH_FRAME")
    scene = BrazingScene(order=product, raw=True)
    try:
        registry = scene.registry
        registry.dock_assembly_tray_to_station("s2a", snap=True)
        registry.place_base_on_tray(snap=True)
        scene.step(2)
        for path in product.active_paths:
            midpoint = 0.5 * (
                np.asarray(path.local_start, dtype=float) + np.asarray(path.local_end, dtype=float)
            )
            expected = np.asarray(registry.product_to_world(midpoint), dtype=float)
            actual = registry.free_body_pose(path.name).position
            assert actual == pytest.approx(expected, abs=0.001)

        # Re-docking the same rigid pallet must carry the already attached
        # base and bead bodies together, without retaining an S2A offset.
        registry.dock_assembly_tray_to_station("s2b", snap=True)
        scene.step(20)
        first = product.active_paths[0]
        midpoint = 0.5 * (
            np.asarray(first.local_start, dtype=float) + np.asarray(first.local_end, dtype=float)
        )
        assert registry.free_body_pose(first.name).position == pytest.approx(
            registry.product_to_world(midpoint),
            abs=0.002,
        )
    finally:
        scene.close()


def test_batch_beads_finish_with_identical_longitudinal_endpoints() -> None:
    plan = build_inline_plan(preset="C", order_id="ALIGNED_BEADS", quantity=1, priority=10)
    product = create_product_state(plan.execution_spec, order_id="ALIGNED_BEADS_UNIT_01")
    scene = BrazingScene(order=plan.execution_spec, raw=True)
    try:
        registry = scene.registry
        registry.configure_batch_tray(0, product)
        # Reproduce the small, direction-dependent TCP residual that used to
        # remain visible after alternating serpentine passes.
        for index in range(len(plan.brazing_paths)):
            registry.set_batch_brazing_path_progress(
                0,
                index,
                0.985 - 0.001 * (index % 3),
                reverse=bool(index // 2 % 2),
            )
        for index in range(len(plan.brazing_paths)):
            registry.set_batch_brazing_path_progress(0, index, 1.0, reverse=False)

        expected_half_length = plan.execution_spec.base_length / 2.0 - plan.execution_spec.path_margin
        for index in range(len(plan.brazing_paths)):
            geom = scene.model.geom(f"batch_tray_01_braze_{index + 1:02d}")
            assert float(geom.pos[0]) == pytest.approx(0.0, abs=1.0e-12)
            assert float(geom.size[1]) == pytest.approx(expected_half_length, abs=1.0e-12)
    finally:
        scene.close()


def test_batch_comb_physically_slides_in_and_slots_match_fin_targets() -> None:
    plan = build_inline_plan(preset="C", order_id="VISIBLE_COMB", quantity=1, priority=10)
    product = create_product_state(plan.execution_spec, order_id="VISIBLE_COMB_UNIT_01")
    scene = BrazingScene(order=plan.execution_spec, raw=True)
    try:
        registry = scene.registry
        registry.configure_batch_tray(0, product)
        support_x = plan.execution_spec.fin_length / 2.0 + 0.008 + 0.012

        registry.set_batch_comb_install_progress(0, 0.5)
        assert float(scene.model.geom("batch_tray_01_front_comb_base").pos[0]) < -support_x
        assert float(scene.model.geom("batch_tray_01_front_comb_base").rgba[3]) > 0.0

        registry.set_batch_comb_install_progress(0, 1.0)
        assert float(scene.model.geom("batch_tray_01_front_comb_base").pos[0]) == pytest.approx(-support_x)
        assert float(scene.model.geom("batch_tray_01_rear_comb_base").pos[0]) == pytest.approx(support_x)
        left_guides = [
            float(scene.model.geom(f"batch_tray_01_front_comb_guide_left{index:02d}").pos[1])
            for index in range(plan.execution_spec.fin_count)
        ]
        right_guides = [
            float(scene.model.geom(f"batch_tray_01_front_comb_guide_right{index:02d}").pos[1])
            for index in range(plan.execution_spec.fin_count)
        ]
        slot_centres = [0.5 * (left + right) for left, right in zip(left_guides, right_guides)]
        assert slot_centres == pytest.approx(
            [target.target_position[1] for target in product.active_fins],
            abs=1.0e-12,
        )
        guide = scene.model.geom("batch_tray_01_front_comb_guide_left00")
        assert tuple(float(value) for value in guide.size[:3]) == pytest.approx((0.045, 0.001, 0.014))
        base_top = 0.032 + plan.execution_spec.base_thickness / 2.0
        assert float(guide.pos[2] - guide.size[2]) - base_top >= 0.020
        rail = scene.model.geom("batch_tray_01_front_comb_base")
        post = scene.model.geom("batch_tray_01_front_comb_post_left")
        rear_rail = scene.model.geom("batch_tray_01_rear_comb_base")
        rear_guide = scene.model.geom("batch_tray_01_rear_comb_guide_left00")
        fin_half_length = plan.execution_spec.fin_length / 2.0
        base_half_width = plan.execution_spec.base_width / 2.0
        # The load-bearing portal sits wholly beyond both the fin end face and
        # the base side edge.  The fingers attach to its inner face and extend
        # toward the product centre.
        assert float(-rail.pos[0] - rail.size[0]) > fin_half_length
        assert float(rear_rail.pos[0] - rear_rail.size[0]) > fin_half_length
        assert float(abs(post.pos[1]) - post.size[1]) > base_half_width
        assert float(guide.pos[0]) > float(rail.pos[0])
        assert float(rear_guide.pos[0]) < float(rear_rail.pos[0])
        assert float(guide.pos[0] - guide.size[0]) == pytest.approx(
            float(rail.pos[0] + rail.size[0]),
            abs=1.0e-12,
        )
        assert float(rear_guide.pos[0] + rear_guide.size[0]) == pytest.approx(
            float(rear_rail.pos[0] - rear_rail.size[0]),
            abs=1.0e-12,
        )
        assert float(guide.pos[2] - guide.size[2]) == pytest.approx(
            float(rail.pos[2] + rail.size[2]),
            abs=1.0e-12,
        )
        assert float(post.pos[2] + post.size[2]) >= float(rail.pos[2] - rail.size[2])
        assert all(right - left == pytest.approx(0.006) for left, right in zip(left_guides, right_guides))

        registry.set_batch_comb_install_progress(0, 0.0)
        registry.set_batch_tray_visible(0, carrier=True, payload=True)
        for name in (
            "batch_tray_01_front_comb_base",
            "batch_tray_01_front_comb_post_left",
            "batch_tray_01_front_comb_guide_left00",
        ):
            assert float(scene.model.geom(name).rgba[3]) == 0.0
    finally:
        scene.close()


def test_finished_comb_and_press_removal_precede_every_output_route() -> None:
    graph = build_task_graph(
        build_inline_plan(preset="A", order_id="REMOVE_BEFORE_OUTPUT", quantity=1, priority=10),
        flexible_cell=True,
    )
    removal = next(
        task
        for task in graph
        if task.task_type is TaskType.REMOVE_OLD_COMB and task.payload.get("after_brazing")
    )
    assert graph.get(next(iter(removal.predecessors))).task_type is TaskType.POST_BRAZE_INSPECTION
    press_removal = next(
        task
        for task in graph
        if task.task_type is TaskType.REMOVE_OLD_PRESS and task.payload.get("after_brazing")
    )
    assert press_removal.predecessors == [removal.task_id]
    for route_type in (TaskType.ROUTE_PASS, TaskType.ROUTE_REWORK, TaskType.ROUTE_SCRAP):
        route = next(task for task in graph if task.task_type is route_type)
        assert press_removal.task_id in route.predecessors


def test_two_finished_press_bars_withdraw_together_before_becoming_hidden() -> None:
    from brazing_sim.config import create_product_state, make_order_spec
    from brazing_sim.scene import BrazingScene

    scene = BrazingScene(order="A", raw=True)
    try:
        registry = scene.registry
        registry.configure_batch_tray(
            0,
            create_product_state(
                make_order_spec("A"),
                order_id="PRESS_REMOVE",
                created_at=0.0,
            ),
        )
        registry.set_batch_press_progress(0, 1.0)
        front = scene.model.geom("batch_tray_01_front_press")
        rear = scene.model.geom("batch_tray_01_rear_press")
        front_start = front.pos.copy()
        rear_start = rear.pos.copy()

        registry.set_batch_press_removal_progress(0, 0.5)
        assert float(front.rgba[3]) == pytest.approx(1.0)
        assert float(rear.rgba[3]) == pytest.approx(1.0)
        assert float(front.pos[0]) < float(front_start[0])
        assert float(rear.pos[0]) > float(rear_start[0])
        assert float(front.pos[2]) > float(front_start[2])
        assert float(rear.pos[2]) > float(rear_start[2])

        registry.set_batch_press_removal_progress(0, 1.0)
        assert float(front.rgba[3]) == pytest.approx(0.0)
        assert float(rear.rgba[3]) == pytest.approx(0.0)
    finally:
        scene.close()


def test_shallow_u_transfer_axes_end_at_their_named_station() -> None:
    from brazing_sim.scene import BrazingScene

    scene = BrazingScene(order="A", raw=True)
    try:
        destinations = {
            "s1_s2a": "station_s2a_anchor",
            "s2a_s2b": "station_s2b_anchor",
            "s2b_s3": "station_s3_anchor",
            "s3_rack": "station_rack_infeed_anchor",
        }
        for transfer, destination in destinations.items():
            scene.registry.set_async_transfer_target(
                transfer,
                scene.registry.async_transfer_limit(transfer),
                teleport=True,
            )
            carriage = scene.registry.site_pose(f"transfer_{transfer}_site").position
            target = np.asarray(scene.data.body(destination).xpos, dtype=float)
            np.testing.assert_allclose(carriage, target, atol=2.0e-6)
            scene.registry.set_async_transfer_target(transfer, 0.0, teleport=True)
    finally:
        scene.close()


def test_physical_skill_start_failure_does_not_poison_later_retry() -> None:
    skill = AsyncLinePhysicalSkill(TaskType.PLACE_BASE_PLATE)
    task = ManufacturingTask(
        task_id="TX_PLACE",
        task_type=TaskType.PLACE_BASE_PLATE,
        order_id="ORDER_X",
        unit_id="ORDER_X_UNIT_01",
        tray_id="tray_01",
        station_id="S1_BASE_LOADING",
        eligible_resources=["ARM1"],
    )
    context = SimpleNamespace(scene=object(), manufacturing_runtime=object(), args=None)
    for _ in range(2):
        with pytest.raises(AttributeError):
            skill.start(task, "ARM1", context, 0.0)
        assert skill.task is None


def test_ui_order_ik_is_planned_incrementally_instead_of_blocking_start(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    args = parse_args(
        [
            "--headless",
            "--fast",
            "--no-ui",
            "--no-terminal-commands",
            "--port",
            "0",
        ]
    )
    application = BrazingApplication(args)
    try:
        controller = application.scene.arms["arm1"]
        original_solve_ik = controller.solve_ik
        solve_calls = 0

        def counted_solve_ik(*args, **kwargs):
            nonlocal solve_calls
            solve_calls += 1
            return original_solve_ik(*args, **kwargs)

        monkeypatch.setattr(controller, "solve_ik", counted_solve_ik)
        application.process_command(
            {
                "type": "order_insert",
                "mode": "preset",
                "preset": "A",
                "order_id": "INCREMENTAL_IK",
                "quantity": 1,
                "priority": 10,
                "route_strategy": "STANDARD",
                "urgent": False,
            }
        )
        assert solve_calls == 0

        for _ in range(2_000):
            previous_calls = solve_calls
            application.tick()
            assert solve_calls - previous_calls <= 1
            if solve_calls:
                break
        assert solve_calls == 1
    finally:
        application.close()


def test_ui_inserted_orders_use_the_physical_parallel_pipeline() -> None:
    args = parse_args(
        [
            "--headless",
            "--fast",
            "--no-ui",
            "--no-terminal-commands",
            "--port",
            "0",
        ]
    )
    application = BrazingApplication(args)
    try:
        for order_index, preset in enumerate(("A", "B", "C"), start=1):
            application.process_command(
                {
                    "type": "order_insert",
                    "mode": "preset",
                    "preset": preset,
                    "order_id": f"UI_PARALLEL_{order_index}",
                    "quantity": 1,
                    "priority": 10 + order_index,
                    "route_strategy": "STANDARD",
                    "urgent": order_index == 3,
                }
            )
        assert application.v2_pipeline_active
        assert not application.manufacturing_runtime.paused
        assert not application.batch_active()
        registry = application.scene.registry
        assert application.scene.model.geom_rgba[
            registry.geom_id("heatsink_base_plate_geom"), 3
        ] == pytest.approx(1.0)
        assert application.scene.model.geom_rgba[registry.geom_id("fin_01_geom"), 3] == pytest.approx(1.0)
        assert application.scene.model.geom_contype[registry.geom_id("heatsink_base_plate_geom")] == 0

        observed_arm_overlap = False
        observed_base_carry = False
        observed_fin_carry = False
        observed_base_contact = False
        observed_fin_contact = False
        observed_dispenser_contact = False
        observed_arm2_tool_mounted = False
        observed_partial_bead = False
        published_parallel_state = None
        previous_base_carry = False
        previous_fin_carry = False
        for index in range(30_000):
            application.tick()
            if index % 10 == 0:
                application.service_simulation_frame()
            active_arms = {
                execution.resource_id
                for execution in application.manufacturing_runtime.executor.active.values()
                if execution.resource_id in {"ARM1", "ARM2", "ARM3"}
            }
            observed_arm_overlap |= len(active_arms) >= 2
            base_carry = bool(application.scene.data.eq_active[registry.equality_id("arm1_grasp_base")])
            fin_carries = [
                bool(
                    application.scene.data.eq_active[registry.equality_id(f"arm1_grasp_fin_{fin_index:02d}")]
                )
                for fin_index in range(1, 8)
            ]
            fin_carry = any(fin_carries)
            observed_base_carry |= base_carry
            observed_fin_carry |= fin_carry
            if base_carry and not previous_base_carry:
                plan = application.manufacturing_runtime.orders[
                    next(
                        execution.task.order_id
                        for execution in application.manufacturing_runtime.executor.active.values()
                        if execution.resource_id == "ARM1"
                    )
                ].plan
                tcp = application.scene.arms["arm1"].current_tcp_pose().position
                base = registry.free_body_pose("base_plate").position.copy()
                base[2] += 0.5 * plan.execution_spec.base_thickness + 0.0015
                observed_base_contact |= float(np.linalg.norm(tcp - base)) <= 0.012
            if fin_carry and not previous_fin_carry:
                fin_index = fin_carries.index(True) + 1
                tcp = application.scene.arms["arm1"].current_tcp_pose().position
                fin = registry.free_body_pose(f"fin_{fin_index:02d}").position
                observed_fin_contact |= float(np.linalg.norm(tcp - fin)) <= 0.003
            previous_base_carry = base_carry
            previous_fin_carry = fin_carry
            for execution in application.manufacturing_runtime.executor.active.values():
                if execution.task.task_type is not TaskType.DISPENSE_BRAZING:
                    continue
                skill = execution.skill
                observed_arm2_tool_mounted |= application.scene.tools.current_tool == "brazing_dispenser"
                if not skill.arm_stages or skill.arm_stage_index >= len(skill.arm_stages):
                    continue
                stage = skill.arm_stages[skill.arm_stage_index]
                if stage.path_index is None:
                    continue
                tcp = application.scene.arms["arm2"].current_tcp_pose().position
                observed_dispenser_contact |= abs(float(tcp[2] - stage.target.position[2])) <= 0.012
                geom_id = registry.geom_id(
                    f"batch_tray_{int(execution.task.tray_id[-2:]):02d}" f"_braze_{stage.path_index + 1:02d}"
                )
                full_half_length = (
                    application.manufacturing_runtime.orders[execution.task.order_id]
                    .plan.brazing_paths[stage.path_index]
                    .length_m
                    / 2.0
                )
                current_half_length = float(application.scene.model.geom_size[geom_id, 1])
                observed_partial_bead |= 1.0e-4 < current_half_length < full_half_length - 1.0e-4
            if len(active_arms) >= 2 and published_parallel_state is None:
                application.publish(False)
                published_parallel_state = application.shared.snapshot()
            if application.manufacturing_runtime.terminal:
                break
        else:
            raise AssertionError("UI parallel orders did not terminate")

        assert observed_arm_overlap
        assert observed_base_carry
        assert observed_fin_carry
        assert observed_base_contact
        assert observed_fin_contact
        assert observed_arm2_tool_mounted
        assert observed_dispenser_contact
        assert observed_partial_bead
        assert published_parallel_state is not None
        assert published_parallel_state["stage"] == "PARALLEL_PRODUCTION"
        assert sum(arm["status"] == "busy" for arm in published_parallel_state["arms"].values()) >= 2
        presentation = application._build_v2_presentation(
            application.coordinator.snapshot(application.scene.time),
            None,
        )
        assert {order["status"] for order in presentation["orders"]} == {"COMPLETED"}
        assert all(order["progress"] == pytest.approx(1.0) for order in presentation["orders"])
        parallelism = presentation["async_line"]["parallelism"]
        assert parallelism["max_parallel_arms"] >= 2
        assert parallelism["multi_arm_overlap_s"] > 0.0
        assert presentation["async_line"]["process_router"]["mode"] == "MULTI_PALLET_RUNTIME"

        application.process_command(
            {
                "type": "order_insert",
                "mode": "preset",
                "preset": "A",
                "order_id": "UI_AFTER_COMPLETE",
                "quantity": 1,
                "priority": 10,
                "route_strategy": "STANDARD",
                "urgent": False,
            }
        )
        assert not application.manufacturing_runtime.terminal
        assert application.scene.model.geom_rgba[
            registry.geom_id("heatsink_base_plate_geom"), 3
        ] == pytest.approx(1.0)
        assert application.scene.model.geom_rgba[registry.geom_id("fin_01_geom"), 3] == pytest.approx(1.0)
    finally:
        application.close()


def test_two_ui_orders_physically_overlap_all_three_arms_without_contact() -> None:
    args = parse_args(
        [
            "--headless",
            "--dt",
            "0.02",
            "--no-ui",
            "--no-terminal-commands",
            "--port",
            "0",
        ]
    )
    application = BrazingApplication(args)
    try:
        for order_index, preset in enumerate(("A", "B"), start=1):
            application.process_command(
                {
                    "type": "order_insert",
                    "mode": "preset",
                    "preset": preset,
                    "order_id": f"UI_TWO_ORDER_{order_index}",
                    "quantity": 1,
                    "priority": 10 + order_index,
                    "route_strategy": "STANDARD",
                    "urgent": False,
                }
            )

        triple_overlap_s = 0.0
        unexpected_contacts = []
        previous_time = application.scene.time
        for _index in range(20_000):
            application.tick()
            now = application.scene.time
            active_arms = {
                execution.resource_id
                for execution in application.manufacturing_runtime.executor.active.values()
                if execution.resource_id in {"ARM1", "ARM2", "ARM3"}
            }
            if active_arms == {"ARM1", "ARM2", "ARM3"}:
                triple_overlap_s += now - previous_time
            previous_time = now
            unexpected_contacts.extend(
                contact
                for contact in application.safety.unexpected(application.scene.data)
                if not application.expected_task_contact(contact)
            )
            if application.manufacturing_runtime.terminal:
                break
        else:
            raise AssertionError("two-order physical pipeline did not terminate")

        snapshot = application.manufacturing_runtime.snapshot(application.scene.time)
        assert snapshot["async_line"]["parallelism"]["max_parallel_arms"] == 3
        assert triple_overlap_s >= 1.0
        assert unexpected_contacts == []
        assert not [
            task
            for task in snapshot["tasks"]
            if task["task_type"] == TaskType.INSTALL_FIN.value and task["status"] == "FAILED"
        ]
        assert snapshot["recoveries"] == []
        assert {entry.status.value for entry in application.manufacturing_runtime.orders.values()} == {
            "COMPLETED"
        }
    finally:
        application.close()


def test_material_and_fins_accumulate_without_cross_task_redraw() -> None:
    """Completed beads/fins remain visible while the next item is produced."""

    args = parse_args(
        [
            "--headless",
            "--dt",
            "0.02",
            "--no-ui",
            "--no-terminal-commands",
            "--port",
            "0",
        ]
    )
    application = BrazingApplication(args)
    previous_paths = np.zeros(10, dtype=float)
    previous_fins = np.zeros(5, dtype=float)
    try:
        application.process_command(
            {
                "type": "order_insert",
                "mode": "preset",
                "preset": "A",
                "order_id": "MONOTONIC_VISUAL_A",
                "quantity": 1,
                "priority": 10,
                "route_strategy": "STANDARD",
                "urgent": False,
            }
        )
        for _ in range(12_000):
            application.tick()
            paths = np.asarray(
                [
                    application.scene.model.geom_rgba[
                        application.scene.registry.geom_id(f"batch_tray_01_braze_{index:02d}"),
                        3,
                    ]
                    for index in range(1, 11)
                ],
                dtype=float,
            )
            fins = np.asarray(
                [
                    application.scene.model.geom_rgba[
                        application.scene.registry.geom_id(f"batch_tray_01_fin_{index:02d}"),
                        3,
                    ]
                    for index in range(1, 6)
                ],
                dtype=float,
            )
            assert np.all(paths + 1.0e-12 >= previous_paths)
            assert np.all(fins + 1.0e-12 >= previous_fins)
            previous_paths = paths
            previous_fins = fins
            if np.count_nonzero(paths > 0.5) == 10 and np.count_nonzero(fins > 0.5) == 5:
                break
        else:
            raise AssertionError("A型焊料或翅片没有全部逐项提交")

        assert np.all(previous_paths > 0.5)
        assert np.all(previous_fins > 0.5)
        assert not [
            task
            for task in application.manufacturing_runtime.graph
            if task.task_type is TaskType.INSTALL_FIN and task.status.value == "FAILED"
        ]
        assert application.manufacturing_runtime.recovery.plans == {}
    finally:
        application.close()


def test_normal_order_physically_continues_from_last_fin_to_finished_output() -> None:
    """Every post-fin DAG milestone must change the actual MuJoCo scene."""

    args = parse_args(
        [
            "--headless",
            "--dt",
            "0.02",
            "--no-ui",
            "--no-terminal-commands",
            "--port",
            "0",
        ]
    )
    application = BrazingApplication(args)
    max_outfeed = 0.0
    max_output = 0.0
    max_gate = 0.0
    max_door = 0.0
    press_heights: list[float] = []
    try:
        application.process_command(
            {
                "type": "order_insert",
                "mode": "preset",
                "preset": "A",
                "order_id": "POST_FIN_PHYSICAL_A",
                "quantity": 1,
                "priority": 10,
                "route_strategy": "STANDARD",
                "urgent": False,
            }
        )
        registry = application.scene.registry
        for _ in range(16_000):
            application.tick()
            max_outfeed = max(
                max_outfeed,
                registry.batch_joint_position("batch_outfeed_joint"),
            )
            max_output = max(
                max_output,
                registry.batch_joint_position("batch_output_joint"),
            )
            max_gate = max(max_gate, registry.finished_output_gate_fraction)
            max_door = max(
                max_door,
                registry.batch_joint_position("furnace_door_joint") / 0.64,
            )
            press_heights.append(
                float(application.scene.model.geom_pos[registry.geom_id("batch_tray_01_front_press"), 2])
            )
            if application.manufacturing_runtime.terminal:
                break
        else:
            raise AssertionError("normal order did not reach the finished-goods outlet")

        assert max(press_heights) - min(press_heights) >= 0.055
        assert max_door >= 0.98
        assert max_outfeed >= 0.83
        assert max_output >= 1.10
        assert max_gate >= 0.98
        assert application.scene.model.geom_rgba[registry.geom_id("batch_tray_01_geom"), 3] == pytest.approx(
            0.0
        )
        assert application.scene.model.geom_rgba[registry.geom_id("batch_tray_01_base"), 3] == pytest.approx(
            0.0
        )
        required = {
            TaskType.INSPECT_FINS,
            TaskType.APPLY_PRESS,
            TaskType.LOCK_FIXTURE,
            TaskType.TRANSFER_S3_RACK,
            TaskType.MOVE_ELEVATOR,
            TaskType.LOAD_RACK_LAYER,
            TaskType.LOCK_RACK_LAYER,
            TaskType.RUN_FURNACE,
            TaskType.UNLOAD_RACK_LAYER,
            TaskType.POST_BRAZE_INSPECTION,
            TaskType.REMOVE_OLD_COMB,
            TaskType.REMOVE_OLD_PRESS,
            TaskType.ROUTE_PASS,
        }
        assert all(
            task.status.value == "SUCCEEDED"
            for task in application.manufacturing_runtime.graph
            if task.task_type in required
        )
        assert not [task for task in application.manufacturing_runtime.graph if task.status.value == "FAILED"]
    finally:
        application.close()


@pytest.mark.parametrize(("preset", "material_pass_count"), [("A", 5), ("B", 4), ("C", 7)])
def test_arm1_installation_descents_lock_xy_and_attitude_before_moving_z(
    preset: str,
    material_pass_count: int,
) -> None:
    """The visible payload may translate only vertically after high alignment."""

    args = parse_args(
        [
            "--headless",
            "--dt",
            "0.01",
            "--no-ui",
            "--no-terminal-commands",
            "--port",
            "0",
        ]
    )
    application = BrazingApplication(args)
    payload_samples: dict[str, list[tuple[np.ndarray, np.ndarray]]] = {
        "base": [],
        "fin": [],
    }
    alignment_errors: dict[str, list[tuple[float, float]]] = {"base": [], "fin": []}
    joint_samples: dict[str, list[np.ndarray]] = {"arm1": [], "arm2": [], "arm3": []}
    first_fin_task: str | None = None
    saw_fin_descent = False
    material_pass_counts: set[int] = set()
    try:
        application.process_command(
            {
                "type": "order_insert",
                "mode": "preset",
                "preset": preset,
                "order_id": f"STRICT_VERTICAL_{preset}",
                "quantity": 1,
                "priority": 10,
                "route_strategy": "STANDARD",
                "urgent": False,
            }
        )
        for _ in range(15_000):
            application.tick()
            for arm_name, samples in joint_samples.items():
                controller = application.scene.arms[arm_name]
                samples.append(
                    np.asarray(application.scene.data.qpos[controller.qpos_ids], dtype=float).copy()
                )
            fin_descent_active = False
            for execution in application.manufacturing_runtime.executor.active.values():
                skill = execution.skill
                stages = getattr(skill, "arm_stages", ())
                if execution.task.task_type is TaskType.DISPENSE_BRAZING and stages:
                    material_pass_counts.add(sum(stage.path_index is not None for stage in stages))
                if not stages or skill.arm_stage_index >= len(stages):
                    continue
                stage = stages[skill.arm_stage_index]
                if "纯Z下降" not in stage.label:
                    continue
                if getattr(skill, "arm_stage_entry_pending", False):
                    # The high endpoint is still stabilising; Z travel has
                    # deliberately not started yet.
                    continue
                if execution.task.task_type is TaskType.PLACE_BASE_PLATE:
                    payload = application.scene.registry.free_body_pose("base_plate")
                    payload_name = "base"
                    local_target = np.asarray([0.0, 0.0, 0.032])
                elif execution.task.task_type is TaskType.INSTALL_FIN:
                    if first_fin_task is None:
                        first_fin_task = execution.task.task_id
                    if execution.task.task_id != first_fin_task:
                        continue
                    fin_descent_active = True
                    saw_fin_descent = True
                    fin_id = str(execution.task.payload["fin_id"])
                    payload = application.scene.registry.free_body_pose(fin_id)
                    payload_name = "fin"
                    local_target = np.asarray(
                        execution.task.payload["target_position"], dtype=float
                    ) + np.asarray([0.0, 0.0, 0.032])
                else:
                    continue
                payload_samples[payload_name].append((payload.position.copy(), payload.quaternion.copy()))
                tray_index = int(str(execution.task.tray_id)[-2:]) - 1
                tray_pose = application.scene.registry.batch_tray_pose(tray_index)
                desired_position = np.asarray(
                    application.scene.registry.batch_to_world(tray_index, local_target),
                    dtype=float,
                )
                xy_error = float(np.linalg.norm(payload.position[:2] - desired_position[:2]))
                orientation_error_deg = 2.0 * math.degrees(
                    math.acos(
                        min(
                            1.0,
                            max(
                                -1.0,
                                abs(float(np.dot(payload.quaternion, tray_pose.quaternion))),
                            ),
                        )
                    )
                )
                alignment_errors[payload_name].append((xy_error, orientation_error_deg))
            if saw_fin_descent and not fin_descent_active and len(payload_samples["fin"]) > 10:
                break
        else:
            raise AssertionError("first fin pure-Z descent did not complete")

        assert material_pass_counts == {material_pass_count}
        for arm_name, samples in joint_samples.items():
            commands = np.stack(samples)
            # Physical skills are sampled at this test's 10 ms MuJoCo step,
            # independently of the 50 Hz scheduling clock.  The 2.5 rad/s
            # bound therefore permits at most 0.025 rad per local joint.
            assert float(np.max(np.abs(np.diff(commands, axis=0)))) < 0.03, arm_name
        for name, samples in payload_samples.items():
            assert len(samples) > 20, name
            positions = np.stack([position for position, _quaternion in samples])
            quaternions = np.stack([quaternion for _position, quaternion in samples])
            # MuJoCo constraint compliance after mj_step is allowed only a
            # sub-visual numerical residual; the authored trajectory itself
            # has exactly constant XY and quaternion.
            assert np.max(np.ptp(positions[:, :2], axis=0)) < 0.0005, name
            reference = quaternions[0]
            maximum_angle_deg = max(
                2.0 * math.degrees(math.acos(min(1.0, max(-1.0, abs(float(np.dot(reference, quaternion)))))))
                for quaternion in quaternions
            )
            assert maximum_angle_deg < 0.5, name
            # The target is the real tray/comb payload pose, not merely the
            # Arm1 TCP.  The measured grasp transform must already be
            # compensated before descent begins and remain sub-visual for the
            # full contact motion.
            assert max(error[0] for error in alignment_errors[name]) < 0.00025, name
            assert max(error[1] for error in alignment_errors[name]) < 0.25, name
            assert positions[-1, 2] < positions[0, 2] - 0.095, name
            assert np.max(np.diff(positions[:, 2])) < 0.0002, name
    finally:
        application.close()


def test_live_product_router_visits_all_stations_without_a_global_swap() -> None:
    product = create_product_state(order_id="ASYNC_ROUTED_PRODUCT")
    scene = BrazingScene(order=product, raw=True)
    router = AsyncLineProcessRouter(scene)
    router.activate()
    try:
        expected = {
            OrderStage.BASE_LOADING: SHALLOW_U_LAYOUT.station_s1_xy,
            OrderStage.MATERIAL_APPLICATION: SHALLOW_U_LAYOUT.station_s2a_xy,
            OrderStage.MATERIAL_INSPECTION: SHALLOW_U_LAYOUT.station_s2b_xy,
            OrderStage.FIN_ASSEMBLY: SHALLOW_U_LAYOUT.station_s3_xy,
            OrderStage.FURNACE_LOADING: SHALLOW_U_LAYOUT.rack_infeed_xy,
        }
        for stage, xy in expected.items():
            for _ in range(6000):
                if router.gate(
                    stage,
                    scene.time,
                    product_token=product.order_id,
                    safe_to_transfer=True,
                ):
                    break
                scene.step()
            else:
                raise AssertionError(f"router did not reach {stage.value}")
            assert scene.registry.free_body_pose("assembly_tray").position[:2] == pytest.approx(
                xy,
                abs=0.002,
            )
        assert bool(scene.data.eq_active[scene.registry.equality_id("tray_fixture_weld")])
        assert router.snapshot()["active_transfer"] is None
    finally:
        scene.close()
