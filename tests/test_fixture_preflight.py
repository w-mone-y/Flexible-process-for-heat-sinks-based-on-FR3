from __future__ import annotations

from pathlib import Path

import numpy as np

from brazing_sim.config import FIXTURE_CONFIG, create_product_state, make_order_spec
from brazing_sim.domain import Actor, PressState, TaskSpec, TaskStatus, TaskType

ROOT = Path(__file__).resolve().parents[1]


def test_preflight_accepts_all_flexible_fixture_presets() -> None:
    from brazing_sim.preflight import preflight_check
    from brazing_sim.scene import BrazingScene

    scene = BrazingScene(ROOT / "scenes" / "production" / "brazing_line.xml", order="A", raw=True)
    try:
        report = preflight_check(scene, order=("A", "B", "C"), raise_on_error=False)
        assert report.ok
        assert report.checked_presets == ("A", "B", "C")
        assert not report.issues
    finally:
        scene.close()


def test_comb_and_press_are_absent_during_material_application() -> None:
    from brazing_sim.scene import BrazingScene

    scene = BrazingScene(ROOT / "scenes" / "production" / "brazing_line.xml", order="A", raw=True)
    try:
        scene.reset("A", raw=True)
        comb_names = ("front_comb_20_g01l", "rear_comb_20_g01l")
        press_names = ("fixture_front_press_bar", "fixture_rear_press_bar")
        for name in (*comb_names, *press_names):
            geom = scene.model.geom(name)
            assert float(geom.rgba[3]) == 0.0
            assert int(geom.contype[0]) == 0
            assert int(geom.conaffinity[0]) == 0

        # Material inspection passes before the CONFIGURE_COMB task. The comb
        # then appears for fin insertion, while the upper press remains absent
        # until PRESS_FIXTURE starts after every fin is installed.
        scene.fixture_controller.configure_product(make_order_spec("A"))
        for name in comb_names:
            geom = scene.model.geom(name)
            assert float(geom.rgba[3]) > 0.0
            assert int(geom.contype[0]) != 0
        for name in press_names:
            geom = scene.model.geom(name)
            assert float(geom.rgba[3]) == 0.0
            assert int(geom.contype[0]) == 0
    finally:
        scene.close()


def test_order_switch_shows_only_matching_front_and_rear_comb_pair() -> None:
    from brazing_sim.scene import BrazingScene

    scene = BrazingScene(ROOT / "scenes" / "production" / "brazing_line.xml", order="B", raw=True)
    try:
        scene.fixture_controller.configure_product(make_order_spec("B"))
        for end in ("front", "rear"):
            for pitch in (20, 30, 40):
                geom = scene.model.geom(f"{end}_comb_insert_{pitch}mm_rail")
                selected = pitch == 30
                assert (float(geom.rgba[3]) > 0.0) is selected
                # The end rails are visual module keys; only the separated
                # guide fingers form the physical, top-open insertion slot.
                assert int(geom.contype[0]) == 0
                guide = scene.model.geom(f"{end}_comb_{pitch}_g01l")
                assert (float(guide.rgba[3]) > 0.0) is selected
                assert (int(guide.contype[0]) != 0) is selected
                assert int(geom.conaffinity[0]) == 0
                assert (int(guide.conaffinity[0]) != 0) is selected
    finally:
        scene.close()


def test_preflight_rejects_a_floating_comb_pedestal() -> None:
    from brazing_sim.preflight import preflight_check
    from brazing_sim.scene import BrazingScene

    scene = BrazingScene(ROOT / "scenes" / "production" / "brazing_line.xml", order="A", raw=True)
    try:
        scene.model.geom("front_comb_left_support").pos[2] += 0.005
        report = preflight_check(scene, order="A", raise_on_error=False)
        assert not report.ok
        assert any("没有落在托盘顶面" in issue.message for issue in report.issues)
    finally:
        scene.close()


def test_preflight_rejects_a_comb_tooth_disconnected_from_its_base() -> None:
    from brazing_sim.preflight import preflight_check
    from brazing_sim.scene import BrazingScene

    scene = BrazingScene(ROOT / "scenes" / "production" / "brazing_line.xml", order="A", raw=True)
    try:
        guide = scene.model.geom("front_comb_20_g01l")
        guide.pos[0] = 0.030
        guide.size[0] = 0.005
        report = preflight_check(scene, order="A", raise_on_error=False)
        assert not report.ok
        assert any("固定基座断开" in issue.message for issue in report.issues)
    finally:
        scene.close()


def test_preflight_rejects_product_c_raw_fin_inside_arm1_tool_rack() -> None:
    from brazing_sim.preflight import preflight_check
    from brazing_sim.scene import BrazingScene

    scene = BrazingScene(ROOT / "scenes" / "production" / "brazing_line.xml", order="C", raw=True)
    try:
        site = scene.model.site("raw_fin_07_site")
        # Deliberately move fin_07 into the current central quick-change rack
        # column to prove the clearance check follows the new layout.
        site.pos[0] = 0.0
        site.pos[1] = -0.085
        scene.mujoco.mj_forward(scene.model, scene.data)
        report = preflight_check(
            scene,
            order="C",
            raise_on_error=False,
            validate_current_sites=True,
        )
        assert not report.ok
        assert any(
            issue.object_name.startswith("raw_fin_07_site/") and "工具架净距" in issue.message
            for issue in report.issues
        )
    finally:
        scene.close()


def test_preflight_rejects_overlapping_shallow_u_station_tables() -> None:
    from brazing_sim.preflight import preflight_check
    from brazing_sim.scene import BrazingScene

    scene = BrazingScene(ROOT / "scenes" / "production" / "brazing_line.xml", order="A", raw=True)
    try:
        # Restore the former compact S2A Y coordinate, which made the 490 x
        # 350 mm S1/S2A table tops visibly overlap.
        body = scene.model.body("station_s2a_dispensing")
        body.pos[1] = 0.27
        anchor = scene.model.body("station_s2a_anchor")
        anchor.pos[1] = 0.27
        scene.mujoco.mj_forward(scene.model, scene.data)
        report = preflight_check(
            scene,
            order="A",
            raise_on_error=False,
            validate_current_sites=True,
        )
        assert not report.ok
        assert any("相邻工位水平净距" in issue.message for issue in report.issues)
    finally:
        scene.close()


def test_physical_press_uses_touch_force_before_completing_hold() -> None:
    from brazing_sim.fixture import FixtureTaskActor
    from brazing_sim.scene import BrazingScene

    product = create_product_state(order_id="press-touch")
    scene = BrazingScene(ROOT / "scenes" / "production" / "brazing_line.xml", order=product, raw=True)
    try:
        scene.registry.place_base_on_tray()
        for fin in product.active_fins:
            scene.registry.place_fin_in_slot(fin.fin_id)

        fixture = product.fixture
        fixture.base_weld_active = True
        fixture.comb_configured = True
        fixture.comb_aligned = True
        fixture.material_passed = True
        fixture.fins_passed = True
        actor = FixtureTaskActor(scene, product)
        actor.start_task(
            TaskSpec("press", Actor.FIXTURE, TaskType.PRESS_FIXTURE, timeout=30.0),
            scene.time,
        )

        result = TaskStatus.RUNNING
        peak_touch_force = 0.0
        while scene.time < 15.0 and result is TaskStatus.RUNNING:
            result = actor.poll_task(scene.time)
            scene.step()
            peak_touch_force = max(
                peak_touch_force,
                scene.fixture_controller.measured_touch_force_n,
            )

        assert result is TaskStatus.SUCCEEDED
        assert fixture.press_state is PressState.COMPLETE
        assert fixture.press_force_held
        assert peak_touch_force >= (
            FIXTURE_CONFIG.target_clamping_force_n - FIXTURE_CONFIG.clamping_force_tolerance_n
        )
        assert fixture.press_position_m < -0.015
        hold_id = scene.registry.equality_id("fixture_press_hold_weld")
        assert int(scene.data.eq_active[hold_id]) == 1

        # Once force hold succeeds, the two short bars must remain visually
        # still instead of alternating between the nominal and back-off
        # commands on successive solver frames.
        settled_z: list[float] = []
        for _ in range(1000):
            scene.step()
            settled_z.append(float(scene.data.body("fixture_upper_plate").xpos[2]))
        assert np.ptp(settled_z) < 0.0001
    finally:
        scene.close()
