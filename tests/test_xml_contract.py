from __future__ import annotations

import math
from pathlib import Path
import unittest
from xml.etree import ElementTree

from brazing_sim.config import DISPENSER_CONFIG

ROOT = Path(__file__).resolve().parents[1]


class BrazingXmlContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        try:
            import mujoco
        except ImportError as exc:  # pragma: no cover
            raise unittest.SkipTest(str(exc))
        cls.model = mujoco.MjModel.from_xml_path(str(ROOT / "brazing_line.xml"))

    def test_three_prefixed_fr3_arms(self) -> None:
        for arm in ("arm1", "arm2", "arm3"):
            self.model.body(f"{arm}_base")
            self.model.site(f"{arm}_attachment_site")
            for index in range(1, 8):
                self.model.joint(f"{arm}_fr3_joint{index}")
                self.model.actuator(f"{arm}_fr3_joint{index}")

    def test_product_pool_and_constraints(self) -> None:
        self.model.body("base_plate")
        self.model.body("heatsink_base_plate")
        self.model.equality("raw_base_rack_weld")
        self.model.body("assembly_tray")
        self.model.body("assembly_fixture")
        for index in range(1, 13):
            fin = f"fin_{index:02d}"
            self.model.body(fin)
            self.model.equality(f"arm1_grasp_{fin}")
            self.model.equality(f"raw_{fin}_rack_weld")
            self.model.equality(f"{fin}_fixture_weld")
            self.model.equality(f"{fin}_base_weld")
            for side in ("left", "right"):
                path = f"slot_{index:02d}_{side}_brazing_path"
                self.model.body(path)
                self.model.equality(f"{path}_base_weld")

    def test_base_plate_is_one_pure_rectangular_solid(self) -> None:
        import mujoco

        base_body = self.model.body("heatsink_base_plate")
        self.assertEqual(int(base_body.geomnum[0]), 1)

        base_geom = self.model.geom("heatsink_base_plate_geom")
        self.assertEqual(int(base_geom.type[0]), int(mujoco.mjtGeom.mjGEOM_BOX))
        self.assertEqual(tuple(float(value) for value in base_geom.size), (0.18, 0.11, 0.004))

        # These former mounting/detail geoms must not silently return when the
        # order or fixture changes: the product itself is a plain cuboid plate.
        for removed_name in (
            "base_plate_geom",
            "base_plate_left_rail",
            "base_plate_right_rail",
            "base_plate_mount_hole_01",
        ):
            self.assertEqual(
                mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, removed_name),
                -1,
            )

    def test_table1_is_open_and_raw_fins_have_safe_spacing(self) -> None:
        import mujoco

        top = self.model.geom("raw_material_rack_top")
        half_x = float(self.model.geom_size[top.id, 0])
        half_y = float(self.model.geom_size[top.id, 1])
        self.assertAlmostEqual(half_x, 2.0 * half_y, places=6)
        self.assertGreater(half_x, half_y)
        table_position = self.model.body("raw_material_rack").pos
        self.assertAlmostEqual(float(table_position[0]), -1.079, places=6)
        self.assertAlmostEqual(float(table_position[1]), 0.508, places=6)
        self.assertEqual(
            mujoco.mj_name2id(
                self.model,
                mujoco.mjtObj.mjOBJ_GEOM,
                "raw_material_rack_back",
            ),
            -1,
        )
        base_site = self.model.site("raw_base_site").pos
        self.assertLess(float(base_site[0]), 0.0)
        active_sites = [self.model.site(f"raw_fin_{index:02d}_site").pos for index in range(1, 5)]
        self.assertTrue(all(float(site[0]) > 0.0 for site in active_sites))
        fin_y = [float(site[1]) for site in active_sites]
        spacing = [right - left for left, right in zip(fin_y, fin_y[1:])]
        self.assertTrue(all(value >= 0.07 - 1e-9 for value in spacing))
        self.assertGreater(
            float(self.model.geom("heatsink_base_plate_geom").size[0]),
            float(self.model.geom("heatsink_base_plate_geom").size[1]),
        )
        self.assertGreater(
            float(self.model.geom("fin_01_geom").size[0]),
            float(self.model.geom("fin_01_geom").size[1]),
        )

    def test_tools_furnace_and_camera(self) -> None:
        self.model.body("arm1_tool_rack")
        self.model.body("arm1_parallel_gripper")
        self.model.body("arm1_suction_tool")
        self.model.site("arm1_parallel_gripper_rack_site")
        self.model.site("arm1_suction_tool_rack_site")
        self.model.equality("arm1_toolchange_parallel_gripper")
        self.model.equality("arm1_toolchange_suction_tool")
        self.model.equality("arm1_rack_parallel_gripper")
        self.model.equality("arm1_rack_suction_tool")
        self.model.site("arm1_suction_tcp")
        self.model.geom("arm1_suction_pad")
        self.model.joint("arm1_left_finger_joint")
        self.model.joint("arm1_right_finger_joint")
        self.model.actuator("arm1_left_finger_actuator")
        self.model.actuator("arm1_right_finger_actuator")
        self.model.body("arm2_dual_brazing_dispenser_tool")
        self.model.site("arm2_dispenser_center_tcp")
        self.model.site("arm2_left_nozzle_tip_site")
        self.model.site("arm2_right_nozzle_tip_site")
        self.model.geom("arm2_left_nozzle_tip")
        self.model.geom("arm2_right_nozzle_tip")
        self.model.equality("arm2_dispenser_tool_weld")
        self.model.equality("fixture_press_hold_weld")
        self.model.equality("fixture_press_drive_hold_weld")
        self.model.body("furnace")
        self.model.joint("furnace_door_joint")
        self.model.actuator("furnace_door_actuator")
        self.model.camera("arm3_wrist_camera")

    def test_industrial_detail_layer_is_visual_only(self) -> None:
        """High-detail shells must never change motion or contact behaviour."""

        xml_root = ElementTree.parse(ROOT / "brazing_line.xml").getroot()
        visual_geoms = xml_root.findall(".//geom[@class='visual']")
        self.assertGreaterEqual(len(visual_geoms), 100)

        representative_details = (
            "arm1_gripper_mount_flange",
            "arm2_dispenser_flange_ring",
            "arm2_left_nozzle_sleeve",
            "conveyor_drive_motor",
            "conveyor_home_sensor",
            "batch_transfer_left_mast",
            "furnace_left_outer_skin",
            "furnace_control_cabinet",
            "furnace_exhaust_stack",
            "furnace_door_outer_skin",
        )
        for name in representative_details:
            geom = self.model.geom(name)
            self.assertEqual(int(geom.contype[0]), 0, name)
            self.assertEqual(int(geom.conaffinity[0]), 0, name)
            self.assertEqual(int(geom.group[0]), 1, name)

        # Door skins move with the existing door joint, while nozzle detail
        # follows the existing permanent Arm2 tool body and does not add a
        # second kinematic mechanism.
        door_skin = self.model.geom("furnace_door_outer_skin")
        dispenser_ring = self.model.geom("arm2_dispenser_flange_ring")
        self.assertEqual(
            int(door_skin.bodyid[0]),
            int(self.model.body("furnace_door").id),
        )
        self.assertEqual(
            int(dispenser_ring.bodyid[0]),
            int(self.model.body("arm2_dual_brazing_dispenser_tool").id),
        )

    def test_visual_quality_profile_is_interactive_gpu_friendly(self) -> None:
        quality = self.model.vis.quality
        self.assertLessEqual(int(quality.shadowsize), 2048)
        self.assertLessEqual(int(quality.offsamples), 2)
        self.assertEqual(int(quality.numslices), 16)
        self.assertEqual(int(quality.numstacks), 8)
        self.assertEqual(int(quality.numquads), 2)
        self.assertEqual(
            sum(bool(self.model.light_castshadow[index]) for index in range(self.model.nlight)),
            1,
        )
        # Camera clients request 640 x 480 by default; a larger implicit FBO
        # only consumes GPU memory and does not improve the interactive viewer.
        self.assertLessEqual(int(self.model.vis.global_.offwidth), 640)
        self.assertLessEqual(int(self.model.vis.global_.offheight), 480)

    def test_three_layer_batch_rack_and_transfer_contract(self) -> None:
        import mujoco

        for index in range(1, 4):
            self.model.body(f"batch_tray_{index:02d}")
            self.model.joint(f"batch_tray_{index:02d}_free")
            self.model.geom(f"batch_tray_{index:02d}_geom")
            for fin_index in range(1, 13):
                self.model.geom(f"batch_tray_{index:02d}_fin_{fin_index:02d}")
            for path_index in range(1, 25):
                self.model.geom(f"batch_tray_{index:02d}_braze_{path_index:02d}")
            self.model.equality(f"batch_carrier_tray_{index:02d}_weld")
            self.model.equality(f"batch_rack_tray_{index:02d}_weld")
            self.model.equality(f"batch_output_tray_{index:02d}_weld")
            for shelf_index in range(3):
                self.model.equality(f"batch_rack_tray_{index:02d}_shelf_{shelf_index}_weld")

        axes = {
            "batch_outfeed_joint": ("batch_outfeed_actuator", (0.0, 1.0, 0.0)),
            "batch_lift_joint": ("batch_lift_actuator", (0.0, 0.0, 1.0)),
            "batch_pusher_joint": ("batch_pusher_actuator", (0.0, 1.0, 0.0)),
            "batch_output_joint": ("batch_output_actuator", (1.0, 0.0, 0.0)),
        }
        for joint_name, (actuator_name, axis) in axes.items():
            joint = self.model.joint(joint_name)
            actuator = self.model.actuator(actuator_name)
            self.assertEqual(int(joint.type[0]), int(mujoco.mjtJoint.mjJNT_SLIDE))
            self.assertEqual(tuple(float(value) for value in joint.axis), axis)
            self.assertEqual(int(actuator.trnid[0]), int(joint.id))

        shelf_heights = []
        for index in range(3):
            shelf = self.model.body(f"batch_rack_shelf_{index}")
            shelf_heights.append(float(shelf.pos[2]))
            self.model.site(f"batch_rack_shelf_site_{index}")
            self.model.geom(f"batch_rack_{index}_left_rail")
            self.model.geom(f"batch_rack_{index}_right_rail")
            self.model.geom(f"batch_rack_{index}_stop")
            self.model.geom(f"batch_rack_{index}_lock_pin")
            self.model.body(f"batch_rack_lock_body_{index}")
            lock_joint = self.model.joint(f"batch_rack_lock_joint_{index}")
            lock_actuator = self.model.actuator(f"batch_rack_lock_actuator_{index}")
            self.model.sensor(f"batch_rack_lock_position_sensor_{index}")
            self.assertEqual(
                int(lock_joint.type[0]),
                int(mujoco.mjtJoint.mjJNT_SLIDE),
            )
            self.assertEqual(
                tuple(float(value) for value in lock_joint.axis),
                (-1.0, 0.0, 0.0),
            )
            self.assertEqual(
                tuple(float(value) for value in lock_joint.range),
                (0.0, 0.025),
            )
            self.assertEqual(int(lock_actuator.trnid[0]), int(lock_joint.id))
            for roller in range(1, 6):
                self.model.geom(f"batch_rack_{index}_roller_{roller}")
            self.model.geom(f"batch_rack_{index}_left_entry_wheel")
            self.model.geom(f"batch_rack_{index}_right_entry_wheel")
            self.model.geom(f"batch_rack_{index}_lock_indicator")
        self.assertEqual(shelf_heights, sorted(shelf_heights))
        self.assertTrue(all(right - left >= 0.15 for left, right in zip(shelf_heights, shelf_heights[1:])))

        for name in (
            "batch_lift_position_sensor",
            "batch_pusher_position_sensor",
            "batch_output_position_sensor",
            "batch_outfeed_position_sensor",
        ):
            self.model.sensor(name)

        output_x = []
        output_z = []
        for index in range(1, 4):
            slot = self.model.body(f"batch_output_slot_{index:02d}")
            output_x.append(float(slot.pos[0]))
            output_z.append(float(slot.pos[2]))
            self.model.geom(f"batch_output_slot_{index:02d}_deck")
        self.assertTrue(all(right - left >= 0.48 for left, right in zip(output_x, output_x[1:])))
        self.assertAlmostEqual(max(output_z) - min(output_z), 0.0, places=9)

        for name in (
            "batch_pusher_outer_left_housing",
            "batch_pusher_outer_right_housing",
            "batch_pusher_inner_left_beam",
            "batch_pusher_inner_right_beam",
            "batch_pusher_contact_beam",
            "batch_pusher_left_contact_pad",
            "batch_pusher_right_contact_pad",
            "batch_pusher_drive_rod",
        ):
            geom = self.model.geom(name)
            self.assertEqual(int(geom.contype[0]), 0)
            self.assertEqual(int(geom.conaffinity[0]), 0)

    def test_front_and_rear_comb_modules_are_complete_matching_pairs(self) -> None:
        self.model.body("fixture_front_comb_frame")
        self.model.body("fixture_rear_comb_frame")
        self.model.body("fixture_front_comb_insert")
        self.model.body("fixture_rear_comb_insert")

        expected_direct_geoms = {15: 20, 20: 16, 30: 12, 40: 8}
        for pitch_mm, geom_count in expected_direct_geoms.items():
            for end in ("front", "rear"):
                module = self.model.body(f"{end}_comb_insert_{pitch_mm}mm")
                self.assertEqual(int(module.geomnum[0]), geom_count)
                self.model.geom(f"{end}_comb_insert_{pitch_mm}mm_rail")

        front_frame = self.model.body("fixture_front_comb_frame").pos
        rear_frame = self.model.body("fixture_rear_comb_frame").pos
        self.assertAlmostEqual(float(front_frame[0]), -0.120, places=6)
        self.assertAlmostEqual(float(rear_frame[0]), 0.120, places=6)

        for index in range(1, 9):
            front = self.model.site(f"front_comb_slot_{index:02d}").pos
            rear = self.model.site(f"rear_comb_slot_{index:02d}").pos
            self.assertAlmostEqual(float(front[0]), -float(rear[0]), places=9)
            self.assertAlmostEqual(float(front[1]), float(rear[1]), places=9)
            self.assertAlmostEqual(float(front[2]), float(rear[2]), places=9)

    def test_comb_teeth_are_cantilevered_from_grounded_tray_pedestals(self) -> None:
        """No active guide may appear as an unsupported floating rectangle."""

        tray_top = float(
            self.model.geom("fixture_tray_geom").pos[2] + self.model.geom("fixture_tray_geom").size[2]
        )
        base_half_length = float(self.model.geom("heatsink_base_plate_geom").size[0])
        for end, sign in (("front", -1.0), ("rear", 1.0)):
            frame_x = float(self.model.body(f"fixture_{end}_comb_frame").pos[0])
            for support_name in ("left_support", "right_support"):
                support = self.model.geom(f"{end}_comb_{support_name}")
                self.assertAlmostEqual(
                    float(support.pos[2] - support.size[2]),
                    tray_top,
                    places=9,
                )

            pedestal = self.model.geom(f"{end}_comb_frame_beam")
            pedestal_world_x = frame_x + float(pedestal.pos[0])
            self.assertGreater(sign * pedestal_world_x, base_half_length)

            for pitch_mm in (15, 20, 30, 40):
                rail = self.model.geom(f"{end}_comb_insert_{pitch_mm}mm_rail")
                rail_x = (float(rail.pos[0] - rail.size[0]), float(rail.pos[0] + rail.size[0]))
                rail_z = (float(rail.pos[2] - rail.size[2]), float(rail.pos[2] + rail.size[2]))
                guide = self.model.geom(f"{end}_comb_{pitch_mm}_g01l")
                guide_x = (
                    float(guide.pos[0] - guide.size[0]),
                    float(guide.pos[0] + guide.size[0]),
                )
                guide_z = (
                    float(guide.pos[2] - guide.size[2]),
                    float(guide.pos[2] + guide.size[2]),
                )
                self.assertLess(max(rail_x[0], guide_x[0]), min(rail_x[1], guide_x[1]))
                self.assertLess(max(rail_z[0], guide_z[0]), min(rail_z[1], guide_z[1]))

                guide_world_inner_x = frame_x + (guide_x[1] if end == "front" else guide_x[0])
                self.assertLess(sign * guide_world_inner_x, base_half_length)

    def test_arm2_dispenser_has_symmetric_configured_dual_nozzles(self) -> None:
        import mujoco

        left = self.model.site("arm2_left_nozzle_tip_site").pos
        right = self.model.site("arm2_right_nozzle_tip_site").pos
        centre = self.model.site("arm2_dispenser_center_tcp").pos

        self.assertAlmostEqual(
            float(right[1] - left[1]),
            DISPENSER_CONFIG.nozzle_spacing,
            places=9,
        )
        for axis in range(3):
            self.assertAlmostEqual(
                float(centre[axis]),
                float((left[axis] + right[axis]) / 2.0),
                places=9,
            )
        for name in (
            "arm2_left_feed_tube",
            "arm2_right_feed_tube",
            "arm2_left_nozzle",
            "arm2_right_nozzle",
        ):
            geom = self.model.geom(name)
            self.assertEqual(int(geom.type[0]), int(mujoco.mjtGeom.mjGEOM_CAPSULE))

        xml_root = ElementTree.parse(ROOT / "brazing_line.xml").getroot()

        def fromto(name: str) -> tuple[tuple[float, ...], tuple[float, ...]]:
            geom = xml_root.find(f".//geom[@name='{name}']")
            self.assertIsNotNone(geom)
            assert geom is not None
            self.assertIn("fromto", geom.attrib)
            values = tuple(float(value) for value in geom.attrib["fromto"].split())
            self.assertEqual(len(values), 6)
            return values[:3], values[3:]

        left_feed_start, left_feed_end = fromto("arm2_left_feed_tube")
        right_feed_start, right_feed_end = fromto("arm2_right_feed_tube")
        left_nozzle_start, left_nozzle_end = fromto("arm2_left_nozzle")
        right_nozzle_start, right_nozzle_end = fromto("arm2_right_nozzle")

        self.assertEqual(left_feed_end, left_nozzle_start)
        self.assertEqual(right_feed_end, right_nozzle_start)
        for actual, expected in (
            (left_nozzle_end, tuple(float(value) for value in left)),
            (right_nozzle_end, tuple(float(value) for value in right)),
        ):
            for actual_value, expected_value in zip(actual, expected):
                self.assertAlmostEqual(actual_value, expected_value, places=9)

        left_points = (left_feed_start, left_feed_end, left_nozzle_start, left_nozzle_end)
        right_points = (right_feed_start, right_feed_end, right_nozzle_start, right_nozzle_end)
        for left_point, right_point in zip(left_points, right_points):
            self.assertAlmostEqual(left_point[0], right_point[0], places=9)
            self.assertAlmostEqual(left_point[1], -right_point[1], places=9)
            self.assertAlmostEqual(left_point[2], right_point[2], places=9)

        manifold = self.model.geom("arm2_dispenser_manifold")
        manifold_y = float(manifold.pos[1])
        manifold_z = float(manifold.pos[2])
        for feed_start in (left_feed_start, right_feed_start):
            self.assertLessEqual(abs(feed_start[1] - manifold_y), float(manifold.size[1]))
            self.assertLessEqual(
                abs(feed_start[2] - manifold_z),
                float(manifold.size[2]),
            )

        for nozzle_start, nozzle_end in (
            (left_nozzle_start, left_nozzle_end),
            (right_nozzle_start, right_nozzle_end),
        ):
            inward_angle_deg = math.degrees(
                math.atan2(
                    abs(nozzle_end[1] - nozzle_start[1]),
                    abs(nozzle_end[2] - nozzle_start[2]),
                )
            )
            self.assertAlmostEqual(
                inward_angle_deg,
                DISPENSER_CONFIG.nozzle_inward_angle_deg,
                places=5,
            )

        left_channel_length = math.dist(left_feed_start, left_feed_end) + math.dist(
            left_nozzle_start,
            left_nozzle_end,
        )
        right_channel_length = math.dist(right_feed_start, right_feed_end) + math.dist(
            right_nozzle_start,
            right_nozzle_end,
        )
        self.assertAlmostEqual(left_channel_length, right_channel_length, places=9)
        self.model.equality("arm2_dispenser_tool_weld")

    def test_fixture_press_has_slide_actuator_and_complete_feedback_chain(self) -> None:
        import mujoco

        slide = self.model.joint("fixture_press_slide")
        self.assertEqual(int(slide.type[0]), int(mujoco.mjtJoint.mjJNT_SLIDE))
        self.assertEqual(tuple(float(value) for value in slide.range), (-0.024, 0.0))

        actuator = self.model.actuator("fixture_press_actuator")
        self.assertEqual(int(actuator.trnid[0]), int(slide.id))
        self.assertEqual(tuple(float(value) for value in actuator.ctrlrange), (-0.024, 0.0))

        expected_sensors = {
            "fixture_press_touch_sensor": mujoco.mjtSensor.mjSENS_TOUCH,
            "fixture_press_jointpos_sensor": mujoco.mjtSensor.mjSENS_JOINTPOS,
            "fixture_press_force_sensor": mujoco.mjtSensor.mjSENS_ACTUATORFRC,
        }
        for sensor_name, sensor_type in expected_sensors.items():
            sensor = self.model.sensor(sensor_name)
            self.assertEqual(int(sensor.type[0]), int(sensor_type))

        touch = self.model.sensor("fixture_press_touch_sensor")
        joint_position = self.model.sensor("fixture_press_jointpos_sensor")
        actuator_force = self.model.sensor("fixture_press_force_sensor")
        self.assertEqual(int(touch.objid[0]), int(self.model.site("fixture_press_touch_site").id))
        self.assertEqual(int(joint_position.objid[0]), int(slide.id))
        self.assertEqual(int(actuator_force.objid[0]), int(actuator.id))
        hold = self.model.equality("fixture_press_hold_weld")
        self.assertEqual(int(self.model.eq_active0[hold.id]), 0)
        drive_hold = self.model.equality("fixture_press_drive_hold_weld")
        self.assertEqual(int(self.model.eq_active0[drive_hold.id]), 0)

    def test_upper_press_keeps_only_two_transverse_contact_bars(self) -> None:
        import mujoco

        for removed in (
            "fixture_upper_plate_left_rail",
            "fixture_upper_plate_right_rail",
        ):
            self.assertEqual(
                mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, removed),
                -1,
            )

        front = self.model.geom("fixture_front_press_bar")
        rear = self.model.geom("fixture_rear_press_bar")
        self.assertEqual(int(front.type[0]), int(mujoco.mjtGeom.mjGEOM_BOX))
        self.assertEqual(int(rear.type[0]), int(mujoco.mjtGeom.mjGEOM_BOX))
        self.assertAlmostEqual(float(front.pos[0]), -float(rear.pos[0]), places=9)
        self.assertAlmostEqual(float(front.pos[1]), float(rear.pos[1]), places=9)
        self.assertGreater(float(front.size[1]), float(front.size[0]))

    def test_conveyor_is_level_and_aligned_with_the_front_furnace(self) -> None:
        import mujoco

        slide = self.model.joint("conveyor_slide_joint")
        actuator = self.model.actuator("conveyor_slide_actuator")
        self.assertEqual(int(slide.type[0]), int(mujoco.mjtJoint.mjJNT_SLIDE))
        self.assertEqual(tuple(float(value) for value in slide.axis), (0.0, 1.0, 0.0))
        self.assertEqual(tuple(float(value) for value in slide.range), (0.0, 0.63))
        self.assertEqual(int(actuator.trnid[0]), int(slide.id))

        home = self.model.site("conveyor_home_site").pos
        furnace_target = self.model.site("conveyor_furnace_site").pos
        furnace_body = self.model.body("furnace").pos
        furnace_site = self.model.site("furnace_tray_pose").pos
        furnace_world = furnace_body + furnace_site
        self.assertAlmostEqual(float(home[0]), float(furnace_target[0]), places=9)
        self.assertAlmostEqual(float(home[2]), float(furnace_target[2]), places=9)
        self.assertGreater(float(furnace_target[1]), float(home[1]))
        for actual, expected in zip(furnace_world, furnace_target):
            self.assertAlmostEqual(float(actual), float(expected), places=9)

        left_wall = self.model.geom("furnace_left_wall")
        right_wall = self.model.geom("furnace_right_wall")
        clear_width = float((right_wall.pos[0] - right_wall.size[0]) - (left_wall.pos[0] + left_wall.size[0]))
        handle = self.model.geom("fixture_tray_right_handle")
        handled_tray_width = 2.0 * float(handle.pos[0] + handle.size[0])
        self.assertGreaterEqual(clear_width, handled_tray_width + 0.04)

    def test_arm1_tool_rack_is_between_table1_and_table2(self) -> None:
        table1_x = float(self.model.body("raw_material_rack").pos[0])
        rack_x = float(self.model.body("arm1_tool_rack").pos[0])
        table2_x = float(self.model.body("table2").pos[0])
        self.assertLess(table1_x, rack_x)
        self.assertLess(rack_x, table2_x)
        self.assertAlmostEqual(rack_x, -0.49, places=6)
        self.assertAlmostEqual(float(self.model.body("arm1_tool_rack").pos[1]), 0.42, places=6)

    def test_arm2_has_only_a_permanently_mounted_dispenser(self) -> None:
        import mujoco

        for removed in (
            "arm2_tool_rack",
            "arm2_tray_transfer",
        ):
            self.assertEqual(
                mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, removed),
                -1,
            )
        for removed in (
            "arm2_toolchange_tray_transfer",
            "arm2_rack_brazing_dispenser",
            "arm2_rack_tray_transfer",
            "arm2_tray_carry",
        ):
            self.assertEqual(
                mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_EQUALITY, removed),
                -1,
            )
        dispenser_weld = self.model.equality("arm2_dispenser_tool_weld")
        self.assertEqual(int(self.model.eq_active0[dispenser_weld.id]), 1)

    def test_fixture_and_tray_do_not_self_collide(self) -> None:
        xml = (ROOT / "brazing_line.xml").read_text(encoding="utf-8")
        self.assertIn('<exclude body1="assembly_fixture" body2="assembly_tray"/>', xml)


if __name__ == "__main__":
    unittest.main()
