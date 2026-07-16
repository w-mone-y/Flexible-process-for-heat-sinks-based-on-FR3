from __future__ import annotations

from pathlib import Path
import unittest

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
        self.model.equality("raw_base_rack_weld")
        self.model.body("assembly_tray")
        self.model.body("assembly_fixture")
        for index in range(1, 9):
            fin = f"fin_{index:02d}"
            self.model.body(fin)
            self.model.equality(f"arm1_grasp_{fin}")
            self.model.equality(f"raw_{fin}_rack_weld")
            self.model.equality(f"{fin}_fixture_weld")
            self.model.equality(f"{fin}_base_weld")
            for side in ("left", "right"):
                self.model.body(f"brazing_path_{fin}_{side}")

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
            float(self.model.geom("base_plate_geom").size[0]),
            float(self.model.geom("base_plate_geom").size[1]),
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
        self.model.body("arm2_brazing_dispenser")
        self.model.body("arm2_tray_transfer")
        self.model.body("arm2_tool_rack")
        self.model.site("arm2_brazing_dispenser_rack_site")
        self.model.site("arm2_tray_transfer_rack_site")
        self.model.site("arm2_brazing_dispenser_tcp")
        self.model.site("arm2_tray_transfer_tcp")
        self.model.geom("arm2_dispenser_tip")
        self.model.equality("arm2_toolchange_brazing_dispenser")
        self.model.equality("arm2_toolchange_tray_transfer")
        self.model.equality("arm2_rack_brazing_dispenser")
        self.model.equality("arm2_rack_tray_transfer")
        self.model.equality("arm2_tray_carry")
        self.model.body("furnace")
        self.model.joint("furnace_door_joint")
        self.model.actuator("furnace_door_actuator")
        self.model.camera("arm3_wrist_camera")

    def test_arm1_tool_rack_is_between_table1_and_table2(self) -> None:
        table1_x = float(self.model.body("raw_material_rack").pos[0])
        rack_x = float(self.model.body("arm1_tool_rack").pos[0])
        table2_x = float(self.model.body("table2").pos[0])
        self.assertLess(table1_x, rack_x)
        self.assertLess(rack_x, table2_x)
        self.assertAlmostEqual(rack_x, -0.49, places=6)
        self.assertAlmostEqual(float(self.model.body("arm1_tool_rack").pos[1]), 0.42, places=6)

    def test_arm2_tool_rack_is_in_front_and_carry_weld_uses_transfer_tool(self) -> None:
        rack = self.model.body("arm2_tool_rack").pos
        base = self.model.body("arm2_base").pos
        self.assertGreater(float(rack[1]), float(base[1]))
        self.assertAlmostEqual(float(rack[1]), 0.12, places=6)
        self.assertAlmostEqual(
            float(self.model.site("arm2_brazing_dispenser_rack_site").pos[2]), 0.295, places=6
        )
        xml = (ROOT / "brazing_line.xml").read_text(encoding="utf-8")
        self.assertIn(
            '<weld name="arm2_tray_carry" body1="arm2_tray_transfer" body2="assembly_tray"',
            xml,
        )

    def test_fixture_and_tray_do_not_self_collide(self) -> None:
        xml = (ROOT / "brazing_line.xml").read_text(encoding="utf-8")
        self.assertIn('<exclude body1="assembly_fixture" body2="assembly_tray"/>', xml)


if __name__ == "__main__":
    unittest.main()
