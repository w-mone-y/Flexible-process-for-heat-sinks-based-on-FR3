from __future__ import annotations

from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[2]


class ContactMonitorTests(unittest.TestCase):
    def test_home_scene_has_no_unexpected_contact(self) -> None:
        try:
            import mujoco
            import numpy as np
        except ImportError as exc:  # pragma: no cover
            self.skipTest(str(exc))
        from brazing_sim.safety import ContactMonitor

        model = mujoco.MjModel.from_xml_path(str(ROOT / "scenes" / "production" / "brazing_line.xml"))
        data = mujoco.MjData(model)
        home = np.asarray([0.0, 0.0, 0.0, -1.57079, 0.0, 1.57079, -0.7853])
        for arm in ("arm1", "arm2", "arm3"):
            for index, value in enumerate(home, 1):
                joint = int(model.joint(f"{arm}_fr3_joint{index}").id)
                data.qpos[int(model.jnt_qposadr[joint])] = value
                data.ctrl[int(model.actuator(f"{arm}_fr3_joint{index}").id)] = value
        mujoco.mj_forward(model, data)
        for _ in range(200):
            mujoco.mj_step(model, data)
        self.assertEqual(ContactMonitor(model).unexpected(data), [])

    def test_product_c_raw_fins_remain_stable_and_clear_of_arm1_tool_rack(self) -> None:
        try:
            import numpy as np
        except ImportError as exc:  # pragma: no cover
            self.skipTest(str(exc))
        from brazing_sim.scene import BrazingScene
        from brazing_sim.safety import ContactMonitor

        scene = BrazingScene(ROOT / "scenes" / "production" / "brazing_line.xml", order="C", raw=True)
        try:
            initial = {
                index: scene.registry.free_body_pose(f"fin_{index:02d}").position.copy()
                for index in range(1, 8)
            }
            scene.step(300)
            self.assertEqual(ContactMonitor(scene.model).unexpected(scene.data), [])
            for index, position in initial.items():
                actual = scene.registry.free_body_pose(f"fin_{index:02d}").position
                self.assertLess(float(np.linalg.norm(actual - position)), 1.0e-9)
                target = scene.registry.site_pose(f"raw_fin_{index:02d}_site").position
                self.assertLess(float(np.linalg.norm(actual - target)), 1.0e-9)
        finally:
            scene.close()


if __name__ == "__main__":
    unittest.main()
