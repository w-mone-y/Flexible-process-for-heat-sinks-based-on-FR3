from __future__ import annotations

from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]


class ContactMonitorTests(unittest.TestCase):
    def test_home_scene_has_no_unexpected_contact(self) -> None:
        try:
            import mujoco
            import numpy as np
        except ImportError as exc:  # pragma: no cover
            self.skipTest(str(exc))
        from brazing_sim.safety import ContactMonitor

        model = mujoco.MjModel.from_xml_path(str(ROOT / "brazing_line.xml"))
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


if __name__ == "__main__":
    unittest.main()
