from __future__ import annotations

import importlib
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]


class LegacySmokeTests(unittest.TestCase):
    def test_legacy_entry_imports(self) -> None:
        module = importlib.import_module("multi_arm_line")
        self.assertTrue(callable(module.main))
        self.assertTrue(callable(module.build_job_stages))

    def test_legacy_xml_still_compiles(self) -> None:
        try:
            import mujoco
        except ImportError as exc:  # pragma: no cover - runtime dependency guard
            self.skipTest(str(exc))
        model = mujoco.MjModel.from_xml_path(str(ROOT / "multi_arm_line.xml"))
        self.assertEqual(model.nkey, 1)
        self.assertGreater(model.nu, 0)
        self.assertGreaterEqual(model.ncam, 1)


if __name__ == "__main__":
    unittest.main()
