from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from mmdrive_pipeline.geometry import diagnose_calibration, load_calibration


class CalibrationDiagnosticsTests(unittest.TestCase):
    def test_diagnostics_expose_field_of_view_and_translation(self) -> None:
        bundle = load_calibration("configs/calibration.example.json")

        diagnostics = diagnose_calibration(bundle)

        self.assertEqual(diagnostics.source_frame, "lidar")
        self.assertEqual(diagnostics.target_frame, "camera")
        self.assertGreater(diagnostics.horizontal_fov_deg, 70.0)
        self.assertGreater(diagnostics.translation_norm_m, 1.0)


if __name__ == "__main__":
    unittest.main()
