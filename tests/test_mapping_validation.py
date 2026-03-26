from __future__ import annotations

import unittest
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from mmdrive_pipeline.data import load_scene
from mmdrive_pipeline.mapping import map_scene_detections
from mmdrive_pipeline.validation import validate_scene_projection


class MappingValidationTests(unittest.TestCase):
    def test_scene_detections_are_grounded(self) -> None:
        scene = load_scene("data/sample/scene_0001.json")

        mappings = map_scene_detections(scene, min_support_points=2)

        self.assertEqual(len(mappings), 3)
        self.assertEqual(mappings[0].support_count, 3)
        self.assertEqual(mappings[0].lidar_point_xyz, (10.0, -1.1, 0.4))

    def test_validation_report_matches_sample_scene(self) -> None:
        scene = load_scene("data/sample/scene_0001.json")

        report = validate_scene_projection(scene, min_support_points=2)

        self.assertEqual(report.matched_objects, 3)
        self.assertEqual(report.total_objects, 3)
        self.assertEqual(report.match_rate, 1.0)
        self.assertEqual(report.mean_distance_error, 0.0)


if __name__ == "__main__":
    unittest.main()
