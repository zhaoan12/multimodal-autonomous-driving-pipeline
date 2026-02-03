from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from mmdrive_pipeline import run_scene_analysis_pipeline
from mmdrive_pipeline.analytics import summarize_scene
from mmdrive_pipeline.data import load_scene


class SceneAnalyticsTests(unittest.TestCase):
    def test_scene_metrics_capture_visibility_and_labels(self) -> None:
        scene = load_scene("data/sample/scene_0001.json")

        metrics = summarize_scene(scene)

        self.assertEqual(metrics.object_count, 3)
        self.assertEqual(metrics.detected_object_count, 3)
        self.assertEqual(metrics.point_count, 9)
        self.assertEqual(metrics.visible_point_count, 9)
        self.assertEqual(metrics.labels, {"car": 1, "pedestrian": 1, "traffic_cone": 1})

    def test_pipeline_returns_serializable_metrics(self) -> None:
        reports = run_scene_analysis_pipeline("data/sample/scene_manifest.json")

        self.assertEqual(len(reports), 1)
        self.assertEqual(reports[0]["scene_id"], "scene_0001")
        self.assertEqual(reports[0]["visible_point_ratio"], 1.0)


if __name__ == "__main__":
    unittest.main()
