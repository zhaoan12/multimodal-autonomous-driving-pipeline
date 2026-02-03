from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from mmdrive_pipeline.data import load_scene
from mmdrive_pipeline.geometry import (
    compute_reprojection_residuals,
    project_lidar_to_image,
    summarize_projection,
)


class ProjectionDiagnosticsTests(unittest.TestCase):
    def test_projection_summary_reports_depth_statistics(self) -> None:
        scene = load_scene("data/sample/scene_0001.json")

        diagnostics = summarize_projection(scene)

        self.assertEqual(diagnostics.scene_id, "scene_0001")
        self.assertEqual(diagnostics.visible_point_count, 9)
        self.assertGreater(diagnostics.max_depth or 0.0, diagnostics.min_depth or 0.0)

    def test_reprojection_residuals_are_zero_for_identical_pixels(self) -> None:
        scene = load_scene("data/sample/scene_0001.json")
        projection = project_lidar_to_image(
            scene.point_cloud,
            intrinsics=scene.intrinsics,
            lidar_to_camera=scene.lidar_to_camera,
        )

        residuals = compute_reprojection_residuals(
            projection,
            [point.pixel_xy for point in projection.points],
        )

        self.assertEqual(len(residuals), 9)
        self.assertTrue(all(residual == 0.0 for residual in residuals))


if __name__ == "__main__":
    unittest.main()
