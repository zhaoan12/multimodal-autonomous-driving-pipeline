from __future__ import annotations

import unittest
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from mmdrive_pipeline.data import load_scene
from mmdrive_pipeline.geometry import project_lidar_to_image


class ProjectionTests(unittest.TestCase):
    def test_projection_retains_visible_points(self) -> None:
        scene = load_scene("data/sample/scene_0001.json")

        result = project_lidar_to_image(
            scene.point_cloud,
            intrinsics=scene.intrinsics,
            lidar_to_camera=scene.lidar_to_camera,
        )

        self.assertEqual(len(result.points), 9)
        self.assertTrue(all(result.visible_mask))
        self.assertAlmostEqual(result.points[0].pixel_xy[0], 1098.4615, places=3)
        self.assertAlmostEqual(result.points[0].pixel_xy[1], 478.4615, places=3)


if __name__ == "__main__":
    unittest.main()
