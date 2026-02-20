from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from mmdrive_pipeline.data import load_scene
from mmdrive_pipeline.qa import build_scene_context, render_scene_prompt


class QAContextTests(unittest.TestCase):
    def test_scene_context_builds_nearest_object_relations(self) -> None:
        scene = load_scene("data/sample/scene_0001.json")

        context = build_scene_context(scene)

        self.assertEqual(context.scene_id, "scene_0001")
        self.assertEqual(len(context.relations), 3)
        self.assertTrue(all(relation.nearest_object_id is not None for relation in context.relations))

    def test_prompt_includes_relation_section(self) -> None:
        scene = load_scene("data/sample/scene_0001.json")

        prompt = render_scene_prompt(scene, num_pairs=2)

        self.assertIn("Spatial relations:", prompt)
        self.assertIn("nearest", prompt)


if __name__ == "__main__":
    unittest.main()
