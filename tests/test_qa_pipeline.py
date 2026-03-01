from __future__ import annotations

import unittest
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from mmdrive_pipeline.data import load_scene
from mmdrive_pipeline.qa import filter_generated_pairs, generate_scene_qa
from mmdrive_pipeline.qa.schemas import QAGenerationRecord, QAPair


class QAPipelineTests(unittest.TestCase):
    def test_mock_generation_produces_grounded_pairs(self) -> None:
        scene = load_scene("data/sample/scene_0001.json")

        record = generate_scene_qa(scene, num_pairs=3)
        filtered = filter_generated_pairs(record, scene)

        self.assertEqual(len(record.pairs), 3)
        self.assertEqual(len(filtered.kept_pairs), 3)
        self.assertEqual(filtered.rejection_reasons, [])
        self.assertEqual(record.pairs[0].question_type, "grounding")
        self.assertEqual(record.pairs[1].question_type, "spatial-relation")
        self.assertIsNotNone(filtered.mean_consistency_score)

    def test_filter_rejects_duplicates_and_ungrounded_pairs(self) -> None:
        scene = load_scene("data/sample/scene_0001.json")
        record = QAGenerationRecord(
            scene_id=scene.scene_id,
            prompt="placeholder",
            provider="test",
            pairs=[
                QAPair(
                    question="Where is car_01?",
                    answer="car_01 is ahead.",
                    rationale="car_01 is in the labels.",
                    question_type="grounding",
                ),
                QAPair(
                    question="Where is car_01?",
                    answer="Repeated question.",
                    rationale="duplicate",
                    question_type="grounding",
                ),
                QAPair(
                    question="What color is the sky?",
                    answer="Purple.",
                    rationale="Not in the scene.",
                    question_type="attribute",
                ),
            ],
        )

        filtered = filter_generated_pairs(record, scene)

        self.assertEqual(len(filtered.kept_pairs), 1)
        self.assertEqual(filtered.rejection_reasons, ["duplicate-question", "not-grounded-in-scene"])
        self.assertGreater(filtered.mean_consistency_score or 0.0, 0.0)


if __name__ == "__main__":
    unittest.main()
