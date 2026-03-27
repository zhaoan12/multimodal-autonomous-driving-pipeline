from __future__ import annotations

import unittest
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from mmdrive_pipeline import run_qa_generation_pipeline, run_validation_pipeline


class PipelineEntrypointTests(unittest.TestCase):
    def test_validation_pipeline_runs_from_manifest(self) -> None:
        reports = run_validation_pipeline(
            "data/sample/scene_manifest.json",
            min_support_points=2,
        )

        self.assertEqual(len(reports), 1)
        self.assertEqual(reports[0]["match_rate"], 1.0)

    def test_qa_pipeline_runs_from_manifest(self) -> None:
        payload = run_qa_generation_pipeline(
            "data/sample/scene_manifest.json",
            num_pairs=2,
            filter_output=True,
        )

        self.assertEqual(len(payload), 1)
        self.assertEqual(len(payload[0]["pairs"]), 2)
        self.assertEqual(len(payload[0]["filtered_pairs"]), 2)


if __name__ == "__main__":
    unittest.main()
