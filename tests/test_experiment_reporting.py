from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from mmdrive_pipeline import run_experiment_report_pipeline
from mmdrive_pipeline.reporting import compile_experiment_report


class ExperimentReportingTests(unittest.TestCase):
    def test_compile_experiment_report_aggregates_metrics(self) -> None:
        report = compile_experiment_report(
            analytics_reports=[{"visible_point_ratio": 1.0}],
            validation_reports=[{"match_rate": 1.0, "grounding_rate": 1.0}],
            qa_reports=[{"mean_consistency_score": 0.4}],
        )

        self.assertEqual(report.scene_count, 1)
        self.assertEqual(report.mean_match_rate, 1.0)
        self.assertEqual(report.mean_consistency_score, 0.4)

    def test_pipeline_generates_experiment_summary(self) -> None:
        report = run_experiment_report_pipeline(
            "data/sample/scene_manifest.json",
            min_support_points=2,
            num_pairs=3,
        )

        self.assertEqual(report["scene_count"], 1)
        self.assertEqual(report["mean_match_rate"], 1.0)
        self.assertIsNotNone(report["mean_consistency_score"])


if __name__ == "__main__":
    unittest.main()
