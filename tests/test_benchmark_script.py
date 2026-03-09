from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from mmdrive_pipeline.reporting import ExperimentReport, render_experiment_report_markdown


class BenchmarkScriptTests(unittest.TestCase):
    def test_markdown_renderer_outputs_core_metrics(self) -> None:
        report = ExperimentReport(
            scene_count=1,
            mean_match_rate=1.0,
            mean_grounding_rate=1.0,
            mean_consistency_score=0.4,
            mean_visible_point_ratio=1.0,
        )

        markdown = render_experiment_report_markdown(report)

        self.assertIn("# Experiment Summary", markdown)
        self.assertIn("Mean match rate", markdown)


if __name__ == "__main__":
    unittest.main()
