from __future__ import annotations

import json
import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class WorkflowScriptTests(unittest.TestCase):
    def test_benchmark_script_emits_json_and_markdown(self) -> None:
        subprocess.run(
            [sys.executable, "scripts/run_benchmark.py"],
            cwd=ROOT,
            check=True,
        )

        json_path = ROOT / "data/generated/benchmark_report.json"
        markdown_path = ROOT / "data/generated/benchmark_report.md"

        self.assertTrue(json_path.exists())
        self.assertTrue(markdown_path.exists())
        payload = json.loads(json_path.read_text(encoding="utf-8"))
        self.assertEqual(payload["scene_count"], 1)

    def test_split_and_calibration_scripts_emit_outputs(self) -> None:
        subprocess.run(
            [sys.executable, "scripts/create_split.py"],
            cwd=ROOT,
            check=True,
        )
        subprocess.run(
            [sys.executable, "scripts/check_calibration.py"],
            cwd=ROOT,
            check=True,
        )

        self.assertTrue((ROOT / "data/generated/dataset_split.json").exists())
        self.assertTrue((ROOT / "data/generated/calibration_diagnostics.json").exists())


if __name__ == "__main__":
    unittest.main()
