"""Run a compact end-to-end benchmark and emit JSON plus markdown summaries."""

from __future__ import annotations

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from mmdrive_pipeline import run_experiment_report_pipeline
from mmdrive_pipeline.reporting import ExperimentReport, render_experiment_report_markdown
from mmdrive_pipeline.utils.io import read_yaml


def main(config_path: str = "configs/benchmark.example.json") -> None:
    config = read_yaml(config_path)
    report_payload = run_experiment_report_pipeline(
        dataset_manifest=config["dataset_manifest"],
        distance_tolerance_m=float(config.get("distance_tolerance_m", 2.0)),
        min_support_points=int(config.get("min_support_points", 3)),
        num_pairs=int(config.get("num_pairs", 5)),
    )
    report = ExperimentReport(**report_payload)

    json_path = Path(config["output_json_path"])
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(report_payload, indent=2), encoding="utf-8")

    markdown_path = Path(config["output_markdown_path"])
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.write_text(render_experiment_report_markdown(report), encoding="utf-8")


if __name__ == "__main__":
    main()
