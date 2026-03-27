"""Run projection validation on a scene manifest."""

from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from mmdrive_pipeline import run_validation_pipeline
from mmdrive_pipeline.utils.io import read_yaml, write_json


def main(config_path: str = "configs/validation.example.json") -> None:
    config = read_yaml(config_path)
    reports = run_validation_pipeline(
        dataset_manifest=config["dataset_manifest"],
        distance_tolerance_m=float(config.get("distance_tolerance_m", 2.0)),
        min_support_points=int(config.get("min_support_points", 3)),
    )
    write_json(Path(config["output_path"]), reports)


if __name__ == "__main__":
    main()
