"""Run calibration diagnostics from a configuration file."""

from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from mmdrive_pipeline.geometry import diagnose_calibration, load_calibration
from mmdrive_pipeline.utils.io import read_yaml, write_json


def main(config_path: str = "configs/calibration_check.example.json") -> None:
    config = read_yaml(config_path)
    bundle = load_calibration(config["calibration_path"])
    diagnostics = diagnose_calibration(bundle)
    write_json(Path(config["output_path"]), diagnostics.to_dict())


if __name__ == "__main__":
    main()
