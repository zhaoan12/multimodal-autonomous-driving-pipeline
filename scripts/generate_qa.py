"""Generate and optionally filter QA pairs from labeled scenes."""

from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from mmdrive_pipeline import run_qa_generation_pipeline
from mmdrive_pipeline.utils.io import read_yaml, write_json


def main(config_path: str = "configs/qa_generation.example.json") -> None:
    config = read_yaml(config_path)
    payload = run_qa_generation_pipeline(
        dataset_manifest=config["dataset_manifest"],
        num_pairs=int(config.get("num_pairs", 5)),
        filter_output=bool(config.get("filter_output", True)),
    )
    write_json(Path(config["output_path"]), payload)


if __name__ == "__main__":
    main()
