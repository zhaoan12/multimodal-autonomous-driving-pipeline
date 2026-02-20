"""Create a deterministic dataset split from a scene manifest."""

from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from mmdrive_pipeline.data import create_dataset_split, load_dataset_manifest
from mmdrive_pipeline.utils.io import read_yaml, write_json


def main(config_path: str = "configs/split_generation.example.json") -> None:
    config = read_yaml(config_path)
    scene_paths = load_dataset_manifest(config["dataset_manifest"])
    split = create_dataset_split(
        scene_paths=scene_paths,
        train_ratio=float(config.get("train_ratio", 0.7)),
        val_ratio=float(config.get("val_ratio", 0.15)),
    )
    write_json(Path(config["output_path"]), split.to_dict())


if __name__ == "__main__":
    main()
