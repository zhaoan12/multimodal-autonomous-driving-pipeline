"""Generate and optionally filter QA pairs from labeled scenes."""

from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from mmdrive_pipeline.data import load_scene_collection
from mmdrive_pipeline.qa import filter_generated_pairs, generate_scene_qa
from mmdrive_pipeline.utils.io import read_yaml, write_json


def main(config_path: str = "configs/qa_generation.example.json") -> None:
    config = read_yaml(config_path)
    scenes = load_scene_collection(config["dataset_manifest"])
    payload = []
    for scene in scenes:
        record = generate_scene_qa(scene, num_pairs=int(config.get("num_pairs", 5)))
        item = record.to_dict()
        if config.get("filter_output", True):
            filtered = filter_generated_pairs(record, scene)
            item["filtered_pairs"] = [pair.to_dict() for pair in filtered.kept_pairs]
            item["rejected_pairs"] = [pair.to_dict() for pair in filtered.rejected_pairs]
            item["rejection_reasons"] = filtered.rejection_reasons
        payload.append(item)

    write_json(Path(config["output_path"]), payload)


if __name__ == "__main__":
    main()
