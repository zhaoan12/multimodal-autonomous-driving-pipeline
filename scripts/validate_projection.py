"""Run projection validation on a scene manifest."""

from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from mmdrive_pipeline.data import load_scene_collection
from mmdrive_pipeline.utils.io import read_yaml, write_json
from mmdrive_pipeline.validation import validate_scene_projection


def main(config_path: str = "configs/validation.example.json") -> None:
    config = read_yaml(config_path)
    scenes = load_scene_collection(config["dataset_manifest"])
    reports = []
    for scene in scenes:
        report = validate_scene_projection(
            scene,
            distance_tolerance_m=float(config.get("distance_tolerance_m", 2.0)),
            min_support_points=int(config.get("min_support_points", 3)),
        )
        reports.append(
            {
                "scene_id": report.scene_id,
                "match_rate": report.match_rate,
                "mean_distance_error": report.mean_distance_error,
                "matched_objects": report.matched_objects,
                "total_objects": report.total_objects,
                "results": [
                    {
                        "object_id": item.object_id,
                        "label": item.label,
                        "matched": item.matched,
                        "distance_error": item.distance_error,
                        "support_count": item.support_count,
                    }
                    for item in report.results
                ],
            }
        )

    write_json(Path(config["output_path"]), reports)


if __name__ == "__main__":
    main()
