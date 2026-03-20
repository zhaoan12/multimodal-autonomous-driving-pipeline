"""Scene ingestion helpers for labeled autonomous driving data."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from mmdrive_pipeline.data.models import (
    CameraIntrinsics,
    Detection2D,
    Extrinsics,
    LabeledScene,
    LidarPointCloud,
    SceneObject,
)
from mmdrive_pipeline.utils.io import read_yaml


def load_scene(scene_path: str | Path) -> LabeledScene:
    """Load a labeled scene from a JSON file."""

    path = Path(scene_path)
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    intrinsics_cfg = payload["camera_intrinsics"]
    extrinsics_cfg = payload["lidar_to_camera"]
    intrinsics = CameraIntrinsics(**intrinsics_cfg)
    extrinsics = Extrinsics(
        rotation=np.array(extrinsics_cfg["rotation"], dtype=float),
        translation=np.array(extrinsics_cfg["translation"], dtype=float),
        source_frame=extrinsics_cfg.get("source_frame", "lidar"),
        target_frame=extrinsics_cfg.get("target_frame", "camera"),
    )
    point_cloud = LidarPointCloud(
        points_xyz=np.array(payload["point_cloud"]["points_xyz"], dtype=float),
        intensities=np.array(payload["point_cloud"]["intensities"], dtype=float)
        if payload["point_cloud"].get("intensities") is not None
        else None,
        frame_id=payload["point_cloud"].get("frame_id", "lidar"),
    )
    objects = []
    for item in payload.get("objects", []):
        detection_payload = item.get("detection")
        detection = Detection2D(**detection_payload) if detection_payload else None
        objects.append(
            SceneObject(
                object_id=item["object_id"],
                label=item["label"],
                position_xyz=tuple(item["position_xyz"]),
                bbox_size_xyz=tuple(item["bbox_size_xyz"]) if item.get("bbox_size_xyz") else None,
                detection=detection,
                metadata=item.get("metadata", {}),
            )
        )

    return LabeledScene(
        scene_id=payload["scene_id"],
        intrinsics=intrinsics,
        lidar_to_camera=extrinsics,
        point_cloud=point_cloud,
        objects=objects,
        image_path=payload.get("image_path"),
        metadata=payload.get("metadata", {}),
    )


def load_dataset_manifest(path: str | Path) -> list[Path]:
    """Load a YAML manifest listing scene JSON files."""

    manifest = read_yaml(path)
    root = Path(path).parent
    return [root / relative_path for relative_path in manifest.get("scenes", [])]


def load_scene_collection(manifest_path: str | Path) -> list[LabeledScene]:
    """Load a collection of scenes from a manifest."""

    return [load_scene(scene_path) for scene_path in load_dataset_manifest(manifest_path)]

