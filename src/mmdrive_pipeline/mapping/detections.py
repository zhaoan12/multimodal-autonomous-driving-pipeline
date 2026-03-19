"""Map image detections into real-world coordinates using projected LiDAR."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from mmdrive_pipeline.data.models import Detection2D, LabeledScene
from mmdrive_pipeline.geometry.projection import project_lidar_to_image


@dataclass(slots=True)
class DetectionMapping:
    """Grounded detection with estimated camera- and LiDAR-frame location."""

    detection: Detection2D
    lidar_point_xyz: tuple[float, float, float]
    camera_point_xyz: tuple[float, float, float]
    support_count: int
    depth: float


def _point_in_bbox(pixel_xy: tuple[float, float], bbox_xyxy: tuple[float, float, float, float]) -> bool:
    x, y = pixel_xy
    x1, y1, x2, y2 = bbox_xyxy
    return x1 <= x <= x2 and y1 <= y <= y2


def map_detection_to_world(
    scene: LabeledScene,
    detection: Detection2D,
    min_support_points: int = 3,
) -> DetectionMapping | None:
    """Estimate a world coordinate for one 2D detection."""

    projection = project_lidar_to_image(
        scene.point_cloud,
        intrinsics=scene.intrinsics,
        lidar_to_camera=scene.lidar_to_camera,
    )
    support_indices = [
        index
        for index, projected_point in enumerate(projection.points)
        if _point_in_bbox(projected_point.pixel_xy, detection.bbox_xyxy)
    ]

    if len(support_indices) < min_support_points:
        return None

    camera_support = np.array(
        [projection.camera_points_xyz[projection.visible_mask][index] for index in support_indices],
        dtype=float,
    )
    lidar_support = np.array(
        [projection.points[index].point_xyz_lidar for index in support_indices],
        dtype=float,
    )
    representative_camera = np.median(camera_support, axis=0)
    representative_lidar = np.median(lidar_support, axis=0)

    return DetectionMapping(
        detection=detection,
        lidar_point_xyz=tuple(float(value) for value in representative_lidar),
        camera_point_xyz=tuple(float(value) for value in representative_camera),
        support_count=len(support_indices),
        depth=float(representative_camera[2]),
    )


def map_scene_detections(scene: LabeledScene, min_support_points: int = 3) -> list[DetectionMapping]:
    """Ground all detections attached to scene objects."""

    mappings: list[DetectionMapping] = []
    for scene_object in scene.objects:
        if scene_object.detection is None:
            continue
        mapping = map_detection_to_world(
            scene=scene,
            detection=scene_object.detection,
            min_support_points=min_support_points,
        )
        if mapping is not None:
            mappings.append(mapping)
    return mappings

