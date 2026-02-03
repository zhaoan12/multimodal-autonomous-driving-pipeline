"""Scene-level analytics for inspection and experiment summaries."""

from __future__ import annotations

from dataclasses import asdict, dataclass

from mmdrive_pipeline.data.models import LabeledScene
from mmdrive_pipeline.geometry.projection import project_lidar_to_image


@dataclass(slots=True)
class SceneMetrics:
    """Aggregate metrics describing one scene."""

    scene_id: str
    object_count: int
    detected_object_count: int
    point_count: int
    visible_point_count: int
    detection_coverage: float
    visible_point_ratio: float
    mean_object_distance: float | None
    labels: dict[str, int]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def summarize_scene(scene: LabeledScene) -> SceneMetrics:
    """Compute lightweight metrics useful for dataset inspection."""

    projection = project_lidar_to_image(
        scene.point_cloud,
        intrinsics=scene.intrinsics,
        lidar_to_camera=scene.lidar_to_camera,
    )
    object_count = len(scene.objects)
    detected_object_count = len([obj for obj in scene.objects if obj.detection is not None])
    point_count = len(scene.point_cloud.points_xyz)
    visible_point_count = len(projection.points)
    label_counts: dict[str, int] = {}
    for scene_object in scene.objects:
        label_counts[scene_object.label] = label_counts.get(scene_object.label, 0) + 1

    mean_object_distance = None
    if scene.objects:
        mean_object_distance = sum(
            (x * x + y * y + z * z) ** 0.5 for x, y, z in (obj.position_xyz for obj in scene.objects)
        ) / object_count

    return SceneMetrics(
        scene_id=scene.scene_id,
        object_count=object_count,
        detected_object_count=detected_object_count,
        point_count=point_count,
        visible_point_count=visible_point_count,
        detection_coverage=(detected_object_count / object_count) if object_count else 0.0,
        visible_point_ratio=(visible_point_count / point_count) if point_count else 0.0,
        mean_object_distance=mean_object_distance,
        labels=label_counts,
    )


def summarize_dataset(scenes: list[LabeledScene]) -> list[SceneMetrics]:
    """Summarize every scene in a collection."""

    return [summarize_scene(scene) for scene in scenes]
