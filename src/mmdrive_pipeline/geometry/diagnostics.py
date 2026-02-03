"""Projection diagnostics for calibration and scene geometry inspection."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from math import sqrt

from mmdrive_pipeline.data.models import LabeledScene
from mmdrive_pipeline.geometry.projection import ProjectionResult, project_lidar_to_image


@dataclass(slots=True)
class ProjectionDiagnostics:
    """Summary statistics describing one projection run."""

    scene_id: str
    point_count: int
    visible_point_count: int
    visibility_ratio: float
    min_depth: float | None
    max_depth: float | None
    mean_depth: float | None
    image_center_bias: float | None

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def summarize_projection(scene: LabeledScene) -> ProjectionDiagnostics:
    """Compute depth and image-space diagnostics for one scene."""

    projection = project_lidar_to_image(
        scene.point_cloud,
        intrinsics=scene.intrinsics,
        lidar_to_camera=scene.lidar_to_camera,
    )
    depths = [point.depth for point in projection.points]
    center_x = scene.intrinsics.width / 2.0
    center_y = scene.intrinsics.height / 2.0
    if projection.points:
        center_bias = sum(
            sqrt((point.pixel_xy[0] - center_x) ** 2 + (point.pixel_xy[1] - center_y) ** 2)
            for point in projection.points
        ) / len(projection.points)
    else:
        center_bias = None
    return ProjectionDiagnostics(
        scene_id=scene.scene_id,
        point_count=len(scene.point_cloud.points_xyz),
        visible_point_count=len(projection.points),
        visibility_ratio=(len(projection.points) / len(scene.point_cloud.points_xyz))
        if scene.point_cloud.points_xyz
        else 0.0,
        min_depth=min(depths) if depths else None,
        max_depth=max(depths) if depths else None,
        mean_depth=(sum(depths) / len(depths)) if depths else None,
        image_center_bias=center_bias,
    )


def compute_reprojection_residuals(
    projection: ProjectionResult,
    reference_pixels: list[tuple[float, float]],
) -> list[float]:
    """Measure Euclidean residuals against reference image points."""

    if len(projection.points) != len(reference_pixels):
        raise ValueError("Projection and reference pixel counts must match.")
    return [
        sqrt((projected.pixel_xy[0] - reference[0]) ** 2 + (projected.pixel_xy[1] - reference[1]) ** 2)
        for projected, reference in zip(projection.points, reference_pixels, strict=True)
    ]
