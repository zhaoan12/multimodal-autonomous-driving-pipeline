"""Projection from LiDAR coordinates into image coordinates."""

from __future__ import annotations

from dataclasses import dataclass

from mmdrive_pipeline.data.models import (
    CameraIntrinsics,
    LidarPointCloud,
    ProjectedPoint,
)
from mmdrive_pipeline.data.models import Extrinsics
from mmdrive_pipeline.geometry.transforms import transform_points


@dataclass(slots=True)
class ProjectionResult:
    """Container for projected points and supporting masks."""

    points: list[ProjectedPoint]
    camera_points_xyz: list[tuple[float, float, float]]
    visible_mask: list[bool]


def project_lidar_to_image(
    point_cloud: LidarPointCloud,
    intrinsics: CameraIntrinsics,
    lidar_to_camera: Extrinsics,
    min_depth: float = 1e-6,
    clip_to_image: bool = True,
) -> ProjectionResult:
    """Project LiDAR points into camera pixels."""

    camera_points = transform_points(point_cloud.points_xyz, lidar_to_camera)
    visible_mask = [False] * len(camera_points)
    points: list[ProjectedPoint] = []

    for index, camera_point in enumerate(camera_points):
        x_cam, y_cam, z_cam = camera_point
        if z_cam <= min_depth:
            continue

        pixel_x = (intrinsics.fx * x_cam / z_cam) + intrinsics.cx
        pixel_y = (intrinsics.fy * y_cam / z_cam) + intrinsics.cy
        in_bounds = (
            0.0 <= pixel_x < intrinsics.width
            and 0.0 <= pixel_y < intrinsics.height
        )

        if clip_to_image and not in_bounds:
            continue

        visible_mask[index] = True
        points.append(
            ProjectedPoint(
                pixel_xy=(float(pixel_x), float(pixel_y)),
                depth=float(z_cam),
                point_xyz_lidar=tuple(float(v) for v in point_cloud.points_xyz[index]),
            )
        )

    return ProjectionResult(
        points=points,
        camera_points_xyz=camera_points,
        visible_mask=visible_mask,
    )


def project_points_xyz(
    points_xyz: list[tuple[float, float, float]],
    intrinsics: CameraIntrinsics,
    lidar_to_camera: Extrinsics,
) -> list[tuple[float, float]]:
    """Convenience projection helper that returns visible pixel coordinates."""

    point_cloud = LidarPointCloud(points_xyz=points_xyz)
    result = project_lidar_to_image(point_cloud, intrinsics, lidar_to_camera)
    return [point.pixel_xy for point in result.points]
