"""Projection from LiDAR coordinates into image coordinates."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from mmdrive_pipeline.data.models import (
    CameraIntrinsics,
    LidarPointCloud,
    ProjectedPoint,
)
from mmdrive_pipeline.geometry.transforms import transform_points
from mmdrive_pipeline.data.models import Extrinsics


@dataclass(slots=True)
class ProjectionResult:
    """Container for projected points and supporting masks."""

    points: list[ProjectedPoint]
    camera_points_xyz: np.ndarray
    visible_mask: np.ndarray


def project_lidar_to_image(
    point_cloud: LidarPointCloud,
    intrinsics: CameraIntrinsics,
    lidar_to_camera: Extrinsics,
    min_depth: float = 1e-6,
    clip_to_image: bool = True,
) -> ProjectionResult:
    """Project LiDAR points into camera pixels."""

    camera_points = transform_points(point_cloud.points_xyz, lidar_to_camera)
    depths = camera_points[:, 2]
    valid_depth_mask = depths > min_depth

    projected = camera_points[valid_depth_mask] @ intrinsics.matrix().T
    projected_xy = projected[:, :2] / projected[:, 2:3]

    visible_mask = valid_depth_mask.copy()
    valid_indices = np.flatnonzero(valid_depth_mask)

    if clip_to_image:
        in_bounds = (
            (projected_xy[:, 0] >= 0.0)
            & (projected_xy[:, 0] < intrinsics.width)
            & (projected_xy[:, 1] >= 0.0)
            & (projected_xy[:, 1] < intrinsics.height)
        )
        visible_mask[valid_indices] = in_bounds
        projected_xy = projected_xy[in_bounds]
        valid_indices = valid_indices[in_bounds]

    points = [
        ProjectedPoint(
            pixel_xy=(float(pixel_x), float(pixel_y)),
            depth=float(depths[index]),
            point_xyz_lidar=tuple(float(v) for v in point_cloud.points_xyz[index]),
        )
        for (pixel_x, pixel_y), index in zip(projected_xy, valid_indices, strict=True)
    ]

    return ProjectionResult(
        points=points,
        camera_points_xyz=camera_points,
        visible_mask=visible_mask,
    )


def project_points_xyz(
    points_xyz: np.ndarray,
    intrinsics: CameraIntrinsics,
    lidar_to_camera: Extrinsics,
) -> np.ndarray:
    """Convenience projection helper that returns visible pixel coordinates."""

    point_cloud = LidarPointCloud(points_xyz=points_xyz)
    result = project_lidar_to_image(point_cloud, intrinsics, lidar_to_camera)
    return np.array([point.pixel_xy for point in result.points], dtype=float)
