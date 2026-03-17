"""Core data structures for multimodal driving scenes."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

ArrayLike = np.ndarray


@dataclass(slots=True)
class CameraIntrinsics:
    """Pinhole camera intrinsics."""

    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int

    def matrix(self) -> ArrayLike:
        return np.array(
            [
                [self.fx, 0.0, self.cx],
                [0.0, self.fy, self.cy],
                [0.0, 0.0, 1.0],
            ],
            dtype=float,
        )


@dataclass(slots=True)
class Extrinsics:
    """Rigid transform from source coordinates into target coordinates."""

    rotation: ArrayLike
    translation: ArrayLike
    source_frame: str = "lidar"
    target_frame: str = "camera"

    def transform_matrix(self) -> ArrayLike:
        matrix = np.eye(4, dtype=float)
        matrix[:3, :3] = self.rotation
        matrix[:3, 3] = self.translation
        return matrix


@dataclass(slots=True)
class LidarPointCloud:
    """LiDAR point set in homogeneous driving coordinates."""

    points_xyz: ArrayLike
    intensities: ArrayLike | None = None
    frame_id: str = "lidar"


@dataclass(slots=True)
class Detection2D:
    """Image-space object detection."""

    label: str
    bbox_xyxy: tuple[float, float, float, float]
    confidence: float
    attributes: dict[str, Any] = field(default_factory=dict)

    @property
    def center(self) -> tuple[float, float]:
        x1, y1, x2, y2 = self.bbox_xyxy
        return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


@dataclass(slots=True)
class SceneObject:
    """Annotated object with world-frame position."""

    object_id: str
    label: str
    position_xyz: tuple[float, float, float]
    bbox_size_xyz: tuple[float, float, float] | None = None
    detection: Detection2D | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class LabeledScene:
    """Multimodal scene representation used throughout the pipeline."""

    scene_id: str
    intrinsics: CameraIntrinsics
    lidar_to_camera: Extrinsics
    point_cloud: LidarPointCloud
    objects: list[SceneObject] = field(default_factory=list)
    image_path: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ProjectedPoint:
    """Projected LiDAR point with pixel coordinates and depth."""

    pixel_xy: tuple[float, float]
    depth: float
    point_xyz_lidar: tuple[float, float, float]

