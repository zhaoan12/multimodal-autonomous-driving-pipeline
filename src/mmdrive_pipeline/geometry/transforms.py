"""Rigid transform helpers for LiDAR-camera geometry."""

from __future__ import annotations

import numpy as np

from mmdrive_pipeline.data.models import Extrinsics


def as_homogeneous(points_xyz: np.ndarray) -> np.ndarray:
    """Append a homogeneous coordinate to an ``Nx3`` point array."""

    if points_xyz.ndim != 2 or points_xyz.shape[1] != 3:
        raise ValueError("Expected points_xyz with shape (N, 3).")
    ones = np.ones((points_xyz.shape[0], 1), dtype=float)
    return np.hstack([points_xyz.astype(float), ones])


def transform_points(points_xyz: np.ndarray, extrinsics: Extrinsics) -> np.ndarray:
    """Transform points with a rigid rotation and translation."""

    homogeneous_points = as_homogeneous(points_xyz)
    transformed = homogeneous_points @ extrinsics.transform_matrix().T
    return transformed[:, :3]


def invert_extrinsics(extrinsics: Extrinsics) -> Extrinsics:
    """Invert a rigid transform."""

    rotation_inv = extrinsics.rotation.T
    translation_inv = -rotation_inv @ extrinsics.translation
    return Extrinsics(
        rotation=rotation_inv,
        translation=translation_inv,
        source_frame=extrinsics.target_frame,
        target_frame=extrinsics.source_frame,
    )

