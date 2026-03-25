"""Rigid transform helpers for LiDAR-camera geometry."""

from __future__ import annotations

from mmdrive_pipeline.data.models import Extrinsics


def _mat_vec_mul(matrix: tuple[tuple[float, float, float], ...], vector: tuple[float, float, float]) -> tuple[float, float, float]:
    return tuple(
        sum(matrix[row][col] * vector[col] for col in range(3))
        for row in range(3)
    )


def _transpose(matrix: tuple[tuple[float, float, float], ...]) -> tuple[tuple[float, float, float], ...]:
    return tuple(tuple(matrix[row][col] for row in range(3)) for col in range(3))


def transform_points(
    points_xyz: list[tuple[float, float, float]],
    extrinsics: Extrinsics,
) -> list[tuple[float, float, float]]:
    """Transform points with a rigid rotation and translation."""

    transformed = []
    for point in points_xyz:
        rotated = _mat_vec_mul(extrinsics.rotation, point)
        transformed.append(
            tuple(rotated[index] + extrinsics.translation[index] for index in range(3))
        )
    return transformed


def invert_extrinsics(extrinsics: Extrinsics) -> Extrinsics:
    """Invert a rigid transform."""

    rotation_inv = _transpose(extrinsics.rotation)
    translation_inv_rot = _mat_vec_mul(rotation_inv, extrinsics.translation)
    translation_inv = tuple(-value for value in translation_inv_rot)
    return Extrinsics(
        rotation=rotation_inv,
        translation=translation_inv,
        source_frame=extrinsics.target_frame,
        target_frame=extrinsics.source_frame,
    )
