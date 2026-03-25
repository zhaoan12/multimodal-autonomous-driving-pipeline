"""Calibration loading and validation helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from mmdrive_pipeline.data.models import CameraIntrinsics, Extrinsics
from mmdrive_pipeline.utils.io import read_yaml


@dataclass(slots=True)
class CalibrationBundle:
    """Coupled intrinsics and extrinsics for one camera-LiDAR pair."""

    intrinsics: CameraIntrinsics
    lidar_to_camera: Extrinsics


def validate_rotation_matrix(
    rotation: tuple[tuple[float, float, float], ...],
    atol: float = 1e-5,
) -> None:
    """Validate that a matrix is close to a proper 3D rotation."""

    if len(rotation) != 3 or any(len(row) != 3 for row in rotation):
        raise ValueError("Rotation matrix must have shape (3, 3).")
    columns = [tuple(rotation[row][col] for row in range(3)) for col in range(3)]
    for index, column in enumerate(columns):
        norm = sum(value * value for value in column)
        if abs(norm - 1.0) > atol:
            raise ValueError("Rotation matrix is not orthonormal.")
        for other_column in columns[index + 1 :]:
            dot = sum(a * b for a, b in zip(column, other_column, strict=True))
            if abs(dot) > atol:
                raise ValueError("Rotation matrix is not orthonormal.")

    determinant = (
        rotation[0][0] * (rotation[1][1] * rotation[2][2] - rotation[1][2] * rotation[2][1])
        - rotation[0][1] * (rotation[1][0] * rotation[2][2] - rotation[1][2] * rotation[2][0])
        + rotation[0][2] * (rotation[1][0] * rotation[2][1] - rotation[1][1] * rotation[2][0])
    )
    if abs(determinant - 1.0) > atol:
        raise ValueError("Rotation matrix determinant must be 1.")


def load_calibration(path: str | Path) -> CalibrationBundle:
    """Load camera intrinsics and LiDAR extrinsics from YAML."""

    config = read_yaml(path)
    intrinsics_cfg = config["camera_intrinsics"]
    extrinsics_cfg = config["lidar_to_camera"]

    intrinsics = CameraIntrinsics(
        fx=float(intrinsics_cfg["fx"]),
        fy=float(intrinsics_cfg["fy"]),
        cx=float(intrinsics_cfg["cx"]),
        cy=float(intrinsics_cfg["cy"]),
        width=int(intrinsics_cfg["width"]),
        height=int(intrinsics_cfg["height"]),
    )

    rotation = tuple(
        tuple(float(value) for value in row)
        for row in extrinsics_cfg["rotation"]
    )
    translation = tuple(float(value) for value in extrinsics_cfg["translation"])
    validate_rotation_matrix(rotation)
    if len(translation) != 3:
        raise ValueError("Extrinsic translation must have shape (3,).")

    extrinsics = Extrinsics(
        rotation=rotation,
        translation=translation,
        source_frame=extrinsics_cfg.get("source_frame", "lidar"),
        target_frame=extrinsics_cfg.get("target_frame", "camera"),
    )
    return CalibrationBundle(intrinsics=intrinsics, lidar_to_camera=extrinsics)
