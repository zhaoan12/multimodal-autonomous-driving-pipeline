"""Calibration loading and validation helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from mmdrive_pipeline.data.models import CameraIntrinsics, Extrinsics
from mmdrive_pipeline.utils.io import read_yaml


@dataclass(slots=True)
class CalibrationBundle:
    """Coupled intrinsics and extrinsics for one camera-LiDAR pair."""

    intrinsics: CameraIntrinsics
    lidar_to_camera: Extrinsics


def validate_rotation_matrix(rotation: np.ndarray, atol: float = 1e-5) -> None:
    """Validate that a matrix is close to a proper 3D rotation."""

    if rotation.shape != (3, 3):
        raise ValueError("Rotation matrix must have shape (3, 3).")
    identity = np.eye(3, dtype=float)
    if not np.allclose(rotation @ rotation.T, identity, atol=atol):
        raise ValueError("Rotation matrix is not orthonormal.")
    if not np.isclose(np.linalg.det(rotation), 1.0, atol=atol):
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

    rotation = np.array(extrinsics_cfg["rotation"], dtype=float)
    translation = np.array(extrinsics_cfg["translation"], dtype=float)
    validate_rotation_matrix(rotation)
    if translation.shape != (3,):
        raise ValueError("Extrinsic translation must have shape (3,).")

    extrinsics = Extrinsics(
        rotation=rotation,
        translation=translation,
        source_frame=extrinsics_cfg.get("source_frame", "lidar"),
        target_frame=extrinsics_cfg.get("target_frame", "camera"),
    )
    return CalibrationBundle(intrinsics=intrinsics, lidar_to_camera=extrinsics)

