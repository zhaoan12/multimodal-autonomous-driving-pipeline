"""Geometry and calibration utilities."""

from mmdrive_pipeline.geometry.calibration import (
    CalibrationBundle,
    CalibrationDiagnostics,
    diagnose_calibration,
    load_calibration,
    validate_rotation_matrix,
)
from mmdrive_pipeline.geometry.diagnostics import (
    ProjectionDiagnostics,
    compute_reprojection_residuals,
    summarize_projection,
)
from mmdrive_pipeline.geometry.projection import (
    ProjectionResult,
    project_lidar_to_image,
    project_points_xyz,
)
from mmdrive_pipeline.geometry.transforms import invert_extrinsics, transform_points

__all__ = [
    "CalibrationBundle",
    "CalibrationDiagnostics",
    "ProjectionDiagnostics",
    "ProjectionResult",
    "compute_reprojection_residuals",
    "diagnose_calibration",
    "invert_extrinsics",
    "load_calibration",
    "project_lidar_to_image",
    "project_points_xyz",
    "summarize_projection",
    "transform_points",
    "validate_rotation_matrix",
]
