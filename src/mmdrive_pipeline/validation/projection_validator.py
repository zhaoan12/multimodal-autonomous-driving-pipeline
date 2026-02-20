"""Validation for projected detections against labeled scene annotations."""

from __future__ import annotations

from dataclasses import dataclass
from math import dist

from mmdrive_pipeline.data.models import LabeledScene
from mmdrive_pipeline.mapping.detections import DetectionMapping, map_scene_detections


@dataclass(slots=True)
class ObjectValidationResult:
    """Validation outcome for a single labeled object."""

    object_id: str
    label: str
    matched: bool
    distance_error: float | None
    support_count: int
    failure_reason: str | None = None


@dataclass(slots=True)
class SceneValidationReport:
    """Aggregate validation report for a scene."""

    scene_id: str
    matched_objects: int
    total_objects: int
    mean_distance_error: float | None
    grounding_rate: float
    failure_breakdown: dict[str, int]
    results: list[ObjectValidationResult]

    @property
    def match_rate(self) -> float:
        if self.total_objects == 0:
            return 0.0
        return self.matched_objects / self.total_objects


def _match_mapping(scene: LabeledScene, mapping: DetectionMapping) -> tuple[str, str, tuple[float, float, float]] | None:
    for scene_object in scene.objects:
        if scene_object.detection is None:
            continue
        if scene_object.detection is mapping.detection:
            return scene_object.object_id, scene_object.label, scene_object.position_xyz
    return None


def validate_scene_projection(
    scene: LabeledScene,
    distance_tolerance_m: float = 2.0,
    min_support_points: int = 3,
) -> SceneValidationReport:
    """Validate mapped detections against labeled object positions."""

    mappings = map_scene_detections(scene, min_support_points=min_support_points)
    results: list[ObjectValidationResult] = []
    matched_objects = 0
    distance_errors: list[float] = []
    grounded_objects = 0
    failure_breakdown = {
        "not-grounded": 0,
        "distance-out-of-tolerance": 0,
    }

    mapping_index = {
        match[0]: (mapping, match[2])
        for mapping in mappings
        if (match := _match_mapping(scene, mapping)) is not None
    }

    for scene_object in scene.objects:
        if scene_object.detection is None:
            continue
        mapping_entry = mapping_index.get(scene_object.object_id)
        if mapping_entry is None:
            results.append(
                ObjectValidationResult(
                    object_id=scene_object.object_id,
                    label=scene_object.label,
                    matched=False,
                    distance_error=None,
                    support_count=0,
                    failure_reason="not-grounded",
                )
            )
            failure_breakdown["not-grounded"] += 1
            continue

        mapping, label_position = mapping_entry
        grounded_objects += 1
        error = dist(mapping.lidar_point_xyz, label_position)
        matched = error <= distance_tolerance_m
        matched_objects += int(matched)
        distance_errors.append(error)
        failure_reason = None if matched else "distance-out-of-tolerance"
        if failure_reason is not None:
            failure_breakdown[failure_reason] += 1
        results.append(
            ObjectValidationResult(
                object_id=scene_object.object_id,
                label=scene_object.label,
                matched=matched,
                distance_error=error,
                support_count=mapping.support_count,
                failure_reason=failure_reason,
            )
        )

    mean_distance_error = (
        sum(distance_errors) / len(distance_errors) if distance_errors else None
    )
    return SceneValidationReport(
        scene_id=scene.scene_id,
        matched_objects=matched_objects,
        total_objects=len([obj for obj in scene.objects if obj.detection is not None]),
        mean_distance_error=mean_distance_error,
        grounding_rate=(grounded_objects / len([obj for obj in scene.objects if obj.detection is not None]))
        if [obj for obj in scene.objects if obj.detection is not None]
        else 0.0,
        failure_breakdown=failure_breakdown,
        results=results,
    )
