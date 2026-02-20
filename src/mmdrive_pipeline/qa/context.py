"""Structured scene context used by prompt builders and QA validators."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from math import dist

from mmdrive_pipeline.data.models import LabeledScene


@dataclass(slots=True)
class ObjectRelation:
    """Nearest-neighbor relation for one scene object."""

    object_id: str
    nearest_object_id: str | None
    nearest_distance_m: float | None


@dataclass(slots=True)
class SceneContext:
    """Compact relational summary used for QA generation."""

    scene_id: str
    relations: list[ObjectRelation]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def build_scene_context(scene: LabeledScene) -> SceneContext:
    """Build nearest-object relations for every labeled object."""

    relations: list[ObjectRelation] = []
    for scene_object in scene.objects:
        nearest_object_id = None
        nearest_distance = None
        for other_object in scene.objects:
            if other_object.object_id == scene_object.object_id:
                continue
            candidate_distance = dist(scene_object.position_xyz, other_object.position_xyz)
            if nearest_distance is None or candidate_distance < nearest_distance:
                nearest_distance = candidate_distance
                nearest_object_id = other_object.object_id
        relations.append(
            ObjectRelation(
                object_id=scene_object.object_id,
                nearest_object_id=nearest_object_id,
                nearest_distance_m=nearest_distance,
            )
        )
    return SceneContext(scene_id=scene.scene_id, relations=relations)
