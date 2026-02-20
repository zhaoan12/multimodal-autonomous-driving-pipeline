"""Prompt templates for scene-to-QA generation."""

from __future__ import annotations

from textwrap import dedent

from mmdrive_pipeline.data.models import LabeledScene
from mmdrive_pipeline.qa.context import build_scene_context


DEFAULT_QA_TEMPLATE = dedent(
    """
    You are generating grounded driving-scene question-answer pairs.
    Use only the scene facts provided below.
    Avoid speculation and avoid referring to hidden sensor states.

    Scene ID: {scene_id}
    Weather: {weather}
    Time of day: {time_of_day}
    Objects:
    {object_lines}
    Spatial relations:
    {relation_lines}

    Generate {num_pairs} diverse QA pairs.
    Each pair must include:
    - a concise question
    - a factual answer
    - a short rationale linked to scene evidence
    """
).strip()


def render_scene_prompt(scene: LabeledScene, num_pairs: int, template: str = DEFAULT_QA_TEMPLATE) -> str:
    """Render a generation prompt from a labeled scene."""

    object_lines = "\n".join(
        f"- {scene_object.object_id}: {scene_object.label} at {scene_object.position_xyz}"
        for scene_object in scene.objects
    )
    context = build_scene_context(scene)
    relation_lines = "\n".join(
        (
            f"- {relation.object_id} nearest {relation.nearest_object_id} "
            f"at {relation.nearest_distance_m:.2f} m"
        )
        if relation.nearest_object_id is not None and relation.nearest_distance_m is not None
        else f"- {relation.object_id} has no neighboring objects"
        for relation in context.relations
    )
    return template.format(
        scene_id=scene.scene_id,
        weather=scene.metadata.get("weather", "unknown"),
        time_of_day=scene.metadata.get("time_of_day", "unknown"),
        object_lines=object_lines or "- none",
        relation_lines=relation_lines or "- none",
        num_pairs=num_pairs,
    )
