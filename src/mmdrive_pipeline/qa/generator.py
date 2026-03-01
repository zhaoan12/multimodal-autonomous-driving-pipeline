"""LLM-oriented QA generation pipeline with a mock offline backend."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from mmdrive_pipeline.data.models import LabeledScene
from mmdrive_pipeline.qa.context import build_scene_context
from mmdrive_pipeline.qa.schemas import QAGenerationRecord, QAPair
from mmdrive_pipeline.qa.templates import render_scene_prompt


class QAGeneratorBackend(Protocol):
    """Backend interface for text generation providers."""

    provider_name: str

    def generate(self, scene: LabeledScene, prompt: str, num_pairs: int) -> list[QAPair]:
        """Generate question-answer pairs for one scene."""


@dataclass(slots=True)
class MockQAGeneratorBackend:
    """Deterministic offline generator for demos, tests, and placeholders."""

    provider_name: str = "mock"

    def generate(self, scene: LabeledScene, prompt: str, num_pairs: int) -> list[QAPair]:
        pairs: list[QAPair] = []
        context = build_scene_context(scene)
        relation_index = {relation.object_id: relation for relation in context.relations}
        question_types = ["grounding", "spatial-relation", "counting"]
        for index, scene_object in enumerate(scene.objects[:num_pairs], start=1):
            question_type = question_types[(index - 1) % len(question_types)]
            if question_type == "spatial-relation":
                relation = relation_index[scene_object.object_id]
                target_id = relation.nearest_object_id or scene_object.object_id
                question = f"Which object is closest to {scene_object.object_id}?"
                answer = f"The closest object to {scene_object.object_id} is {target_id}."
                rationale = (
                    f"Nearest-neighbor scene context links {scene_object.object_id} to {target_id}."
                )
            elif question_type == "counting":
                label_count = len([obj for obj in scene.objects if obj.label == scene_object.label])
                question = f"How many {scene_object.label} objects are annotated in {scene.scene_id}?"
                answer = f"There are {label_count} {scene_object.label} objects annotated in the scene."
                rationale = (
                    f"Scene labels contain {label_count} instances of class {scene_object.label}."
                )
            else:
                question = (
                    f"What is the approximate location of the {scene_object.label} "
                    f"labeled {scene_object.object_id}?"
                )
                answer = (
                    f"The {scene_object.label} is near {scene_object.position_xyz} in the LiDAR frame."
                )
                rationale = (
                    f"Scene annotations list {scene_object.object_id} as a {scene_object.label} "
                    f"at {scene_object.position_xyz}."
                )
            pairs.append(
                QAPair(
                    question=question,
                    answer=answer,
                    rationale=rationale,
                    question_type=question_type,
                    difficulty="basic" if question_type == "grounding" else "intermediate",
                    metadata={"rank": index, "object_id": scene_object.object_id},
                )
            )
        return pairs


def generate_scene_qa(
    scene: LabeledScene,
    backend: QAGeneratorBackend | None = None,
    num_pairs: int = 5,
) -> QAGenerationRecord:
    """Generate QA pairs for one scene."""

    backend = backend or MockQAGeneratorBackend()
    prompt = render_scene_prompt(scene, num_pairs=num_pairs)
    pairs = backend.generate(scene=scene, prompt=prompt, num_pairs=num_pairs)
    return QAGenerationRecord(
        scene_id=scene.scene_id,
        prompt=prompt,
        provider=backend.provider_name,
        pairs=pairs,
    )
