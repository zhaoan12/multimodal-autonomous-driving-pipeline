"""LLM-oriented QA generation pipeline with a mock offline backend."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from mmdrive_pipeline.data.models import LabeledScene
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
        for index, scene_object in enumerate(scene.objects[:num_pairs], start=1):
            question = f"What is the approximate location of the {scene_object.label} labeled {scene_object.object_id}?"
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

