"""Heuristic filtering for generated QA pairs."""

from __future__ import annotations

from dataclasses import dataclass

from mmdrive_pipeline.data.models import LabeledScene
from mmdrive_pipeline.qa.schemas import QAGenerationRecord, QAPair


@dataclass(slots=True)
class FilteredGenerationRecord:
    """Result of filtering a scene generation record."""

    scene_id: str
    kept_pairs: list[QAPair]
    rejected_pairs: list[QAPair]
    rejection_reasons: list[str]


def _normalize(text: str) -> str:
    return " ".join(text.lower().split())


def _contains_scene_reference(pair: QAPair, scene: LabeledScene) -> bool:
    scene_tokens = {
        scene_object.object_id.lower()
        for scene_object in scene.objects
    } | {
        scene_object.label.lower()
        for scene_object in scene.objects
    }
    content = _normalize(" ".join([pair.question, pair.answer, pair.rationale]))
    return any(token in content for token in scene_tokens)


def filter_generated_pairs(
    record: QAGenerationRecord,
    scene: LabeledScene,
) -> FilteredGenerationRecord:
    """Drop empty, duplicate, or ungrounded QA pairs."""

    kept_pairs: list[QAPair] = []
    rejected_pairs: list[QAPair] = []
    rejection_reasons: list[str] = []
    seen_questions: set[str] = set()

    for pair in record.pairs:
        normalized_question = _normalize(pair.question)
        if not normalized_question or not _normalize(pair.answer):
            rejected_pairs.append(pair)
            rejection_reasons.append("empty-question-or-answer")
            continue
        if normalized_question in seen_questions:
            rejected_pairs.append(pair)
            rejection_reasons.append("duplicate-question")
            continue
        if not _contains_scene_reference(pair, scene):
            rejected_pairs.append(pair)
            rejection_reasons.append("not-grounded-in-scene")
            continue

        kept_pairs.append(pair)
        seen_questions.add(normalized_question)

    return FilteredGenerationRecord(
        scene_id=record.scene_id,
        kept_pairs=kept_pairs,
        rejected_pairs=rejected_pairs,
        rejection_reasons=rejection_reasons,
    )

