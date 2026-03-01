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
    mean_consistency_score: float | None = None


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


def _consistency_score(pair: QAPair, scene: LabeledScene) -> float:
    scene_tokens = {
        scene_object.object_id.lower()
        for scene_object in scene.objects
    } | {
        scene_object.label.lower()
        for scene_object in scene.objects
    }
    content_tokens = set(_normalize(" ".join([pair.question, pair.answer, pair.rationale])).split())
    if not scene_tokens:
        return 0.0
    overlap = len(scene_tokens & content_tokens)
    score = overlap / len(scene_tokens)
    if pair.question_type in {"grounding", "spatial-relation", "counting"}:
        score += 0.1
    return min(score, 1.0)


def filter_generated_pairs(
    record: QAGenerationRecord,
    scene: LabeledScene,
) -> FilteredGenerationRecord:
    """Drop empty, duplicate, or ungrounded QA pairs."""

    kept_pairs: list[QAPair] = []
    rejected_pairs: list[QAPair] = []
    rejection_reasons: list[str] = []
    seen_questions: set[str] = set()
    kept_scores: list[float] = []

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
        score = _consistency_score(pair, scene)
        if score < 0.1:
            rejected_pairs.append(pair)
            rejection_reasons.append("low-consistency-score")
            continue

        kept_pairs.append(pair)
        seen_questions.add(normalized_question)
        kept_scores.append(score)

    return FilteredGenerationRecord(
        scene_id=record.scene_id,
        kept_pairs=kept_pairs,
        rejected_pairs=rejected_pairs,
        rejection_reasons=rejection_reasons,
        mean_consistency_score=(sum(kept_scores) / len(kept_scores)) if kept_scores else None,
    )
