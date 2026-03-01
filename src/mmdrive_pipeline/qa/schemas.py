"""Schemas for QA generation from driving scenes."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(slots=True)
class QAPair:
    """One generated question-answer pair."""

    question: str
    answer: str
    rationale: str
    question_type: str = "grounding"
    difficulty: str = "basic"
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class QAGenerationRecord:
    """Generated samples and prompt context for a scene."""

    scene_id: str
    prompt: str
    provider: str
    pairs: list[QAPair]

    def to_dict(self) -> dict[str, Any]:
        return {
            "scene_id": self.scene_id,
            "prompt": self.prompt,
            "provider": self.provider,
            "pairs": [pair.to_dict() for pair in self.pairs],
        }
