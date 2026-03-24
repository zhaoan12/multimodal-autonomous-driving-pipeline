"""Question-answer generation and filtering modules."""

from mmdrive_pipeline.qa.generator import (
    MockQAGeneratorBackend,
    generate_scene_qa,
)
from mmdrive_pipeline.qa.filtering import (
    FilteredGenerationRecord,
    filter_generated_pairs,
)
from mmdrive_pipeline.qa.schemas import QAGenerationRecord, QAPair
from mmdrive_pipeline.qa.templates import DEFAULT_QA_TEMPLATE, render_scene_prompt

__all__ = [
    "DEFAULT_QA_TEMPLATE",
    "FilteredGenerationRecord",
    "MockQAGeneratorBackend",
    "QAGenerationRecord",
    "QAPair",
    "filter_generated_pairs",
    "generate_scene_qa",
    "render_scene_prompt",
]
