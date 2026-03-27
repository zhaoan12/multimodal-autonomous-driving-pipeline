"""Multimodal autonomous driving pipeline package."""

from mmdrive_pipeline.pipeline import (
    run_qa_generation_pipeline,
    run_validation_pipeline,
)

__all__ = [
    "__version__",
    "run_qa_generation_pipeline",
    "run_validation_pipeline",
]

__version__ = "0.1.0"
