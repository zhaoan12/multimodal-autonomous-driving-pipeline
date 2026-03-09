"""Aggregate reporting helpers for dataset-scale experiments."""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(slots=True)
class ExperimentReport:
    """Serializable multi-stage experiment summary."""

    scene_count: int
    mean_match_rate: float | None
    mean_grounding_rate: float | None
    mean_consistency_score: float | None
    mean_visible_point_ratio: float | None

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def compile_experiment_report(
    analytics_reports: list[dict[str, object]],
    validation_reports: list[dict[str, object]],
    qa_reports: list[dict[str, object]],
) -> ExperimentReport:
    """Aggregate scene-level outputs into one experiment summary."""

    def _mean(values: list[float]) -> float | None:
        return (sum(values) / len(values)) if values else None

    match_rates = [float(report["match_rate"]) for report in validation_reports]
    grounding_rates = [float(report["grounding_rate"]) for report in validation_reports]
    consistency_scores = [
        float(report["mean_consistency_score"])
        for report in qa_reports
        if report.get("mean_consistency_score") is not None
    ]
    visible_ratios = [float(report["visible_point_ratio"]) for report in analytics_reports]

    return ExperimentReport(
        scene_count=max(len(analytics_reports), len(validation_reports), len(qa_reports)),
        mean_match_rate=_mean(match_rates),
        mean_grounding_rate=_mean(grounding_rates),
        mean_consistency_score=_mean(consistency_scores),
        mean_visible_point_ratio=_mean(visible_ratios),
    )


def render_experiment_report_markdown(report: ExperimentReport) -> str:
    """Render a compact markdown summary for GitHub-friendly reporting."""

    return "\n".join(
        [
            "# Experiment Summary",
            "",
            f"- Scene count: {report.scene_count}",
            f"- Mean match rate: {report.mean_match_rate}",
            f"- Mean grounding rate: {report.mean_grounding_rate}",
            f"- Mean QA consistency score: {report.mean_consistency_score}",
            f"- Mean visible point ratio: {report.mean_visible_point_ratio}",
        ]
    )
