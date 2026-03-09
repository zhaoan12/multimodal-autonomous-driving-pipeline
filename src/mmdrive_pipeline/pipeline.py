"""High-level orchestration helpers for analytics, validation, and QA generation."""

from __future__ import annotations

from mmdrive_pipeline.analytics import summarize_dataset
from mmdrive_pipeline.data import load_scene_collection
from mmdrive_pipeline.qa import filter_generated_pairs, generate_scene_qa
from mmdrive_pipeline.reporting import compile_experiment_report
from mmdrive_pipeline.validation import validate_scene_projection


def run_scene_analysis_pipeline(dataset_manifest: str) -> list[dict[str, object]]:
    """Run dataset analytics over all scenes in a manifest."""

    scenes = load_scene_collection(dataset_manifest)
    return [metrics.to_dict() for metrics in summarize_dataset(scenes)]


def run_experiment_report_pipeline(
    dataset_manifest: str,
    distance_tolerance_m: float = 2.0,
    min_support_points: int = 3,
    num_pairs: int = 5,
) -> dict[str, object]:
    """Run all primary pipelines and compile an aggregate report."""

    analytics_reports = run_scene_analysis_pipeline(dataset_manifest)
    validation_reports = run_validation_pipeline(
        dataset_manifest=dataset_manifest,
        distance_tolerance_m=distance_tolerance_m,
        min_support_points=min_support_points,
    )
    qa_reports = run_qa_generation_pipeline(
        dataset_manifest=dataset_manifest,
        num_pairs=num_pairs,
        filter_output=True,
    )
    return compile_experiment_report(
        analytics_reports=analytics_reports,
        validation_reports=validation_reports,
        qa_reports=qa_reports,
    ).to_dict()


def run_validation_pipeline(
    dataset_manifest: str,
    distance_tolerance_m: float = 2.0,
    min_support_points: int = 3,
) -> list[dict[str, object]]:
    """Run projection validation over all scenes in a manifest."""

    scenes = load_scene_collection(dataset_manifest)
    reports: list[dict[str, object]] = []
    for scene in scenes:
        report = validate_scene_projection(
            scene,
            distance_tolerance_m=distance_tolerance_m,
            min_support_points=min_support_points,
        )
        reports.append(
            {
                "scene_id": report.scene_id,
                "match_rate": report.match_rate,
                "mean_distance_error": report.mean_distance_error,
                "grounding_rate": report.grounding_rate,
                "failure_breakdown": report.failure_breakdown,
                "matched_objects": report.matched_objects,
                "total_objects": report.total_objects,
                "results": [
                    {
                        "object_id": item.object_id,
                        "label": item.label,
                        "matched": item.matched,
                        "distance_error": item.distance_error,
                        "support_count": item.support_count,
                        "failure_reason": item.failure_reason,
                    }
                    for item in report.results
                ],
            }
        )
    return reports


def run_qa_generation_pipeline(
    dataset_manifest: str,
    num_pairs: int = 5,
    filter_output: bool = True,
) -> list[dict[str, object]]:
    """Run QA generation over all scenes in a manifest."""

    scenes = load_scene_collection(dataset_manifest)
    payload: list[dict[str, object]] = []
    for scene in scenes:
        record = generate_scene_qa(scene, num_pairs=num_pairs)
        item = record.to_dict()
        if filter_output:
            filtered = filter_generated_pairs(record, scene)
            item["filtered_pairs"] = [pair.to_dict() for pair in filtered.kept_pairs]
            item["rejected_pairs"] = [pair.to_dict() for pair in filtered.rejected_pairs]
            item["rejection_reasons"] = filtered.rejection_reasons
            item["mean_consistency_score"] = filtered.mean_consistency_score
        payload.append(item)
    return payload
