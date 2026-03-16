# Multimodal Autonomous Driving Pipeline

Research-style Python project for building and validating a multimodal autonomous driving workflow that combines LiDAR geometry, camera detections, labeled scene annotations, and large-scale QA generation.

## What this repository does

This repository implements a compact but runnable end-to-end pipeline for:

1. projecting LiDAR points into camera image coordinates using camera intrinsics and LiDAR-to-camera extrinsics,
2. relating 2D image detections to real-world coordinates using projected LiDAR support points,
3. validating grounded detections against labeled driving-scene annotations,
4. analyzing scene structure, visibility statistics, and calibration health,
5. generating scene-grounded question-answer pairs from labeled scenes using prompt templates,
6. filtering and scoring QA generations for consistency,
7. producing dataset splits and experiment-scale benchmark summaries from reproducible scripts.

The included sample data is synthetic and intentionally small. It exists to make the repository executable without external downloads. Replace it with real labeled autonomous driving scenes for research usage.

## Repository structure

```text
.
├── configs/
│   ├── calibration.example.json
│   ├── calibration_check.example.json
│   ├── benchmark.example.json
│   ├── qa_generation.example.json
│   ├── split_generation.example.json
│   └── validation.example.json
├── data/
│   └── sample/
│       ├── scene_0001.json
│       └── scene_manifest.json
├── scripts/
│   ├── check_calibration.py
│   ├── create_split.py
│   ├── generate_qa.py
│   ├── run_benchmark.py
│   └── validate_projection.py
├── src/
│   └── mmdrive_pipeline/
│       ├── analytics/
│       ├── data/
│       ├── geometry/
│       ├── mapping/
│       ├── qa/
│       ├── reporting/
│       ├── validation/
│       └── pipeline.py
└── tests/
```

## Core modules

- `mmdrive_pipeline.geometry`
  - calibration loading, sanity checks, rigid transforms, and projection diagnostics
- `mmdrive_pipeline.mapping`
  - bounding-box to 3D grounding using projected LiDAR support points with support-density and depth-dispersion metrics
- `mmdrive_pipeline.validation`
  - comparison of grounded detections against labeled object positions with failure breakdowns
- `mmdrive_pipeline.analytics`
  - scene-level metrics such as point visibility, detection coverage, and label histograms
- `mmdrive_pipeline.qa`
  - relational prompt context, structured QA records, mock generation backend, and output filtering/scoring
- `mmdrive_pipeline.reporting`
  - aggregate experiment summaries and markdown benchmark rendering
- `mmdrive_pipeline.pipeline`
  - high-level orchestration entrypoints used by both scripts and library consumers

## Installation

Python 3.10+ is recommended.

```bash
python -m venv .venv
.venv\Scripts\activate
python -m pip install -e .
```

This project intentionally uses only the Python standard library at runtime. No external dataset or model dependency is required for the sample workflow.

## Data format

Each labeled scene JSON contains:

- `camera_intrinsics`
- `lidar_to_camera`
- `point_cloud`
- `objects`
- per-object optional `detection`

Each object stores a world or LiDAR-frame position label plus an associated image-space detection when available.

## Example workflow

Run projection validation:

```bash
python scripts/validate_projection.py
```

Run calibration diagnostics:

```bash
python scripts/check_calibration.py
```

Create a deterministic train/val/test split:

```bash
python scripts/create_split.py
```

Run QA generation:

```bash
python scripts/generate_qa.py
```

Run the aggregate benchmark:

```bash
python scripts/run_benchmark.py
```

Run tests:

```bash
python -m unittest discover -s tests -v
```

Generated outputs are written to:

- `data/generated/calibration_diagnostics.json`
- `data/generated/dataset_split.json`
- `data/generated/benchmark_report.json`
- `data/generated/benchmark_report.md`
- `data/generated/validation_report.json`
- `data/generated/qa_pairs.json`

## Library usage

Validation:

```python
from mmdrive_pipeline import run_validation_pipeline

reports = run_validation_pipeline(
    "data/sample/scene_manifest.json",
    distance_tolerance_m=2.0,
    min_support_points=2,
)
```

Analytics:

```python
from mmdrive_pipeline import run_scene_analysis_pipeline

metrics = run_scene_analysis_pipeline("data/sample/scene_manifest.json")
```

Experiment report:

```python
from mmdrive_pipeline import run_experiment_report_pipeline

summary = run_experiment_report_pipeline(
    "data/sample/scene_manifest.json",
    min_support_points=2,
    num_pairs=3,
)
```

QA generation:

```python
from mmdrive_pipeline import run_qa_generation_pipeline

records = run_qa_generation_pipeline(
    "data/sample/scene_manifest.json",
    num_pairs=3,
    filter_output=True,
)
```

## Method summary

### 1. Projection

LiDAR points are transformed into the camera frame with the provided extrinsics and then projected with a pinhole camera model:

- `X_camera = R * X_lidar + t`
- `u = fx * X / Z + cx`
- `v = fy * Y / Z + cy`

Points behind the camera or outside the image bounds are filtered out.

### 2. Detection-to-world grounding

For each image detection, the pipeline finds projected LiDAR points that fall inside the detection bounding box. If enough support points exist, it uses the median LiDAR and camera coordinates as a robust representative object location.

### 3. Validation

Grounded detections are matched back to labeled scene objects and scored using Euclidean distance against annotated positions. The sample validation report includes:

- match rate
- grounding rate
- mean distance error
- failure breakdown by category
- per-object support counts

### 4. QA generation

The repository renders a scene-grounded prompt from annotation metadata, object lists, and nearest-neighbor scene relations. A mock backend is included for reproducible offline execution and emits multiple QA types including grounding, spatial-relation, and counting questions. In a real deployment, replace that backend with an API-backed LLM client.

### 5. Filtering

Generated QA pairs are filtered for:

- empty questions or answers
- duplicate questions
- outputs not grounded in scene object labels or IDs
- low lexical consistency with scene context

### 6. Experiment reporting

The benchmark workflow composes analytics, validation, and QA filtering into an aggregate report that can be saved as both JSON and markdown. This makes it straightforward to compare runs across prompts, datasets, or calibration variants.

## Reproducible research workflow

One reasonable iteration loop for this repository is:

1. inspect calibration health with `python scripts/check_calibration.py`,
2. inspect dataset composition and visibility through `run_scene_analysis_pipeline`,
3. create deterministic splits with `python scripts/create_split.py`,
4. run projection validation and QA generation baselines,
5. execute `python scripts/run_benchmark.py` to capture a compact experiment summary.

## Replacing placeholders with real research assets

To adapt this repository for a real project:

1. replace `data/sample` with real labeled scenes,
2. extend the dataset loader for the exact schema used by your dataset,
3. plug in calibrated camera and LiDAR parameters from your platform,
4. replace the mock QA backend with a real LLM provider,
5. strengthen filtering with additional semantic or model-based checks,
6. add dataset-scale evaluation and experiment tracking.

## Limitations

- The bundled sample scene is synthetic and tiny.
- The QA backend is deterministic and mock-only.
- Object matching currently assumes one detection per labeled object in the sample schema.
- Occlusion reasoning and multi-camera fusion are out of scope for this baseline repository.
- Dataset splitting is deterministic by path order and should be replaced by sequence-aware logic for real temporal datasets.

## Development notes

- Scripts are thin wrappers around package entrypoints.
- Tests use `unittest` so they run in minimal environments.
- JSON configs are provided for runnable examples.
- YAML loading remains supported only when `PyYAML` is installed separately.
- The generated benchmark markdown is intended for quick experiment logging in a repository or lab notebook.

## License

MIT
